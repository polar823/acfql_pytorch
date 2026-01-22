"""
PyTorch BC Training Script

Offline Behavior Cloning training with periodic evaluation.
Supports flow-based BC agents: fbc, mfbc, imfbc.
Supports both state-based (low_dim) and image-based environments.

Features:
- PyTorch DataLoader for efficient multi-process data loading
- Warmup scheduler for flow map learning
- Cosine annealing learning rate scheduler
- Performance monitoring (data_load, update, total_step)
- Wandb logging and checkpoint saving
- Support for robomimic low_dim and image environments

Reference:
- /home/xukainan/much-ado-about-noising/examples/train_robomimic.py
"""

import os
import json
import random
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set float32 matmul precision for TensorCore acceleration
torch.set_float32_matmul_precision("high")

from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_wandb_video
from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env, is_robomimic_image_env
from utils.datasets import Dataset
from utils.sequence_dataset import RandomSequenceDataset
from evaluation import evaluate
from agents import get_agent

# Set EGL device for rendering
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

# Run configuration
flags.DEFINE_string('run_group', 'BC', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'lift-mh-low_dim', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'runs/', 'Save directory.')

# Training configuration
flags.DEFINE_integer('offline_steps', 300000, 'Number of offline training steps.')
flags.DEFINE_integer('log_interval', 2500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval (-1 to disable).')

# Evaluation configuration
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 3, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# Dataset configuration
flags.DEFINE_float('dataset_proportion', 1.0, 'Proportion of dataset to use.')
flags.DEFINE_bool('sparse', False, 'Use sparse reward.')

# Scheduler configuration
flags.DEFINE_float('warmup_ratio', 0.0, 'Ratio of steps to stay at min_value (warmup phase).')
flags.DEFINE_float('rampup_ratio', 0.0, 'Ratio of steps to ramp up from min to max value.')
flags.DEFINE_float('min_delta_t', 0.0, 'Minimum delta_t value for warmup scheduler.')
flags.DEFINE_float('max_delta_t', 1.0, 'Maximum delta_t value for warmup scheduler.')
flags.DEFINE_bool('use_lr_scheduler', True, 'Use cosine annealing LR scheduler.')

# Resume configuration
flags.DEFINE_string('resume_path', '', 'Path to checkpoint to resume from.')

# Agent configuration file (supports --agent=agents/fbc.py:get_config format)
config_flags.DEFINE_config_file(
    'agent',
    'agents/fbc.py',  # Default BC agent config file
    'Agent configuration file path (e.g., agents/fbc.py, agents/mfbc.py, agents/imfbc.py).',
    lock_config=False,
)


class WarmupAnnealingScheduler:
    """Scheduler for warmup and annealing of delta_t parameter.

    Used for flow map learning where we want to start with small delta_t
    and gradually increase it during training.

    Schedule:
    1. Warmup phase (0 to warmup_steps): stay at min_value
    2. Rampup phase (warmup_steps to warmup_steps + rampup_steps): linear ramp from min to max
    3. Final phase: stay at max_value
    """

    def __init__(
        self,
        max_steps: int,
        warmup_ratio: float = 0.1,
        rampup_ratio: float = 0.8,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        """Initialize scheduler.

        Args:
            max_steps: Total number of gradient steps in training
            warmup_ratio: Ratio of steps to stay at min_value (e.g., 0.1 means first 10%)
            rampup_ratio: Ratio of steps to ramp up from min to max (e.g., 0.2 means 20%)
            min_value: Minimum value (during warmup)
            max_value: Maximum value (after rampup)
        """
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.rampup_steps = int(max_steps * rampup_ratio)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, step: int) -> float:
        """Get scheduler value at given step."""
        if step < self.warmup_steps:
            return self.min_value
        elif step < self.warmup_steps + self.rampup_steps:
            # Linear ramp from min_value to max_value
            progress = (step - self.warmup_steps) / max(self.rampup_steps, 1)
            return progress * (self.max_value - self.min_value) + self.min_value
        else:
            return self.max_value


@contextmanager
def timed(section: str, record_dict: dict):
    """Context manager for timing code sections.

    Args:
        section: Name of the section being timed
        record_dict: Dictionary to store timing results
    """
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


class LoggingHelper:
    """Helper class for logging to CSV and wandb."""

    def __init__(self, csv_loggers: Dict[str, CsvLogger], wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data: Dict[str, Any], prefix: str, step: int):
        """Log data to CSV and wandb.

        Args:
            data: Dictionary of metrics to log
            prefix: Prefix for the metrics (e.g., 'eval', 'offline_agent')
            step: Current training step
        """
        assert prefix in self.csv_loggers, f"Unknown prefix: {prefix}"
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_train_dataset(
    ds,
    env_name: str,
    dataset_proportion: float = 1.0,
    sparse: bool = False
) -> Dataset:
    """Process the training dataset.

    Handles:
    - Dataset proportion (use subset of data)
    - Sparse reward conversion
    - Robomimic reward adjustment (shift from [0,1] to [-1,0])

    Args:
        ds: Raw dataset (dict or Dataset)
        env_name: Environment name
        dataset_proportion: Proportion of dataset to use (0.0-1.0)
        sparse: Whether to convert to sparse rewards

    Returns:
        Processed Dataset object
    """
    # Convert to Dataset if dict
    if isinstance(ds, dict):
        ds = Dataset.create(**ds)
    elif isinstance(ds, Dataset):
        ds = ds.copy()

    # Apply dataset proportion
    if dataset_proportion < 1.0:
        new_size = int(ds.size * dataset_proportion)
        ds = Dataset.create(**{k: v[:new_size] for k, v in ds.items()})

    # Adjust rewards for robomimic (shift from [0,1] to [-1,0])
    if is_robomimic_env(env_name) or is_robomimic_image_env(env_name):
        penalty_rewards = ds["rewards"] - 1.0
        ds = ds.copy(add_or_replace=dict(rewards=penalty_rewards))

    # Convert to sparse reward
    if sparse:
        sparse_rewards = (ds["rewards"] != 0.0).astype(np.float32) * -1.0
        ds = ds.copy(add_or_replace=dict(rewards=sparse_rewards))

    return ds


def loop_dataloader(loader):
    """Create infinite iterator over dataloader."""
    while True:
        for batch in loader:
            yield batch


def main(_):
    # Get agent config
    config = FLAGS.agent

    # Setup experiment
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='bc_pytorch', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, 'bc_pytorch', FLAGS.run_group, FLAGS.env_name, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2)

    # Set seeds
    set_seed(FLAGS.seed)

    # Check if image-based environment
    is_image_env = is_robomimic_image_env(FLAGS.env_name)
    print(f"\n===== Environment Setup =====")
    print(f"Environment: {FLAGS.env_name}")
    print(f"Image-based: {is_image_env}")

    # Create environment and datasets
    result = make_env_and_datasets(FLAGS.env_name)
    if len(result) == 5:
        # Image-based environment returns shape_meta
        env, eval_env, train_dataset, val_dataset, shape_meta = result
        print(f"Shape meta: {shape_meta}")
    else:
        # Non-image environment
        env, eval_env, train_dataset, val_dataset = result
        shape_meta = None

    # Process dataset
    train_dataset = process_train_dataset(
        train_dataset,
        FLAGS.env_name,
        FLAGS.dataset_proportion,
        FLAGS.sparse
    )
    print(f"Dataset size: {train_dataset.size}")

    # Get example batch for agent creation
    example_batch = train_dataset.sample(1)

    # Handle dict observations (image environments) vs tensor observations
    if isinstance(example_batch['observations'], dict):
        # Image environment with dict observations - use shape_meta
        observation_shape = shape_meta
        print(f"Observation type: dict (image-based)")
        for key, meta in shape_meta['obs'].items():
            print(f"  {key}: {meta['shape']} ({meta['type']})")
    else:
        # Regular tensor observations
        observation_shape = example_batch['observations'].shape[1:]  # Remove batch dim
        print(f"Observation shape: {observation_shape}")

    action_dim = example_batch['actions'].shape[-1]
    print(f"Action dim: {action_dim}")

    # Get agent class
    agent_name = config['agent_name']
    AgentClass, _ = get_agent(agent_name)
    print(f"\n===== Agent Setup =====")
    print(f"Agent: {agent_name} ({AgentClass.__name__})")

    # Convert ml_collections.ConfigDict to dict for agent
    import ml_collections
    if isinstance(config, ml_collections.ConfigDict):
        agent_config = config.to_dict()
    else:
        agent_config = dict(config)

    # Create agent
    agent = AgentClass.create(
        observation_shape=observation_shape,
        action_dim=action_dim,
        config=agent_config,
    )

    # Setup learning rate scheduler
    lr_scheduler = None
    if FLAGS.use_lr_scheduler and hasattr(agent, 'actor_optimizer'):
        lr_scheduler = CosineAnnealingLR(
            agent.actor_optimizer,
            T_max=FLAGS.offline_steps,
        )
        print(f"LR Scheduler: CosineAnnealingLR (T_max={FLAGS.offline_steps})")

    # Setup warmup scheduler for delta_t (flow map learning)
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=FLAGS.offline_steps,
        warmup_ratio=FLAGS.warmup_ratio,
        rampup_ratio=FLAGS.rampup_ratio,
        min_value=FLAGS.min_delta_t,
        max_value=FLAGS.max_delta_t,
    )
    print(f"Warmup Scheduler: warmup_ratio={FLAGS.warmup_ratio}, rampup_ratio={FLAGS.rampup_ratio}")
    print(f"  delta_t range: [{FLAGS.min_delta_t}, {FLAGS.max_delta_t}]")

    # Resume from checkpoint if specified
    start_step = 0
    if FLAGS.resume_path and os.path.exists(FLAGS.resume_path):
        print(f"\n===== Resuming from checkpoint =====")
        print(f"Loading: {FLAGS.resume_path}")
        checkpoint = agent.load(FLAGS.resume_path)
        if isinstance(checkpoint, dict) and 'step' in checkpoint:
            start_step = checkpoint['step']
            print(f"Resuming from step {start_step}")
            # Fast-forward LR scheduler
            if lr_scheduler is not None:
                for _ in range(start_step):
                    lr_scheduler.step()

    # Setup logging
    prefixes = ["eval", "offline_agent"]
    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
                     for prefix in prefixes},
        wandb_logger=wandb,
    )

    # Create PyTorch DataLoader for efficient data loading
    print("\n===== Setting up DataLoader =====")
    use_dataloader = config.get('use_dataloader', True)

    if use_dataloader:
        # Wrap dataset for PyTorch DataLoader
        seq_dataset = RandomSequenceDataset(
            train_dataset,
            sequence_length=config['horizon_length'],
            discount=config['discount'],
            size=FLAGS.offline_steps * config['batch_size']  # Virtual size
        )

        # Determine number of workers based on observation type
        if is_image_env:
            num_workers = 8  # More workers for image data
        else:
            num_workers = 4  # Fewer workers for state data

        dataloader = torch.utils.data.DataLoader(
            seq_dataset,
            batch_size=config['batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,  # Faster CPU->GPU transfer
            persistent_workers=True,  # Keep workers alive
            drop_last=True,  # Required for consistent batch sizes (helps with compile)
        )

        # Create infinite iterator
        data_iterator = loop_dataloader(dataloader)
        print(f"DataLoader created with {num_workers} workers, batch_size={config['batch_size']}")
    else:
        data_iterator = None
        print("Using manual sampling (no DataLoader)")

    # Performance tracking
    perf_times = {
        "data_load": [],
        "to_device": [],
        "update": [],
        "total_step": [],
    }

    log_step = start_step
    start_time = time.time()

    # ===== Offline BC Training =====
    print("\n===== Starting BC Training =====")
    print(f"Total steps: {FLAGS.offline_steps}")
    print(f"Log interval: {FLAGS.log_interval}")
    print(f"Eval interval: {FLAGS.eval_interval}")

    for i in tqdm.tqdm(range(start_step + 1, FLAGS.offline_steps + 1), desc="BC Training"):
        log_step = i

        with timed("total_step", perf_times):
            # Sample batch with sequence
            with timed("data_load", perf_times):
                if use_dataloader:
                    batch = next(data_iterator)
                else:
                    batch = train_dataset.sample_sequence(
                        batch_size=config['batch_size'],
                        sequence_length=config['horizon_length'],
                        discount=config['discount']
                    )

            # Convert to torch tensors (handle nested dicts for image observations)
            with timed("to_device", perf_times):
                def to_torch(x):
                    if isinstance(x, dict):
                        return {k: to_torch(v) for k, v in x.items()}
                    elif isinstance(x, np.ndarray):
                        return torch.from_numpy(x).float()
                    elif isinstance(x, torch.Tensor):
                        return x.float()
                    else:
                        return x

                batch = {k: to_torch(v) for k, v in batch.items()}

            # Get delta_t from warmup scheduler (for flow map learning)
            delta_t = warmup_scheduler(i - 1)

            # Update agent
            with timed("update", perf_times):
                # Pass delta_t to agent if it supports it
                if hasattr(agent, 'update_with_delta_t'):
                    info = agent.update_with_delta_t(batch, delta_t)
                else:
                    info = agent.update(batch)

                # Step LR scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step()

        # Logging
        if i % FLAGS.log_interval == 0:
            # Add performance metrics to info
            if perf_times["total_step"]:
                info['perf/data_load_ms'] = np.mean(perf_times['data_load'][-FLAGS.log_interval:]) * 1000
                info['perf/to_device_ms'] = np.mean(perf_times['to_device'][-FLAGS.log_interval:]) * 1000
                info['perf/update_ms'] = np.mean(perf_times['update'][-FLAGS.log_interval:]) * 1000
                info['perf/total_step_ms'] = np.mean(perf_times['total_step'][-FLAGS.log_interval:]) * 1000
                info['perf/steps_per_sec'] = 1.0 / np.mean(perf_times['total_step'][-FLAGS.log_interval:])

            # Add scheduler info
            info['scheduler/delta_t'] = delta_t
            if lr_scheduler is not None:
                info['scheduler/lr'] = lr_scheduler.get_last_lr()[0]

            logger.log(info, "offline_agent", step=log_step)

            # Print performance summary periodically
            if i % (FLAGS.log_interval * 10) == 0 and perf_times["total_step"]:
                print(f"\n[Step {i}] Performance (last {FLAGS.log_interval} steps):")
                print(f"  Data load:  {info.get('perf/data_load_ms', 0):.2f}ms")
                print(f"  To device:  {info.get('perf/to_device_ms', 0):.2f}ms")
                print(f"  Update:     {info.get('perf/update_ms', 0):.2f}ms")
                print(f"  Total:      {info.get('perf/total_step_ms', 0):.2f}ms")
                print(f"  Steps/sec:  {info.get('perf/steps_per_sec', 0):.2f}")
                print(f"  delta_t:    {delta_t:.4f}")
                if lr_scheduler is not None:
                    print(f"  LR:         {lr_scheduler.get_last_lr()[0]:.2e}")

        # Saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_path = os.path.join(FLAGS.save_dir, f"agent_{log_step}.pt")
            agent.save(save_path)
            print(f"\n[Step {i}] Saved checkpoint to {save_path}")

        # Evaluation
        if i == FLAGS.offline_steps or (FLAGS.eval_interval > 0 and i % FLAGS.eval_interval == 0):
            print(f"\n[Step {i}] Running evaluation...")
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            if len(renders) > 0:
                eval_info['video'] = get_wandb_video(renders, fps=int(30 / max(FLAGS.video_frame_skip, 1)))
            logger.log(eval_info, "eval", step=log_step)

            eval_return = eval_info.get('return', None)
            eval_success = eval_info.get('success', None)
            if eval_return is not None:
                print(f"[Step {i}] Eval return: {eval_return:.2f}")
            if eval_success is not None:
                print(f"[Step {i}] Eval success: {eval_success:.2%}")

    # Cleanup
    end_time = time.time()
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    # Save final model
    save_path = os.path.join(FLAGS.save_dir, "agent_final.pt")
    agent.save(save_path)

    # Log timing
    print(f"\n===== Training Complete =====")
    print(f"Total time: {end_time - start_time:.1f}s")
    print(f"Final model saved to: {save_path}")

    # Save completion token
    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url if hasattr(run, 'url') else str(run))


if __name__ == '__main__':
    app.run(main)
