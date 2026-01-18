"""
PyTorch BC Training Script

Offline Behavior Cloning training with periodic evaluation.
Supports flow-based BC agents: fbc, mfbc, imfbc.
"""

import os
import json
import random
import time
import dataclasses

import torch
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_wandb_video
from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from utils.datasets import Dataset
from evaluation import evaluate
from agents import get_agent

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
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval (-1 to disable).')

# Evaluation configuration
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# Dataset configuration
flags.DEFINE_float('dataset_proportion', 1.0, 'Proportion of dataset to use.')
flags.DEFINE_bool('sparse', False, 'Use sparse reward.')

# Agent configuration file (supports --agent=agents/fbc.py:get_config format)
# Or use --agent.xxx to override specific parameters
config_flags.DEFINE_config_file(
    'agent',
    'agents/imfbc.py',  # Default BC agent config file
    'Agent configuration file path (e.g., agents/fbc.py, agents/mfbc.py, agents/imfbc.py).',
    lock_config=False,
)


class LoggingHelper:
    """Helper class for logging to CSV and wandb."""
    
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, f"Unknown prefix: {prefix}"
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_train_dataset(ds, env_name, dataset_proportion=1.0, sparse=False):
    """
    Process the training dataset.
    
    Handles:
    - Dataset proportion
    - Sparse reward conversion
    - Robomimic reward adjustment
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
    if is_robomimic_env(env_name):
        penalty_rewards = ds["rewards"] - 1.0
        ds = ds.copy(add_or_replace=dict(rewards=penalty_rewards))
    
    # Convert to sparse reward
    if sparse:
        sparse_rewards = (ds["rewards"] != 0.0).astype(np.float32) * -1.0
        ds = ds.copy(add_or_replace=dict(rewards=sparse_rewards))
    
    return ds


def main(_):
    # Get agent config
    config = FLAGS.agent
    
    # Setup experiment
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='bc_pytorch', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'bc_pytorch', FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2)
    
    # Set seeds
    set_seed(FLAGS.seed)
    
    # Create environment and datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)
    
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
    observation_shape = example_batch['observations'].shape[1:]  # Remove batch dim
    action_dim = example_batch['actions'].shape[-1]
    
    print(f"Observation shape: {observation_shape}")
    print(f"Action dim: {action_dim}")
    
    # Get agent class and config class
    agent_name = config['agent_name']
    AgentClass, ConfigClass = get_agent(agent_name)
    print(f"Using agent: {agent_name} ({AgentClass.__name__})")
    
    # Build config kwargs from ml_collections config
    # Get all fields from ConfigClass dataclass
    config_fields = {f.name for f in dataclasses.fields(ConfigClass)}
    
    # Only pass parameters that the ConfigClass accepts
    config_kwargs = {}
    for key in config.keys():
        if key in config_fields:
            config_kwargs[key] = config[key]
    
    # Create agent config
    agent_config = ConfigClass(**config_kwargs)
    
    # Create agent
    agent = AgentClass.create(
        observation_shape=observation_shape,
        action_dim=action_dim,
        config=agent_config,
    )
    
    # Setup logging
    prefixes = ["eval", "offline_agent"]
    
    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                     for prefix in prefixes},
        wandb_logger=wandb,
    )
    
    log_step = 0
    start_time = time.time()
    
    # ===== Offline BC Training =====
    print("\n===== Starting BC Training =====")
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), desc="BC Training"):
        log_step += 1
        
        # Sample batch with sequence
        batch = train_dataset.sample_sequence(
            batch_size=config['batch_size'],
            sequence_length=config['horizon_length'],
            discount=config['discount']
        )
        
        # Convert to torch tensors
        batch = {k: torch.from_numpy(v).float() for k, v in batch.items()}
        
        # Update agent
        info = agent.update(batch)
        
        # Logging
        if i % FLAGS.log_interval == 0:
            logger.log(info, "offline_agent", step=log_step)
        
        # Saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_path = os.path.join(FLAGS.save_dir, f"agent_{log_step}.pt")
            agent.save(save_path)
        
        # Evaluation
        if i == FLAGS.offline_steps or (FLAGS.eval_interval > 0 and i % FLAGS.eval_interval == 0):
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
            print(f"\n[Step {i}] Eval return: {eval_return:.2f}" if eval_return is not None else f"\n[Step {i}] Eval return: N/A")
    
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
    
    # Save completion token
    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)


if __name__ == '__main__':
    app.run(main)
