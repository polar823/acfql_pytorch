"""
Offline-to-Online Reinforcement Learning Main Script (PyTorch)

Supports:
- Offline pre-training with flow-based BC
- Online fine-tuning with Q-learning (FQL)
- Action chunking
- Visual observations (with encoder)
- Multiple environments (D4RL, Robomimic, OGBench)
"""

import os
import json
import random
import time

import numpy as np
import torch
import tqdm
import wandb
import ml_collections
from ml_collections import config_flags
from absl import app, flags

from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_wandb_video

from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.datasets import Dataset
from utils.replay_buffer import ReplayBuffer

from evaluation import evaluate
from agents import get_agent, get_agent_config

# GPU device handling
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

# Run configuration
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'square-mh-low_dim', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'runs/', 'Save directory.')

# Training configuration
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline training steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online training steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval (-1 to disable).')
flags.DEFINE_integer('start_training', 5000, 'When to start online training.')

flags.DEFINE_integer('utd_ratio', 1, 'Update-to-data ratio.')

# Evaluation configuration
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# Dataset configuration
flags.DEFINE_float('dataset_proportion', 1.0, 'Proportion of dataset to use.')
flags.DEFINE_bool('sparse', False, 'Use sparse reward.')

# Agent configuration file (supports --agent=agents/fql.py:get_config format)
# Or use --agent.xxx to override specific parameters
config_flags.DEFINE_config_file(
    'agent',
    'agents/fql.py',  # Default agent config file
    'Agent configuration file path (e.g., agents/fql.py, agents/imfbc.py).',
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


def batch_to_torch(batch):
    """Convert batch to torch tensors, handling dict observations."""
    def to_torch(x):
        if isinstance(x, dict):
            return {k: to_torch(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        else:
            return x

    return {k: to_torch(v) for k, v in batch.items()}


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
    run = setup_wandb(project='fql_pytorch', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'fql_pytorch', FLAGS.run_group, FLAGS.env_name, exp_name)
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

    # Handle dict observations (image environments) vs tensor observations
    if isinstance(example_batch['observations'], dict):
        # Image environment with dict observations - use shape_meta
        observation_shape = env.shape_meta if hasattr(env, 'shape_meta') else None
        if observation_shape is None:
            # Try to get from dataset (LazyImageDataset has shape_meta)
            if hasattr(train_dataset, 'shape_meta'):
                observation_shape = train_dataset.shape_meta
            else:
                # Fallback: use robomimic_utils to get shape_meta from dataset file
                from envs.robomimic_utils import get_shape_meta_from_dataset, _check_dataset_exists_image
                dataset_path = _check_dataset_exists_image(FLAGS.env_name)
                observation_shape = get_shape_meta_from_dataset(
                    dataset_path=dataset_path,
                    obs_keys=list(example_batch['observations'].keys())
                )
    else:
        observation_shape = example_batch['observations'].shape[1:]  # Remove batch dim

    action_dim = example_batch['actions'].shape[-1]

    print(f"Observation shape: {observation_shape}")
    print(f"Action dim: {action_dim}")
    
    # Get agent class and config class
    agent_name = config['agent_name']
    AgentClass, ConfigClass = get_agent(agent_name)
    print(f"Using agent: {agent_name} ({AgentClass.__name__})")
    
    # Build agent config
    if ConfigClass is None:
        # Agent uses dict-based config (like FQL)
        # Convert ml_collections.ConfigDict to regular dict
        agent_config = dict(config)
    else:
        # Agent uses dataclass-based config
        import dataclasses
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
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    
    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                     for prefix in prefixes},
        wandb_logger=wandb,
    )
    
    log_step = 0
    offline_init_time = time.time()
    
    # ===== Offline Training =====
    print("\n===== Starting Offline Training =====")
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), desc="Offline"):
        log_step += 1
        
        # Sample batch with sequence
        batch = train_dataset.sample_sequence(
            batch_size=config['batch_size'],
            sequence_length=config['horizon_length'],
            discount=config['discount']
        )

        # Convert to torch tensors
        batch = batch_to_torch(batch)

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
            print(f"\n[Offline Step {i}] Eval return: {eval_return:.2f}" if eval_return is not None else f"\n[Offline Step {i}] Eval return: N/A")
    
    # Save after offline training
    if FLAGS.offline_steps > 0:
        save_path = os.path.join(FLAGS.save_dir, "agent_offline_final.pt")
        agent.save(save_path)
    
    # ===== Online Training =====
    if FLAGS.online_steps > 0:
        print("\n===== Starting Online Training =====")
        
        # Create replay buffer from offline dataset
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            train_dataset,
            size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        print(f"Replay buffer initialized with {replay_buffer.size} transitions")
        
        # Reset environment
        ob, _ = env.reset()
        action_queue = []
        
        online_init_time = time.time()
        update_info = {}
        
        for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1), desc="Online"):
            log_step += 1
            
            # Sample action with action chunking
            if len(action_queue) == 0:
                action = agent.sample_actions(observations=ob)
                action_chunk = np.array(action).reshape(-1, action_dim)
                for a in action_chunk:
                    action_queue.append(a)
            
            action = action_queue.pop(0)
            
            # Environment step
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log environment info
            env_info = {k: v for k, v in info.items() if k.startswith("distance")}
            if env_info:
                logger.log(env_info, "env", step=log_step)
            
            # Adjust reward for robomimic
            if is_robomimic_env(FLAGS.env_name):
                reward = reward - 1.0
            
            # Convert to sparse reward
            if FLAGS.sparse:
                reward = (reward != 0.0) * -1.0
            
            # Add transition to buffer
            transition = dict(
                observations=ob,
                actions=action,
                rewards=reward,
                terminals=float(done),
                masks=1.0 - float(terminated),
                next_observations=next_ob,
            )
            replay_buffer.add_transition(transition)
            
            # Reset if done
            if done:
                ob, _ = env.reset()
                action_queue = []
            else:
                ob = next_ob
            
            # Training
            if i >= FLAGS.start_training:
                # Sample batch
                batch = replay_buffer.sample_sequence(
                    batch_size=config['batch_size'] * FLAGS.utd_ratio,
                    sequence_length=config['horizon_length'],
                    discount=config['discount']
                )

                # Convert to torch tensors
                batch = batch_to_torch(batch)

                # Reshape for UTD ratio
                if FLAGS.utd_ratio > 1:
                    def reshape_for_utd(x):
                        if isinstance(x, dict):
                            return {k: reshape_for_utd(v) for k, v in x.items()}
                        else:
                            return x.reshape(FLAGS.utd_ratio, config['batch_size'], *x.shape[1:])

                    batch = {k: reshape_for_utd(v) for k, v in batch.items()}
                    _, update_info["online_agent"] = agent.batch_update(batch)
                else:
                    update_info["online_agent"] = agent.update(batch)
            
            # Logging
            if i % FLAGS.log_interval == 0:
                for key, info in update_info.items():
                    logger.log(info, key, step=log_step)
                update_info = {}
            
            # Evaluation
            if i == FLAGS.online_steps or (FLAGS.eval_interval > 0 and i % FLAGS.eval_interval == 0):
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
                print(f"\n[Online Step {i}] Eval return: {eval_return:.2f}" if eval_return is not None else f"\n[Online Step {i}] Eval return: N/A")
            
            # Saving
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                save_path = os.path.join(FLAGS.save_dir, f"agent_{log_step}.pt")
                agent.save(save_path)
        
        # Save after online training
        save_path = os.path.join(FLAGS.save_dir, "agent_online_final.pt")
        agent.save(save_path)
    
    # Cleanup
    end_time = time.time()
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()
    
    # Log timing
    print(f"\n===== Training Complete =====")
    if FLAGS.offline_steps > 0:
        print(f"Offline training time: {online_init_time - offline_init_time:.1f}s")
    if FLAGS.online_steps > 0:
        print(f"Online training time: {end_time - online_init_time:.1f}s")
    print(f"Total time: {end_time - offline_init_time:.1f}s")
    
    # Save completion token
    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)


if __name__ == '__main__':
    app.run(main)
