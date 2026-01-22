"""Robomimic environment utilities.

Provides functions to create robomimic environments and datasets for both
low-dimensional (state) and image-based observations.

Reference: /home/xukainan/much-ado-about-noising/mip/envs/robomimic/
"""

from os.path import expanduser
import os
import collections

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from utils.datasets import Dataset


def is_robomimic_env(env_name):
    """Determine if an env is robomimic (both low_dim and image)."""
    # Check if it's a robomimic task
    task_name = env_name.split("-")[0]
    if task_name not in ("lift", "can", "square", "transport", "tool_hang"):
        return False

    # Check if it has valid dataset type
    parts = env_name.split("-")
    if len(parts) < 2:
        return False

    dataset_type = parts[1]
    return dataset_type in ("mh", "ph", "mg")


def is_robomimic_image_env(env_name):
    """Check if environment name indicates image observations."""
    return env_name.endswith("-image")


# Default low-dim observation keys
DEFAULT_LOW_DIM_KEYS = (
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object'
)

# Initialize observation modality mapping for low-dim
low_dim_keys = {"low_dim": DEFAULT_LOW_DIM_KEYS}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    """Get maximum episode length for robomimic tasks."""
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


def _check_dataset_exists(env_name):
    """Check if low-dim dataset exists and return path."""
    # Parse environment name
    parts = env_name.replace("-image", "").replace("-low_dim", "").split("-")
    task = parts[0]
    dataset_type = parts[1] if len(parts) > 1 else "mh"

    # Try different file name versions
    file_names = [
        "low_dim_v141.hdf5",
        "low_dim_v15.hdf5",
        "low_dim.hdf5",
    ]
    if dataset_type == "mg":
        file_names = [
            "low_dim_sparse_v141.hdf5",
            "low_dim_sparse_v15.hdf5",
            "low_dim_sparse.hdf5",
        ] + file_names

    for file_name in file_names:
        download_folder = os.path.join(
            expanduser("~/.robomimic"),
            task,
            dataset_type,
            file_name
        )
        if os.path.exists(download_folder):
            return download_folder

    raise FileNotFoundError(
        f"Dataset not found for {env_name}. "
        f"Tried: {file_names} in ~/.robomimic/{task}/{dataset_type}/. "
        f"Please download the robomimic dataset first."
    )


def _check_dataset_exists_image(env_name):
    """Check if image dataset exists and return path."""
    # Parse environment name
    parts = env_name.replace("-image", "").split("-")
    task = parts[0]
    dataset_type = parts[1] if len(parts) > 1 else "mh"

    # Try image dataset first
    file_name = "image_v141.hdf5"
    download_folder = os.path.join(
        expanduser("~/.robomimic"),
        task,
        dataset_type,
        file_name
    )

    if os.path.exists(download_folder):
        return download_folder

    # Fallback to low_dim dataset (can still extract images from it)
    return _check_dataset_exists(env_name)


def make_env(env_name, seed=0):
    """
    Create robomimic environment (auto-detects low_dim vs image).

    Args:
        env_name: Environment name (e.g., "lift-mh-low_dim" or "lift-mh-image")
        seed: Random seed

    Returns:
        env: Wrapped environment instance
    """
    if is_robomimic_image_env(env_name):
        return make_robomimic_image_env(env_name, seed=seed)
    else:
        return make_robomimic_lowdim_env(env_name, seed=seed)


def make_robomimic_lowdim_env(env_name, seed=0):
    """
    Create low-dimensional robomimic environment.

    NOTE: should call get_dataset() first to ensure metadata is downloaded.

    Args:
        env_name: Environment name (e.g., "lift-mh-low_dim")
        seed: Random seed

    Returns:
        env: RobomimicLowdimWrapper instance
    """
    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
    )
    env = RobomimicLowdimWrapper(
        env,
        obs_keys=DEFAULT_LOW_DIM_KEYS,
        max_episode_length=max_episode_length
    )
    env.seed(seed)

    return env


def make_robomimic_image_env(env_name, seed=0, shape_meta=None):
    """
    Create image-based robomimic environment.

    Args:
        env_name: Environment name (e.g., "lift-mh-image")
        seed: Random seed
        shape_meta: Shape metadata dict (if None, will be extracted from dataset)

    Returns:
        env: RobomimicImageWrapper instance
    """
    dataset_path = _check_dataset_exists_image(env_name)

    # Get shape metadata if not provided
    if shape_meta is None:
        shape_meta = get_shape_meta_from_dataset(dataset_path)

    # Get environment metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Disable object state observation for image mode
    env_meta["env_kwargs"]["use_object_obs"] = False

    # Initialize observation modality mapping
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # Create base environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )

    # Disable hard reset to reduce memory consumption
    env.env.hard_reset = False

    # Wrap with custom wrapper
    env = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        init_state=None,
    )

    env.seed(seed)

    return env

def get_shape_meta_from_dataset(dataset_path, obs_keys=None, use_eye_in_hand=True):
    """Extract shape metadata from robomimic dataset.

    Args:
        dataset_path: Path to the HDF5 dataset
        obs_keys: List of observation keys to use (if None, auto-detect from dataset)
        use_eye_in_hand: Whether to include eye-in-hand camera

    Returns:
        shape_meta: Dictionary containing observation and action shapes
    """
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        demo = demos[0]

        # Get action shape
        actions = f[f"data/{demo}/actions"][()]
        action_shape = actions.shape[1:]

        # Get observation shapes
        obs_group = f[f"data/{demo}/obs"]

        # Auto-detect observation keys if not provided
        if obs_keys is None:
            obs_keys = []
            # Add image observations
            for key in obs_group.keys():
                if "image" in key or key in ["agentview", "robot0_eye_in_hand"]:
                    if not use_eye_in_hand and "eye_in_hand" in key:
                        continue
                    obs_keys.append(key)
            # Add proprioceptive observations
            for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
                if key in obs_group:
                    obs_keys.append(key)

        # Build shape_meta
        shape_meta = {
            "action": {"shape": action_shape},
            "obs": {}
        }

        for key in obs_keys:
            if key not in obs_group:
                continue

            data = obs_group[key][()]
            obs_shape = data.shape[1:]

            # Determine observation type
            if "image" in key or key in ["agentview", "robot0_eye_in_hand"]:
                obs_type = "rgb"
                # Ensure key ends with _image for consistency
                if not key.endswith("_image"):
                    key_with_suffix = f"{key}_image"
                else:
                    key_with_suffix = key

                # Convert image shape from (H, W, C) to (C, H, W) for PyTorch
                if len(obs_shape) == 3 and obs_shape[-1] in [1, 3]:
                    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (H, W, C) -> (C, H, W)
            else:
                obs_type = "low_dim"
                key_with_suffix = key

            shape_meta["obs"][key_with_suffix] = {
                "shape": obs_shape,
                "type": obs_type
            }

    return shape_meta


def get_dataset(env, env_name):
    """
    Load low-dimensional robomimic dataset.

    Args:
        env: Environment instance (used to get observation space)
        env_name: Environment name

    Returns:
        Dataset object with observations, actions, rewards, etc.
    """
    dataset_path = _check_dataset_exists(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"Loading low-dim dataset with {num_timesteps} timesteps from {num_demos} demos")

    # Data holders
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []

    # Go through and add to the data holder
    for ep in demos:
        a = np.array(rm_dataset[f"data/{ep}/actions"])
        obs, next_obs = [], []
        for k in DEFAULT_LOW_DIM_KEYS:
            obs.append(np.array(rm_dataset[f"data/{ep}/obs/{k}"]))
        for k in DEFAULT_LOW_DIM_KEYS:
            next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]))
        obs = np.concatenate(obs, axis=-1)
        next_obs = np.concatenate(next_obs, axis=-1)
        dones = np.array(rm_dataset[f"data/{ep}/dones"])
        r = np.array(rm_dataset[f"data/{ep}/rewards"])

        observations.append(obs.astype(np.float32))
        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))

    rm_dataset.close()

    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )
def get_image_dataset(env_name, shape_meta=None):
    """
    Load image-based robomimic dataset with lazy loading.

    Args:
        env_name: Environment name (e.g., "lift-mh-image")
        shape_meta: Shape metadata dict (if None, will be extracted from dataset)

    Returns:
        Dataset object with lazy-loaded image observations
    """
    dataset_path = _check_dataset_exists_image(env_name)

    # Get shape metadata if not provided
    if shape_meta is None:
        shape_meta = get_shape_meta_from_dataset(dataset_path)

    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        num_demos = len(demos)
        inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
        demos = [demos[i] for i in inds]

        # Count total timesteps
        num_timesteps = 0
        for ep in demos:
            num_timesteps += int(f[f"data/{ep}/actions"].shape[0])

        print(f"Loading image dataset with {num_timesteps} timesteps from {num_demos} demos")

        # Data holders - load non-image data into memory
        actions = []
        terminals = []
        rewards = []
        masks = []

        # Store episode info for lazy loading
        episode_starts = []
        episode_lengths = []
        current_idx = 0

        # Load non-image data from each demo
        for ep in demos:
            # Load actions
            a = np.array(f[f"data/{ep}/actions"])
            ep_len = len(a)
            actions.append(a.astype(np.float32))

            episode_starts.append(current_idx)
            episode_lengths.append(ep_len)
            current_idx += ep_len

            # Load rewards and dones
            r = np.array(f[f"data/{ep}/rewards"])
            dones = np.array(f[f"data/{ep}/dones"])
            rewards.append(r.astype(np.float32))
            terminals.append(dones.astype(np.float32))
            masks.append(1.0 - dones.astype(np.float32))

        # Concatenate non-image data
        actions_array = np.concatenate(actions, axis=0)
        rewards_array = np.concatenate(rewards, axis=0)
        terminals_array = np.concatenate(terminals, axis=0)
        masks_array = np.concatenate(masks, axis=0)

    # Create dataset with lazy loading for images
    from utils.lazy_image_dataset import LazyImageDataset

    return LazyImageDataset.create(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        actions=actions_array,
        rewards=rewards_array,
        terminals=terminals_array,
        masks=masks_array,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
    )


def make_env_and_dataset(env_name, seed=0):
    """
    Create environment and load dataset (auto-detects low_dim vs image).

    Args:
        env_name: Environment name (e.g., "lift-mh-low_dim" or "lift-mh-image")
        seed: Random seed

    Returns:
        env: Training environment
        eval_env: Evaluation environment
        dataset: Training dataset
        shape_meta: Shape metadata (None for low_dim, dict for image)
    """
    if is_robomimic_image_env(env_name):
        # Image-based environment
        dataset_path = _check_dataset_exists_image(env_name)
        shape_meta = get_shape_meta_from_dataset(dataset_path)

        env = make_robomimic_image_env(env_name, seed=seed, shape_meta=shape_meta)
        eval_env = make_robomimic_image_env(env_name, seed=seed + 1000, shape_meta=shape_meta)
        dataset = get_image_dataset(env_name, shape_meta=shape_meta)

        return env, eval_env, dataset, shape_meta
    else:
        # Low-dimensional environment
        env = make_robomimic_lowdim_env(env_name, seed=seed)
        eval_env = make_robomimic_lowdim_env(env_name, seed=seed + 1000)
        dataset = get_dataset(env, env_name)

        return env, eval_env, dataset, None


class RobomimicImageWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with image observations.
    Ported from https://github.com/CleanDiffuserTeam/CleanDiffuser.
    """
    def __init__(
        self,
        env,
        shape_meta,
        init_state=None,
        render_obs_key="agentview_image",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = {}
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = gym.spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = gym.spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("quat") or key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported observation type: {key}")

            this_space = gym.spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        # Handle render cache key mapping
        render_key = self.render_obs_key
        if render_key not in raw_obs:
            # Try without _image suffix
            if render_key.endswith("_image"):
                base_key = render_key.replace("_image", "")
                if base_key in raw_obs:
                    render_key = base_key
                else:
                    raise ValueError(
                        f"ERROR: Neither '{render_key}' nor '{base_key}' found in raw_obs keys: {list(raw_obs.keys())}"
                    )
        self.render_cache = raw_obs[render_key]

        obs = {}

        for key in self.observation_space:
            # Map dataset keys to environment keys
            # Dataset has keys like 'robot0_eye_in_hand_image' but env provides 'robot0_eye_in_hand'
            # Also 'agentview_image' -> 'agentview'
            env_key = key
            if key.endswith("_image"):
                # Try removing '_image' suffix for camera observations
                base_key = key.replace("_image", "")
                if base_key in raw_obs:
                    env_key = base_key

            # Check if this key exists in raw_obs
            if env_key not in raw_obs:
                # Special handling for common camera keys
                if key == "agentview_image" and "agentview" in raw_obs:
                    obs[key] = raw_obs["agentview"]
                elif key == "robot0_eye_in_hand_image" and "robot0_eye_in_hand" in raw_obs:
                    obs[key] = raw_obs["robot0_eye_in_hand"]
                else:
                    print(
                        f"Warning: Observation key '{key}' (mapped to '{env_key}') not found in raw observations. "
                        f"Available keys: {list(raw_obs.keys())}"
                    )
                    # For missing keys, create a zero array matching expected shape
                    if key in self.observation_space:
                        obs[key] = np.zeros(
                            self.observation_space[key].shape,
                            dtype=self.observation_space[key].dtype,
                        )
            else:
                obs[key] = raw_obs[env_key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed=None, options=None):
        # Handle seed parameter from gymnasium API
        if seed is not None:
            self.seed(seed)

        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs and info (Gymnasium API)
        obs = self.get_observation(raw_obs)
        info = {}
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        # Return 5 values for Gymnasium API (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


class RobomimicLowdimWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        obs_keys=None,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
    ):
        if obs_keys is None:
            obs_keys = [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ]
        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = {}
        self._seed = None
        self.max_episode_length = max_episode_length
        self.t = 0

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed=None, options=None):
        # Handle seed parameter from Gymnasium API
        if seed is not None:
            self._seed = seed

        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        self.t = 0
        # return obs and info (Gymnasium API requires both)
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

        self.t += 1

        # Add success info based on reward
        if reward > 0.:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0

        # Handle episode termination vs truncation
        terminated = done
        truncated = False
        if self.max_episode_length is not None and self.t >= self.max_episode_length:
            truncated = True
            terminated = False

        # Return 5 values for Gymnasium API (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )


if __name__ == "__main__":
    # for testing 
    env = make_env("lift-mh-low_dim")
    dataset = get_dataset(env, "lift-mh-low_dim")
    print(dataset)
    # transport-mh-low_dim