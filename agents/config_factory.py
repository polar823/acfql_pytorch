"""Shared configuration factory for all agents.

This module provides common configuration parameters and utilities
to reduce duplication across agent implementations.
"""

import ml_collections


def get_base_config() -> ml_collections.ConfigDict:
    """Get base configuration shared by all agents.

    Returns:
        ConfigDict with common parameters
    """
    return ml_collections.ConfigDict(
        dict(
            # Learning parameters (aligned with mip/config.py)
            lr=1e-4,  # Learning rate
            batch_size=1024,  # Batch size
            weight_decay=1e-5,  # Weight decay for optimizer

            # New parameters from mip/config.py
            grad_clip_norm=10.0,  # Gradient clipping norm
            ema_rate=0.995,  # EMA rate for model averaging
            loss_scale=100.0,  # Loss scaling factor

            # Encoder configuration
            encoder='mlp',  # Encoder type: 'identity', 'mlp', 'image'
            obs_type='state',  # Observation type: 'state', 'image'
            emb_dim=256,  # Encoder output dimension
            encoder_hidden_dims=[256, 256],  # For MLP encoder
            encoder_dropout=0.25,  # Dropout rate for encoder
            freeze_encoder=True,  # Freeze encoder parameters

            # Image encoder specific
            rgb_model_name='resnet18',  # ResNet variant: resnet18/34/50
            resize_shape=None,  # Optional image resize (H, W)
            crop_shape=None,  # Optional crop for augmentation (H, W)
            random_crop=True,  # Enable random crop augmentation
            share_rgb_model=False,  # Share encoder across multiple cameras
            use_group_norm=True,  # Use GroupNorm in ResNet
            imagenet_norm=False,  # Use ImageNet normalization

            # Network configuration
            network_type='mlp',  # Network: 'mlp', 'chiunet', 'chitransformer', 'jannerunet'
            n_layers=4,  # Number of layers (aligned with mip/config.py)
            actor_hidden_dims=(512, 512, 512, 512),  # For MLP network
            actor_layer_norm=True,  # Enable LayerNorm in actor MLP
            expansion_factor=4,  # Expansion factor for MLP (aligned with mip/config.py)

            # ChiUNet-specific
            model_dim=256,
            kernel_size=5,
            cond_predict_scale=True,
            obs_as_global_cond=True,
            dim_mult=[1, 2],  # Reduced depth for small horizon_length

            # ChiTransformer-specific
            d_model=256,
            nhead=4,
            num_layers=8,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            attn_dropout=0.3,
            n_cond_layers=0,

            # JannerUNet-specific
            norm_type='groupnorm',
            attention=False,

            # Time embedding
            time_encoder='sinusoidal',  # Time encoder: None, 'sinusoidal', 'fourier', 'positional'
            time_encoder_dim=64,  # Time embedding dimension
            timestep_emb_type='positional',  # For U-Net/Transformer networks
            timestep_emb_params=None,
            disable_time_embedding=False,  # Enable time embedding for ChiUNet
            use_fourier_features=False,  # Legacy parameter
            fourier_feature_dim=64,  # Legacy parameter

            # Critic configuration
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            num_qs=2,  # Critic ensemble size

            # Training
            discount=0.99,  # Discount factor
            tau=0.005,  # Target network update rate
            flow_steps=10,  # Number of flow steps for inference
            use_critic=False,  # Whether to use critic (BC agents: False)
            bc_weight=1.0,  # BC coefficient

            # Compilation and optimization
            use_compile=True,  # Enable torch.compile for faster training
            compile_mode='default',  # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
            use_dataloader=True,  # Use PyTorch DataLoader for multi-process data loading

            # Action chunking
            horizon_length=4,  # Action sequence length
            action_chunking=True,  # Enable action chunking
            obs_steps=1,  # Observation context length
        )
    )


def get_fbc_config() -> ml_collections.ConfigDict:
    """Get FBC-specific configuration.

    Returns:
        ConfigDict for FBC agent
    """
    config = get_base_config()
    config.agent_name = 'fbc'
    config.horizon_length = 5  # FBC default
    return config


def get_mfbc_config() -> ml_collections.ConfigDict:
    """Get MFBC-specific configuration.

    Returns:
        ConfigDict for MFBC agent
    """
    config = get_base_config()
    config.agent_name = 'mfbc'
    config.horizon_length = 5  # MFBC default

    # MFBC-specific time sampling parameters
    config.time_logit_mu = -0.4
    config.time_logit_sigma = 1.0
    config.time_instant_prob = 0.2

    return config


def get_imfbc_config() -> ml_collections.ConfigDict:
    """Get IMFBC-specific configuration.

    Returns:
        ConfigDict for IMFBC agent
    """
    config = get_base_config()
    config.agent_name = 'imfbc'
    config.horizon_length = 4  # IMFBC default

    # IMFBC-specific time sampling parameters
    config.time_logit_mu = -0.4
    config.time_logit_sigma = 1.0
    config.time_instant_prob = 0.2

    return config


def get_fql_config() -> ml_collections.ConfigDict:
    """Get FQL-specific configuration.

    Returns:
        ConfigDict for FQL agent
    """
    config = get_base_config()
    config.agent_name = 'fql'
    config.horizon_length = 5  # FQL default

    # FQL-specific parameters
    config.use_critic = True  # FQL uses critic
    config.alpha = 100.0  # BC coefficient for FQL
    config.q_agg = 'mean'  # Q aggregation method
    config.normalize_q_loss = False

    # Actor type for FQL
    config.actor_type = 'distill-ddpg'  # or 'best-of-n'
    config.actor_num_samples = 32  # For best-of-n

    return config


def update_config_for_env(config: ml_collections.ConfigDict, env_name: str) -> ml_collections.ConfigDict:
    """Update configuration based on environment.

    Args:
        config: Base configuration
        env_name: Environment name

    Returns:
        Updated configuration
    """
    # Robomimic environments
    if any(task in env_name for task in ['lift', 'can', 'square', 'transport']):
        # Robomimic-specific settings
        if 'image' in env_name:
            config.encoder = 'image'
            config.obs_type = 'image'
        else:
            config.encoder = 'identity'
            config.obs_type = 'state'

    # D4RL environments
    elif any(task in env_name for task in ['halfcheetah', 'hopper', 'walker', 'ant']):
        config.encoder = 'identity'
        config.obs_type = 'state'

    # OGBench environments
    elif 'singletask' in env_name or 'multitask' in env_name:
        config.encoder = 'image'
        config.obs_type = 'image'

    return config


def merge_configs(base: ml_collections.ConfigDict, overrides: dict) -> ml_collections.ConfigDict:
    """Merge base config with overrides.

    Args:
        base: Base configuration
        overrides: Dictionary of override values

    Returns:
        Merged configuration
    """
    config = ml_collections.ConfigDict(base)
    config.update(overrides)
    return config
