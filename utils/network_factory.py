"""Factory functions for creating networks and encoders.

Provides unified interface for creating different network architectures
and observation encoders based on configuration.
"""

from utils.networks import ChiUNet, ChiTransformer, JannerUNet
from utils.encoders import IdentityEncoder, MLPEncoder, MultiImageObsEncoder
from utils.model import ActorVectorField, MeanActorVectorField


def get_network(config: dict):
    """Create network based on configuration.

    Args:
        config: Configuration dict with keys:
            - network_type: Type of network ('mlp', 'chiunet', 'chitransformer')
            - act_dim: Action dimension
            - horizon_length: Action sequence length (Ta)
            - obs_dim: Observation dimension (or embedding dimension for images)
            - obs_steps: Observation sequence length (To)
            - Other network-specific parameters

    Returns:
        Network module
    """
    network_type = config.get('network_type', 'mlp')

    # Common parameters
    act_dim = config['act_dim']
    Ta = config.get('horizon_length', 1)
    obs_dim = config.get('obs_dim', config.get('emb_dim', 256))
    To = config.get('obs_steps', 1)

    if network_type == 'mlp':
        # Get time encoder configuration
        time_encoder = config.get('time_encoder', 'sinusoidal')
        time_encoder_dim = config.get('time_encoder_dim', 64)

        # Handle legacy parameters
        if config.get('disable_time_embedding', False):
            time_encoder = None
        elif config.get('use_fourier_features', False):
            time_encoder = 'fourier'
            time_encoder_dim = config.get('fourier_feature_dim', 64)

        # Check if this is IMFBC/MFBC agent which needs MeanActorVectorField
        agent_name = config.get('agent_name', '')
        if agent_name in ['imfbc', 'mfbc']:
            # Use MeanActorVectorField for JVP-based flow matching
            return MeanActorVectorField(
                observation_dim=obs_dim * To,  # Flatten observation sequence
                action_dim=act_dim * Ta,  # Flatten action sequence
                hidden_dim=config.get('actor_hidden_dims', (512, 512, 512, 512)),
                time_encoder=time_encoder,
                time_encoder_dim=time_encoder_dim,
                use_fourier_features=config.get('use_fourier_features', False),
                fourier_feature_dim=config.get('fourier_feature_dim', 64),
                layer_norm=config.get('actor_layer_norm', True),  # Enable layer norm by default
            )
        else:
            # Use standard ActorVectorField for regular flow matching
            return ActorVectorField(
                observation_dim=obs_dim * To,  # Flatten observation sequence
                action_dim=act_dim * Ta,  # Flatten action sequence
                hidden_dim=config.get('actor_hidden_dims', (512, 512, 512, 512)),
                time_encoder=time_encoder,
                time_encoder_dim=time_encoder_dim,
                use_fourier_features=config.get('use_fourier_features', False),
                fourier_feature_dim=config.get('fourier_feature_dim', 64),
                layer_norm=config.get('actor_layer_norm', True),  # Enable layer norm by default
            )

    elif network_type == 'chiunet':
        return ChiUNet(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            model_dim=config.get('model_dim', 256),
            emb_dim=config.get('emb_dim', 256),
            kernel_size=config.get('kernel_size', 5),
            cond_predict_scale=config.get('cond_predict_scale', True),
            obs_as_global_cond=config.get('obs_as_global_cond', True),
            dim_mult=config.get('dim_mult', None),
            timestep_emb_type=config.get('timestep_emb_type', 'positional'),
            timestep_emb_params=config.get('timestep_emb_params', None),
            disable_time_embedding=config.get('disable_time_embedding', False),
        )

    elif network_type == 'chitransformer':
        return ChiTransformer(
            act_dim=act_dim,
            obs_dim=obs_dim,
            Ta=Ta,
            To=To,
            d_model=config.get('emb_dim', 256),
            nhead=config.get('n_heads', 4),
            num_layers=config.get('num_layers', 8),
            p_drop_emb=config.get('dropout', 0.0),
            p_drop_attn=config.get('attn_dropout', 0.3),
            n_cond_layers=config.get('n_cond_layers', 0),
            timestep_emb_type=config.get('timestep_emb_type', 'positional'),
            timestep_emb_params=config.get('timestep_emb_params', None),
            disable_time_embedding=config.get('disable_time_embedding', False),
        )

    elif network_type == 'jannerunet':
        return JannerUNet(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            model_dim=config.get('model_dim', 32),
            emb_dim=config.get('emb_dim', 32),
            kernel_size=config.get('kernel_size', 3),
            dim_mult=config.get('dim_mult', None),
            norm_type=config.get('norm_type', 'groupnorm'),
            attention=config.get('attention', False),
            timestep_emb_type=config.get('timestep_emb_type', 'positional'),
            timestep_emb_params=config.get('timestep_emb_params', None),
            disable_time_embedding=config.get('disable_time_embedding', False),
        )

    else:
        raise ValueError(f"Unknown network type: {network_type}")


def get_encoder(config: dict):
    """Create encoder based on configuration.

    Args:
        config: Configuration dict with keys:
            - encoder: Encoder type ('identity', 'mlp', 'image', None)
            - obs_type: Observation type ('state', 'image')
            - obs_dim: Observation dimension (for state)
            - obs_steps: Observation sequence length (To)
            - emb_dim: Embedding dimension
            - shape_meta: Shape metadata (for image observations)
            - Other encoder-specific parameters

    Returns:
        Encoder module
    """
    encoder_type = config.get('encoder', None)
    obs_type = config.get('obs_type', 'state')

    # Auto-detect encoder type based on obs_type if not specified
    if encoder_type is None:
        if obs_type == 'image':
            encoder_type = 'image'
        elif obs_type in ['state', 'keypoint']:
            encoder_type = 'mlp'
        else:
            encoder_type = 'identity'

    if encoder_type == 'identity' or encoder_type is None:
        return IdentityEncoder(
            dropout=config.get('encoder_dropout', 0.25)
        )

    elif encoder_type == 'mlp':
        return MLPEncoder(
            obs_dim=config['obs_dim'],
            emb_dim=config.get('emb_dim', 256),
            To=config.get('obs_steps', 1),
            hidden_dims=config.get('encoder_hidden_dims', [256, 256]),
            dropout=config.get('encoder_dropout', 0.25),
        )

    elif encoder_type == 'image':
        if 'shape_meta' not in config:
            raise ValueError("shape_meta required for image encoder")

        # Determine if we should keep horizon dims based on obs_steps and network type
        obs_steps = config.get('obs_steps', 1)
        network_type = config.get('network_type', 'mlp')

        # For networks with global conditioning (ChiUNet, ChiTransformer), flatten the sequence
        # For obs_steps=1, we don't need sequence dimension
        if obs_steps == 1 or network_type in ['chiunet', 'chitransformer', 'jannerunet']:
            keep_horizon_dims = False
            use_seq = obs_steps > 1  # Only use seq mode if obs_steps > 1
        else:
            keep_horizon_dims = config.get('keep_horizon_dims', True)
            use_seq = config.get('use_seq', True)

        return MultiImageObsEncoder(
            shape_meta=config['shape_meta'],
            rgb_model_name=config.get('rgb_model_name', 'resnet18'),
            emb_dim=config.get('emb_dim', 256),
            resize_shape=config.get('resize_shape', None),
            crop_shape=config.get('crop_shape', None),
            random_crop=config.get('random_crop', True),
            use_group_norm=config.get('use_group_norm', True),
            share_rgb_model=config.get('share_rgb_model', False),
            imagenet_norm=config.get('imagenet_norm', False),
            use_seq=use_seq,
            keep_horizon_dims=keep_horizon_dims,
            pretrained=config.get('pretrained_encoder', True),
            freeze_rgb_encoder=config.get('freeze_encoder', True),
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_network_and_encoder(config: dict):
    """Create both network and encoder based on configuration.

    Args:
        config: Configuration dict

    Returns:
        Tuple of (network, encoder)
    """
    network = get_network(config)
    encoder = get_encoder(config)
    return network, encoder
