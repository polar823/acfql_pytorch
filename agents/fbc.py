"""
Behavior Cloning Agent with Flow Matching (PyTorch Implementation)

Adapted from JAX/Flax ACFQLAgent to pure BC with flow matching.
Keeps critic structure for future RL experiments.

Supports:
- Encoder types: 'identity', 'mlp', 'image' (ResNet-based), 'impala'
- Policy types: 'mlp', 'chiunet', 'chitransformer', 'jannerunet'
- Multi-modal observations (multiple cameras + proprioceptive data)
- Action chunking with flexible horizon lengths
- Multiple loss types via unified loss module

Config is passed as a dictionary (similar to JAX version).
"""

import copy
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# New architecture components
from utils.flow_map import FlowMap
from utils.interpolant import Interpolant
from utils.networks import MLP, VanillaMLP, ChiUNet, ChiTransformer, JannerUNet, Value
from utils.encoders import IdentityEncoder, MLPEncoder, MultiImageObsEncoder
from utils.losses import get_loss_fn, OptimizationConfig


class BCAgent:
    """
    Behavior Cloning Agent using Flow Matching.

    Features:
    - Flow-based action modeling (continuous normalizing flow)
    - Action chunking support (predict multiple future actions)
    - Visual encoders: ResNet-based, IMPALA, MLP, Identity
    - Policy networks: MLP, ChiUNet, ChiTransformer, JannerUNet
    - Multi-modal observations (multiple cameras + proprioceptive data)
    - Critic network structure (for future RL fine-tuning)

    Config is passed as a dictionary (similar to JAX version).
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        config: Dict[str, Any],
        encoder: Optional[nn.Module] = None,
        critic_encoder: Optional[nn.Module] = None,
        flow_map: Optional[FlowMap] = None,
        interpolant: Optional[Interpolant] = None,
    ):
        """Initialize BC Agent.

        Args:
            actor: Policy network (flow model)
            critic: Critic network (for RL fine-tuning)
            target_critic: Target critic network
            actor_optimizer: Optimizer for actor (and encoder if present)
            critic_optimizer: Optimizer for critic
            config: Configuration dictionary
            encoder: Optional visual encoder for actor (separate from actor's internal encoder)
            critic_encoder: Optional visual encoder for critic
            flow_map: FlowMap wrapper for actor network
            interpolant: Interpolant for flow matching
        """
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config = config
        self.encoder = encoder
        self.critic_encoder = critic_encoder

        # Flow matching components
        self.flow_map = flow_map
        self.interpolant = interpolant

        self.device = torch.device(config['device'])

        # Move models to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
        if self.critic_encoder is not None:
            self.critic_encoder.to(self.device)
        if self.flow_map is not None:
            self.flow_map.to(self.device)

        # Get policy type from config (check both network_type and policy_type for compatibility)
        self.policy_type = config.get('network_type', config.get('policy_type', 'mlp'))

        # Training step counter
        self.step = 0

        # Create OptimizationConfig for loss functions
        self.opt_config = OptimizationConfig(
            loss_type=config.get('loss_type', 'flow'),
            loss_scale=config.get('loss_scale', 100.0),
            norm_type=config.get('norm_type', 'l2'),
            t_two_step=config.get('t_two_step', 0.9),
            discrete_dt=config.get('discrete_dt', 0.01),
            interp_type=config.get('interp_type', 'linear'),
        )

        # Get loss function based on loss_type
        self.loss_fn = get_loss_fn(config.get('loss_type', 'flow'))

        # Compile models for faster training (torch.compile)
        self.use_compile = config.get('use_compile', True)
        if self.use_compile:
            compile_mode = config.get('compile_mode', 'default')
            print(f"🚀 Compiling actor with torch.compile (mode={compile_mode})...")
            self.actor = torch.compile(self.actor, mode=compile_mode)
            if self.encoder is not None:
                print(f"🚀 Compiling encoder with torch.compile (mode={compile_mode})...")
                self.encoder = torch.compile(self.encoder, mode=compile_mode)
            print("✓ Compilation complete!")
    
    def _encode_observations(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode observations using the encoder if present.
        
        Args:
            observations: Raw observations (tensor or dict for multi-image)
        
        Returns:
            Encoded observations tensor
        """
        if self.encoder is not None:
            return self.encoder(observations)
        return observations
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BC flow matching loss using unified loss module.

        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim) or (B, C, H, W) or Dict[str, Tensor]
                - actions: (B, action_dim) or (B, T, action_dim) if chunking
                - valid: (B, T) mask for action chunks (optional)

        Returns:
            loss: Scalar loss
            info: Dictionary of logging information
        """
        observations = batch['observations']
        actions = batch['actions']

        # Get batch size
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            batch_size = observations[first_key].shape[0]
        else:
            batch_size = observations.shape[0]

        # Handle action chunking - determine format based on policy type
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 1)

        # Prepare actions in (B, T, act_dim) format for loss functions
        if actions.ndim == 2:
            batch_actions = actions.reshape(batch_size, horizon_length, action_dim)
        else:
            batch_actions = actions

        # Prepare observations for encoder
        # Ensure observations has sequence dimension for encoder
        if isinstance(observations, dict):
            obs_for_encoder = observations
        elif observations.ndim == 2:
            obs_for_encoder = observations.unsqueeze(1)  # (B, 1, obs_dim)
        else:
            obs_for_encoder = observations

        # Create delta_t tensor for loss function
        delta_t = torch.ones(batch_size, device=self.device)

        # Call unified loss function
        bc_flow_loss, loss_info = self.loss_fn(
            config=self.opt_config,
            flow_map=self.flow_map,
            encoder=self.encoder,
            interp=self.interpolant,
            act=batch_actions,
            obs=obs_for_encoder,
            delta_t=delta_t,
        )

        total_loss = self.config.get('bc_weight', 1.0) * bc_flow_loss

        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
        }
        # Add any additional info from loss function
        for k, v in loss_info.items():
            info[k] = v

        return total_loss, info
    
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute critic loss (placeholder for future RL).
        
        For BC training, this returns zero loss.
        """
        if not self.config.get('use_critic', False):
            # Return zero loss for BC-only training
            return torch.tensor(0.0, device=self.device), {
                'critic_loss': 0.0,
                'q_mean': 0.0,
                'q_max': 0.0,
                'q_min': 0.0,
            }
        
        # TODO: Implement for RL fine-tuning
        raise NotImplementedError("Critic loss for RL not implemented yet")
    
    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss (actor + critic)."""
        info = {}
        
        # Actor loss (BC flow matching)
        actor_loss, actor_info = self.actor_loss(batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Critic loss (zero for BC, can be activated for RL)
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        total = actor_loss + critic_loss
        info['total_loss'] = total.item()
        
        return total, info
    
    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step."""
        # Compute loss
        loss, info = self.total_loss(batch)

        # Update actor
        self.actor_optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping (aligned with mip/config.py)
        grad_clip_norm = self.config.get('grad_clip_norm', 10.0)
        if grad_clip_norm > 0:
            # Collect all parameters for gradient clipping
            params_to_clip = list(self.actor.parameters())
            if self.encoder is not None:
                params_to_clip += [p for p in self.encoder.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)

        self.actor_optimizer.step()

        # Update critic (if used)
        if self.config.get('use_critic', False):
            self.critic_optimizer.zero_grad()
            self.critic_optimizer.step()
            self.target_update()

        self.step += 1

        return info
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step.

        Args:
            batch: Dictionary of batched data (already on device)

        Returns:
            info: Dictionary of logging information
        """
        # Mark CUDA graph step boundary for torch.compile
        if self.use_compile and torch.cuda.is_available():
            torch.compiler.cudagraph_mark_step_begin()

        # Helper function to move to device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            return x

        # Move batch to device if needed
        batch = {k: to_device(v) for k, v in batch.items()}

        return self._update(batch)
    
    def batch_update(
        self, 
        batches: Dict[str, torch.Tensor]
    ) -> Tuple['BCAgent', Dict[str, float]]:
        """
        Perform multiple updates (one per batch).
        
        Args:
            batches: Dictionary where each value has shape (num_batches, batch_size, ...)
        
        Returns:
            agent: Self (for compatibility with JAX-style API)
            info: Averaged logging information
        """
        num_batches = next(iter(batches.values())).shape[0]
        
        all_info = []
        for i in range(num_batches):
            batch = {k: v[i] for k, v in batches.items()}
            info = self.update(batch)
            all_info.append(info)
        
        # Average info across batches
        avg_info = {}
        for key in all_info[0].keys():
            avg_info[key] = np.mean([info[key] for info in all_info])
        
        return self, avg_info
    
    def target_update(self):
        """Soft update of target critic network."""
        if not self.config.get('use_critic', False):
            return
        
        tau = self.config['tau']
        for target_param, param in zip(
            self.target_critic.parameters(), 
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    @torch.no_grad()
    def compute_flow_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noises: torch.Tensor,
    ):
        """Compute actions from the BC flow model using the Euler method.

        Uses ODE sampling aligned with much-ado-about-noising/mip/samplers.py:
        - act_s = act_s + b_s * (t - s)
        - where b_s = flow_map.get_velocity(s, act_s, obs_emb)

        Args:
            observations: Raw or encoded observations
            noises: Initial noise tensor

        Returns:
            Predicted actions
        """
        # Encode observations if needed
        obs_emb = self._encode_observations(observations)

        # Get batch size
        if isinstance(obs_emb, dict):
            first_key = list(obs_emb.keys())[0]
            batch_size = obs_emb[first_key].shape[0]
        else:
            batch_size = obs_emb.shape[0]

        flow_steps = self.config.get('flow_steps', 10)
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 1)

        # Ensure actions are in (B, T, act_dim) format for flow_map
        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit']:
            actions = noises  # Already (B, T, act_dim)
        else:
            # MLP-based policy - reshape to (B, T, act_dim)
            actions = noises.reshape(batch_size, horizon_length, action_dim)

        # Ensure obs_emb has correct format
        if self.encoder is not None:
            if not isinstance(obs_emb, dict) and obs_emb.ndim == 2:
                obs_emb = obs_emb.unsqueeze(1)  # (B, 1, emb_dim)
            condition = obs_emb
        else:
            if isinstance(observations, dict):
                condition = observations
            elif observations.ndim == 2:
                condition = observations.unsqueeze(1)  # (B, 1, obs_dim)
            else:
                condition = observations

        # ODE sampling using Euler method (aligned with mip/samplers.py ode_sampler)
        # act_s = act_s + b_s * (t - s)
        for i in range(flow_steps):
            s_val = i / flow_steps
            t_val = (i + 1) / flow_steps
            s = torch.full((batch_size,), s_val, device=self.device)
            t = torch.full((batch_size,), t_val, device=self.device)

            # Get velocity at current state (key fix: use get_velocity, not forward)
            b_s = self.flow_map.get_velocity(s, actions, condition)

            # Euler step: act_s = act_s + b_s * (t - s)
            dt = t_val - s_val
            actions = actions + b_s * dt

        # Clamp actions to valid range
        actions = torch.clamp(actions, -1, 1)

        # Reshape back to flat format for MLP
        if self.policy_type not in ['chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit']:
            actions = actions.reshape(batch_size, -1)

        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor], np.ndarray],
        temperature: float = 0.0,
    ) -> np.ndarray:
        """
        Sample actions using the trained flow model.
        
        Args:
            observations: (B, obs_dim) or (B, C, H, W) or Dict[str, Tensor]
            temperature: Sampling temperature (0 = deterministic)
        
        Returns:
            actions: (B, action_dim * horizon_length) flattened
        """
        # Helper to convert to torch
        def to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(self.device)
            elif isinstance(x, dict):
                return {k: to_torch(v) for k, v in x.items()}
            elif isinstance(x, torch.Tensor):
                return x.to(self.device)
            return x
        
        observations = to_torch(observations)
        
        # Ensure batch dimension exists
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            if observations[first_key].ndim in [1, 3]:  # (dim,) or (C, H, W)
                observations = {k: v.unsqueeze(0) for k, v in observations.items()}
            batch_size = observations[first_key].shape[0]
        else:
            if observations.ndim == 1:
                observations = observations.unsqueeze(0)
            batch_size = observations.shape[0]
        
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 1)

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies work with (B, T, act_dim)
            actions = torch.randn(batch_size, horizon_length, action_dim, device=self.device)
        else:
            # Start with noise in flat format
            full_action_dim = action_dim * horizon_length if self.config.get('action_chunking', True) else action_dim
            actions = torch.randn(batch_size, full_action_dim, device=self.device)
        
        if temperature > 0:
            actions = actions * temperature
        
        # Integrate flow using Euler method
        actions = self.compute_flow_actions(observations, actions)

        # Reshape to flat format if needed
        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # (B, T, act_dim) -> (B, T * act_dim)
            actions = actions.reshape(batch_size, -1)

        return actions.cpu().numpy()
    
    def save(self, path: str):
        """Save agent checkpoint."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
        }
        if self.encoder is not None:
            save_dict['encoder'] = self.encoder.state_dict()
        if self.critic_encoder is not None:
            save_dict['critic_encoder'] = self.critic_encoder.state_dict()
        # Note: flow_map and interpolant don't have learnable parameters,
        # so we don't need to save them. They will be recreated on load.

        torch.save(save_dict, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.encoder is not None and 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if self.critic_encoder is not None and 'critic_encoder' in checkpoint:
            self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.step = checkpoint.get('step', 0)
        # Note: flow_map and interpolant are already created in __init__
        print(f"Agent loaded from {path} (step {self.step})")
    
    @classmethod
    def create(
        cls,
        observation_shape: Union[Tuple[int, ...], Dict],
        action_dim: int,
        config: Dict[str, Any],
    ) -> 'BCAgent':
        """
        Create a new BC agent.

        Args:
            observation_shape: Shape of observations
                - For state: (obs_dim,) tuple
                - For images: (C, H, W) tuple
                - For multi-image: shape_meta dict (from robomimic_image_utils)
            action_dim: Dimension of action space
            config: Agent configuration dictionary with keys:
                - encoder: 'identity', 'mlp', 'image', 'impala'
                - network_type: 'mlp', 'chiunet', 'chitransformer', 'jannerunet'
                - emb_dim: Encoder output dimension (default 256)
                - horizon_length: Action chunking length
                - action_chunking: Enable action chunking
                - ... other standard config options

        Returns:
            agent: Initialized BC agent
        """
        config = dict(config)  # Copy to avoid mutation
        config['action_dim'] = action_dim
        config['act_dim'] = action_dim  # For network constructors

        # Set device if not specified
        if 'device' not in config:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        device = torch.device(config['device'])

        # Determine observation type
        is_multi_image = isinstance(observation_shape, dict)
        is_visual = is_multi_image or (isinstance(observation_shape, tuple) and len(observation_shape) == 3)

        if is_multi_image:
            config['shape_meta'] = observation_shape
            config['obs_type'] = 'image'
        elif is_visual:
            config['obs_dim'] = int(np.prod(observation_shape))
            config['obs_type'] = 'image'
        else:
            config['obs_dim'] = observation_shape[0]
            config['obs_type'] = 'state'

        # Get configuration parameters
        horizon_length = config.get('horizon_length', 5)
        config['Ta'] = horizon_length  # Action sequence length
        config['To'] = config.get('obs_steps', 1)  # Observation sequence length

        # Get network type from policy_type or network_type
        network_type = config.get('network_type', config.get('policy_type', 'mlp'))
        config['network_type'] = network_type

        # ===== Create Encoder =====
        encoder_type = config.get('encoder', 'mlp')
        emb_dim = config.get('emb_dim', 256)

        if encoder_type == 'identity':
            encoder = IdentityEncoder(dropout=config.get('encoder_dropout', 0.25))
            network_input_dim = config['obs_dim']
        elif encoder_type == 'mlp':
            encoder = MLPEncoder(
                obs_dim=config['obs_dim'],
                emb_dim=emb_dim,
                To=config['To'],
                hidden_dims=config.get('encoder_hidden_dims', [256, 256]),
                dropout=config.get('encoder_dropout', 0.25),
            )
            network_input_dim = emb_dim
        elif encoder_type == 'image':
            # Multi-image encoder with ResNet
            encoder = MultiImageObsEncoder(
                shape_meta=config['shape_meta'],
                rgb_model_name=config.get('rgb_model_name', 'resnet18'),
                emb_dim=emb_dim,
                resize_shape=config.get('resize_shape', None),
                crop_shape=config.get('crop_shape', None),
                random_crop=config.get('random_crop', True),
                use_group_norm=config.get('use_group_norm', True),
                share_rgb_model=config.get('share_rgb_model', False),
                imagenet_norm=config.get('imagenet_norm', False),
                use_seq=(config['To'] > 1),
                keep_horizon_dims=True,  # Keep (B, To, emb_dim) format
                pretrained=config.get('pretrained_encoder', True),
                freeze_rgb_encoder=config.get('freeze_encoder', True),
            )
            network_input_dim = emb_dim
        elif encoder_type == 'impala':
            # Legacy IMPALA encoder
            from utils.encoders import ImpalaEncoder
            if is_visual:
                input_shape = observation_shape if not is_multi_image else (3, 84, 84)
                encoder = ImpalaEncoder(
                    input_shape=input_shape,
                    width=1,
                    stack_sizes=(16, 32, 32),
                    num_blocks=2,
                    mlp_hidden_dims=(emb_dim,),
                )
                network_input_dim = emb_dim
            else:
                encoder = IdentityEncoder(dropout=config.get('encoder_dropout', 0.25))
                network_input_dim = config['obs_dim']
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        print(f"✓ Created encoder: {encoder_type} (output_dim={network_input_dim})")

        # ===== Create Policy Network =====
        if network_type == 'mlp':
            actor = MLP(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                emb_dim=config.get('actor_emb_dim', 512),
                n_layers=len(config.get('actor_hidden_dims', (512, 512, 512, 512))),
                timestep_emb_dim=config.get('time_encoder_dim', 128),
                disable_time_embedding=config.get('disable_time_embedding', False),
                dropout=config.get('dropout', 0.1),
            )
        elif network_type == 'vanillamlp':
            actor = VanillaMLP(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                emb_dim=config.get('actor_emb_dim', 512),
                n_layers=len(config.get('actor_hidden_dims', (512, 512, 512, 512))),
                dropout=config.get('dropout', 0.1),
                expansion_factor=config.get('expansion_factor', 1),
            )
        elif network_type == 'chiunet':
            actor = ChiUNet(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                emb_dim=config.get('model_dim', 256),
                kernel_size=config.get('kernel_size', 5),
                timestep_emb_type=config.get('timestep_emb_type', 'positional'),
                timestep_emb_params=config.get('timestep_emb_params', None),
                cond_predict_scale=config.get('cond_predict_scale', True),
                obs_as_global_cond=config.get('obs_as_global_cond', True),
                dim_mult=config.get('dim_mult', [1, 2]),
                disable_time_embedding=config.get('disable_time_embedding', False),
            )
        elif network_type == 'chitransformer':
            actor = ChiTransformer(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_layers', 8),
                timestep_emb_type=config.get('timestep_emb_type', 'positional'),
                timestep_emb_params=config.get('timestep_emb_params', None),
                p_drop_emb=config.get('p_drop_emb', 0.0),
                p_drop_attn=config.get('p_drop_attn', 0.3),
                n_cond_layers=config.get('n_cond_layers', 0),
            )
        elif network_type == 'jannerunet':
            actor = JannerUNet(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                emb_dim=config.get('model_dim', 256),
                timestep_emb_type=config.get('timestep_emb_type', 'positional'),
                timestep_emb_params=config.get('timestep_emb_params', None),
                norm_type=config.get('unet_norm_type', 'groupnorm'),
                attention=config.get('attention', False),
            )
        elif network_type == 'rnn':
            from utils.networks import RNN
            actor = RNN(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                rnn_hidden_dim=config.get('emb_dim', 256),
                rnn_num_layers=config.get('n_layers', 2),
                rnn_type=config.get('rnn_type', 'LSTM'),
                timestep_emb_dim=config.get('timestep_emb_dim', 128),
                max_freq=config.get('max_freq', 100.0),
                dropout=config.get('dropout', 0.1),
            )
        elif network_type == 'vanillarnn':
            from utils.networks import VanillaRNN
            actor = VanillaRNN(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                rnn_hidden_dim=config.get('emb_dim', 256),
                rnn_num_layers=config.get('n_layers', 2),
                rnn_type=config.get('rnn_type', 'LSTM'),
                dropout=config.get('dropout', 0.1),
            )
        elif network_type == 'dit':
            from utils.networks import DiT
            actor = DiT(
                act_dim=action_dim,
                Ta=horizon_length,
                obs_dim=network_input_dim,
                To=config['To'],
                d_model=config.get('emb_dim', 384),
                n_heads=config.get('n_heads', 6),
                depth=config.get('n_layers', 12),
                dropout=config.get('dropout', 0.0),
                timestep_emb_type=config.get('timestep_emb_type', 'positional'),
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}. Supported: mlp, vanillamlp, chiunet, chitransformer, jannerunet, rnn, vanillarnn, dit")

        print(f"✓ Created policy network: {network_type}")

        # ===== Create FlowMap and Interpolant =====
        flow_map = FlowMap(actor)
        interpolant = Interpolant(interp_type=config.get('interp_type', 'linear'))
        print(f"✓ Created FlowMap with {config.get('interp_type', 'linear')} interpolant")

        # ===== Create Critic Encoder (separate from actor encoder) =====
        if encoder_type == 'identity':
            critic_encoder = IdentityEncoder(dropout=config.get('encoder_dropout', 0.25))
        elif encoder_type == 'mlp':
            critic_encoder = MLPEncoder(
                obs_dim=config['obs_dim'],
                emb_dim=emb_dim,
                To=config['To'],
                hidden_dims=config.get('encoder_hidden_dims', [256, 256]),
                dropout=config.get('encoder_dropout', 0.25),
            )
        elif encoder_type == 'image':
            critic_encoder = MultiImageObsEncoder(
                shape_meta=config['shape_meta'],
                rgb_model_name=config.get('rgb_model_name', 'resnet18'),
                emb_dim=emb_dim,
                resize_shape=config.get('resize_shape', None),
                crop_shape=config.get('crop_shape', None),
                random_crop=config.get('random_crop', True),
                use_group_norm=config.get('use_group_norm', True),
                share_rgb_model=config.get('share_rgb_model', False),
                imagenet_norm=config.get('imagenet_norm', False),
                use_seq=(config['To'] > 1),
                keep_horizon_dims=True,
                pretrained=config.get('pretrained_encoder', True),
                freeze_rgb_encoder=config.get('freeze_encoder', True),
            )
        elif encoder_type == 'impala':
            from utils.encoders import ImpalaEncoder
            if is_visual:
                input_shape = observation_shape if not is_multi_image else (3, 84, 84)
                critic_encoder = ImpalaEncoder(
                    input_shape=input_shape,
                    width=1,
                    stack_sizes=(16, 32, 32),
                    num_blocks=2,
                    mlp_hidden_dims=(emb_dim,),
                )
            else:
                critic_encoder = IdentityEncoder(dropout=config.get('encoder_dropout', 0.25))
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # ===== Create Critic =====
        full_action_dim = action_dim * horizon_length if config.get('action_chunking', True) else action_dim
        critic = Value(
            observation_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dim=config.get('value_hidden_dims', (512, 512, 512, 512)),
            num_ensembles=config.get('num_qs', 2),
            encoder=None,  # Encoder is separate
            layer_norm=config.get('layer_norm', True),
        )

        # Create target critic
        target_critic = copy.deepcopy(critic)

        # ===== Create Optimizers =====
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)

        # Collect parameters for actor optimizer (includes encoder and flow_map)
        # Only include parameters that require gradients (exclude frozen encoder)
        actor_params = list(actor.parameters())
        if encoder is not None:
            encoder_params = [p for p in encoder.parameters() if p.requires_grad]
            actor_params += encoder_params

            # Print parameter counts
            total_encoder_params = sum(p.numel() for p in encoder.parameters())
            trainable_encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            if trainable_encoder_params < total_encoder_params:
                print(f"  Encoder: {trainable_encoder_params:,} / {total_encoder_params:,} parameters trainable "
                      f"({100 * trainable_encoder_params / total_encoder_params:.1f}%)")

        # Collect parameters for critic optimizer
        critic_params = list(critic.parameters())
        if critic_encoder is not None:
            critic_encoder_params = [p for p in critic_encoder.parameters() if p.requires_grad]
            critic_params += critic_encoder_params

        # Always use AdamW with weight_decay (aligned with mip/config.py)
        actor_optimizer = optim.AdamW(actor_params, lr=lr, weight_decay=weight_decay)
        critic_optimizer = optim.AdamW(critic_params, lr=lr, weight_decay=weight_decay)

        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=config,
            encoder=encoder,
            critic_encoder=critic_encoder,
            flow_map=flow_map,
            interpolant=interpolant,
        )


def get_config():
    """Get default configuration for BC agent (ml_collections.ConfigDict).

    Aligned with /home/xukainan/much-ado-about-noising/mip/config.py
    """
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='fbc',  # Agent name

            # ===== Optimization Config (aligned with mip/config.py OptimizationConfig) =====
            lr=1e-4,
            weight_decay=1e-5,
            batch_size=1024,
            grad_clip_norm=10.0,
            ema_rate=0.995,
            loss_scale=100.0,
            norm_type='l2',  # 'l2' or 'l1'
            interp_type='linear',  # 'linear' or 'trig'
            discount=0.99,

            # ===== Network Config (aligned with mip/config.py NetworkConfig) =====
            network_type='mlp',  # 'mlp', 'vanillamlp', 'chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit'
            n_layers=4,
            emb_dim=512,
            dropout=0.1,
            encoder_dropout=0.0,
            expansion_factor=4,
            timestep_emb_dim=128,
            timestep_emb_type='positional',  # 'positional', 'sinusoidal', 'fourier'

            # State encoder configs
            num_encoder_layers=2,
            encoder_hidden_dims=[256, 256],

            # Image encoder configs
            rgb_model_name='resnet18',
            use_seq=True,
            keep_horizon_dims=True,
            pretrained_encoder=True,
            freeze_encoder=True,
            resize_shape=None,
            crop_shape=None,
            random_crop=True,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=False,

            # Transformer specific configs
            n_heads=6,
            n_cond_layers=0,
            attn_dropout=0.1,
            d_model=256,
            nhead=4,
            num_layers=8,
            p_drop_emb=0.0,
            p_drop_attn=0.3,

            # UNet specific configs
            model_dim=256,
            kernel_size=5,
            cond_predict_scale=True,
            obs_as_global_cond=True,
            dim_mult=[1, 2],
            unet_norm_type='groupnorm',
            attention=False,

            # RNN specific configs
            rnn_type='LSTM',  # 'LSTM' or 'GRU'
            max_freq=100.0,

            # ===== Task Config =====
            horizon_length=16,  # Action prediction horizon (Ta)
            To=1,  # Observation steps
            action_dim=7,  # Will be set by create()
            obs_dim=23,  # Will be set by create()
            action_chunking=True,

            # ===== Encoder Config =====
            encoder='mlp',  # 'identity', 'mlp', 'image', 'impala'
            obs_type='state',  # 'state', 'image'

            # ===== Other =====
            device='cuda',
            use_compile=True,
            compile_mode='default',
            use_dataloader=True,

            # Critic (for future RL)
            use_critic=False,
            bc_weight=1.0,
            tau=0.005,
            num_qs=2,
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,

            # Flow sampling
            flow_steps=10,

            # Loss type configuration
            loss_type='flow',  # 'flow', 'regression', 'tsd', 'mip', 'lmd', 'ctm', 'psd', 'lsd', 'esd', 'mf'
            t_two_step=0.9,  # For tsd/mip loss
            discrete_dt=0.01,  # For ctm loss

            # Legacy parameters (for compatibility)
            actor_hidden_dims=(512, 512, 512, 512),
            actor_emb_dim=512,
            actor_layer_norm=True,
            time_encoder='sinusoidal',
            time_encoder_dim=64,
            timestep_emb_params=None,
            disable_time_embedding=False,
            use_fourier_features=False,
            fourier_feature_dim=64,
        )
    )
