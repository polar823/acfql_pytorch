"""
Flow Q-Learning (FQL) Agent (PyTorch Implementation)

Implements FQL algorithm with:
- Flow-based action modeling (continuous normalizing flow)
- Action chunking support
- Critic network for Q-learning
- Distillation policy for one-step action generation

Supports:
- Encoder types: 'identity', 'mlp', 'image' (ResNet-based), 'impala'
- Policy types: 'mlp' (ActorVectorField)
- Multi-modal observations (multiple cameras + proprioceptive data)
- Action chunking with flexible horizon lengths

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


class FQLAgent:
    """
    Flow Q-Learning Agent.

    Features:
    - Flow-based action modeling (continuous normalizing flow)
    - Action chunking support (predict multiple future actions)
    - Visual encoders: ResNet-based, IMPALA, MLP, Identity
    - Multi-modal observations (multiple cameras + proprioceptive data)
    - Critic network for Q-learning
    - Distillation policy for one-step action generation

    Config is passed as a dictionary (similar to JAX version).
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_onestep: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        config: Dict[str, Any],
        encoder: Optional[nn.Module] = None,
        critic_encoder: Optional[nn.Module] = None,
        flow_map: Optional[FlowMap] = None,
        flow_map_onestep: Optional[FlowMap] = None,
        interpolant: Optional[Interpolant] = None,
    ):
        """Initialize FQL Agent.

        Args:
            actor: Flow network (multi-step policy)
            actor_onestep: One-step distillation policy
            critic: Critic network
            target_critic: Target critic network
            actor_optimizer: Optimizer for actor (and encoder if present)
            critic_optimizer: Optimizer for critic
            config: Configuration dictionary
            encoder: Optional visual encoder for actor
            critic_encoder: Optional visual encoder for critic
            flow_map: FlowMap wrapper for actor network
            flow_map_onestep: FlowMap wrapper for one-step actor
            interpolant: Interpolant for flow matching
        """
        self.actor = actor  # Flow network (like BCAgent)
        self.actor_onestep = actor_onestep  # One-step distillation policy
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config = config
        self.encoder = encoder
        self.critic_encoder = critic_encoder

        # Flow matching components
        self.flow_map = flow_map
        self.flow_map_onestep = flow_map_onestep
        self.interpolant = interpolant

        self.device = torch.device(config['device'])

        # Move models to device
        self.actor.to(self.device)
        self.actor_onestep.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
        if self.critic_encoder is not None:
            self.critic_encoder.to(self.device)
        if self.flow_map is not None:
            self.flow_map.to(self.device)
        if self.flow_map_onestep is not None:
            self.flow_map_onestep.to(self.device)

        # Get policy type from config
        self.policy_type = config.get('network_type', config.get('policy_type', 'mlp'))

        # Training step counter
        self.step = 0
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
        """Encode observations using the encoder if present."""
        if self.encoder is not None:
            return self.encoder(observations)
        return observations
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FQL actor loss.

        Includes:
        - BC flow loss (flow matching)
        - Distillation loss (train one-step to match flow)
        - Q loss (maximize Q value)

        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim) or (B, C, H, W)
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

        # Encode observations if using separate encoder
        obs_emb = self._encode_observations(observations)

        # Handle action chunking - determine format based on policy type
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 5)

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies expect (B, T, act_dim)
            if actions.ndim == 2:
                # Reshape flat actions to (B, T, act_dim)
                batch_actions = actions.reshape(batch_size, horizon_length, action_dim)
            else:
                batch_actions = actions

            # Flow matching with interpolant
            x_0 = torch.randn_like(batch_actions)
            x_1 = batch_actions

            # Sample random time
            t = torch.rand(batch_size, device=self.device)

            # Use interpolant to compute x_t and target velocity
            x_t = self.interpolant.calc_It(t, x_0, x_1)
            target_vel = self.interpolant.calc_It_dot(t, x_0, x_1)

            # Predict velocity using flow_map
            if self.encoder is not None:
                pred_vel = self.flow_map.get_velocity(t, x_t, obs_emb)
            else:
                pred_vel = self.flow_map.get_velocity(t, x_t, observations)

            # Compute loss using L2 norm (aligned with mip/losses.py flow_loss)
            loss_scale = self.config.get('loss_scale', 100.0)
            norm_type = self.config.get('norm_type', 'l2')

            if norm_type == 'l2':
                loss_per_sample = torch.norm(pred_vel - target_vel, p=2, dim=-1) ** 2  # (B, T)
            else:
                loss_per_sample = torch.norm(pred_vel - target_vel, p=1, dim=-1) ** 2  # (B, T)

            if 'valid' in batch:
                valid = batch['valid']  # (B, T)
                bc_flow_loss = loss_scale * (loss_per_sample * valid).mean()
            else:
                bc_flow_loss = loss_scale * loss_per_sample.mean()
        else:
            # MLP-based policy expects flat (B, act_dim * T)
            if self.config.get('action_chunking', True):
                if actions.ndim == 3:
                    batch_actions = actions.reshape(batch_size, -1)
                else:
                    batch_actions = actions
            else:
                if actions.ndim == 3:
                    batch_actions = actions[:, 0, :]
                else:
                    batch_actions = actions

            # Flow matching with interpolant
            x_0 = torch.randn_like(batch_actions)
            x_1 = batch_actions

            # Sample random time
            t = torch.rand(batch_size, device=self.device)

            # Use interpolant to compute x_t and target velocity
            x_t = self.interpolant.calc_It(t, x_0, x_1)
            target_vel = self.interpolant.calc_It_dot(t, x_0, x_1)

            # Predict velocity (pass encoded observations)
            # For MLP, we need to reshape to (B, T, act_dim) format for flow_map
            x_t_reshaped = x_t.reshape(batch_size, horizon_length, action_dim)
            if self.encoder is not None:
                # Ensure obs_emb has sequence dimension
                if obs_emb.ndim == 2:
                    obs_emb_seq = obs_emb.unsqueeze(1)  # (B, 1, emb_dim)
                else:
                    obs_emb_seq = obs_emb
                pred_vel_reshaped = self.flow_map.get_velocity(t, x_t_reshaped, obs_emb_seq)
            else:
                # Ensure observations has sequence dimension
                if isinstance(observations, dict):
                    obs_seq = observations
                elif observations.ndim == 2:
                    obs_seq = observations.unsqueeze(1)  # (B, 1, obs_dim)
                else:
                    obs_seq = observations
                pred_vel_reshaped = self.flow_map.get_velocity(t, x_t_reshaped, obs_seq)

            # Reshape back to flat format
            pred_vel = pred_vel_reshaped.reshape(batch_size, -1)

            # Compute loss using L2 norm (aligned with mip/losses.py flow_loss)
            loss_scale = self.config.get('loss_scale', 100.0)
            norm_type = self.config.get('norm_type', 'l2')

            # Reshape for norm calculation: (B, T, act_dim)
            pred_vel_3d = pred_vel.reshape(batch_size, horizon_length, action_dim)
            target_vel_3d = target_vel.reshape(batch_size, horizon_length, action_dim)

            if norm_type == 'l2':
                loss_per_sample = torch.norm(pred_vel_3d - target_vel_3d, p=2, dim=-1) ** 2  # (B, T)
            else:
                loss_per_sample = torch.norm(pred_vel_3d - target_vel_3d, p=1, dim=-1) ** 2

            if self.config.get('action_chunking', True) and 'valid' in batch:
                valid = batch['valid']  # (B, T)
                bc_flow_loss = loss_scale * (loss_per_sample * valid).mean()
            else:
                bc_flow_loss = loss_scale * loss_per_sample.mean()

        # === Distillation Loss ===
        # Train one-step actor to match flow actions
        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit']:
            noises = torch.randn(batch_size, horizon_length, action_dim, device=self.device)
        else:
            full_action_dim = action_dim * horizon_length if self.config.get('action_chunking', True) else action_dim
            noises = torch.randn(batch_size, full_action_dim, device=self.device)

        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(observations, noises)

        # One-step actor prediction
        actor_actions = self.compute_onestep_actions(observations, noises)
        distill_loss = ((actor_actions - target_flow_actions) ** 2).mean()

        # === Q Loss ===
        # Maximize Q value for one-step actions
        actor_actions_clipped = torch.clamp(actor_actions, -1, 1)

        # Flatten actions if needed for critic (critic expects flat actions)
        if actor_actions_clipped.ndim == 3:
            actor_actions_flat = actor_actions_clipped.reshape(batch_size, -1)
        else:
            actor_actions_flat = actor_actions_clipped

        # Encode observations for critic if needed
        if self.critic_encoder is not None:
            obs_emb_critic = self.critic_encoder(observations)
            # Flatten obs_emb_critic if it has sequence dimension
            if obs_emb_critic.ndim == 3:
                obs_emb_critic = obs_emb_critic.reshape(batch_size, -1)
            qs = self.critic(obs_emb_critic, actor_actions_flat)
        else:
            # Flatten observations if needed
            if isinstance(observations, dict):
                obs_flat = observations
            elif observations.ndim == 3:
                obs_flat = observations.reshape(batch_size, -1)
            else:
                obs_flat = observations
            qs = self.critic(obs_flat, actor_actions_flat)
        q = qs.mean(dim=0)  # Mean over ensemble
        q_loss = -q.mean()

        # Total actor loss
        total_loss = self.config['bc_weight'] * bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
        }

        return total_loss, info
    
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FQL critic loss (TD loss).

        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim)
                - actions: (B, action_dim)
                - next_observations: (B, obs_dim)
                - rewards: (B,)
                - masks: (B,) - 1 if not done, 0 if done

        Returns:
            loss: Scalar critic loss
            info: Dictionary of logging information
        """
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch.get('next_observations', observations)
        rewards = batch['rewards']
        masks = batch.get('masks', 1.0 - batch.get('terminals', torch.zeros_like(rewards)))

        # Get batch size
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            batch_size = observations[first_key].shape[0]
        else:
            batch_size = observations.shape[0]

        # Encode observations for critic if needed
        if self.critic_encoder is not None:
            obs_emb = self.critic_encoder(observations)
            next_obs_emb = self.critic_encoder(next_observations)
            # Flatten if sequence dimension exists
            if obs_emb.ndim == 3:
                obs_emb = obs_emb.reshape(batch_size, -1)
            if next_obs_emb.ndim == 3:
                next_obs_emb = next_obs_emb.reshape(batch_size, -1)
        else:
            # Flatten observations if needed
            if isinstance(observations, dict):
                obs_emb = observations
                next_obs_emb = next_observations
            elif observations.ndim == 3:
                obs_emb = observations.reshape(batch_size, -1)
                next_obs_emb = next_observations.reshape(batch_size, -1)
            else:
                obs_emb = observations
                next_obs_emb = next_observations

        # Handle action chunking
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 5)

        if self.config.get('action_chunking', True):
            if actions.ndim == 3:
                batch_actions = actions.reshape(batch_size, -1)
            else:
                batch_actions = actions

            if next_observations.ndim == 3:
                next_obs = next_observations[:, -1, :]
            else:
                next_obs = next_observations

            if rewards.ndim == 2:
                last_reward = rewards[:, -1]
                last_mask = masks[:, -1] if masks.ndim == 2 else masks
            else:
                last_reward = rewards
                last_mask = masks
        else:
            if actions.ndim == 3:
                batch_actions = actions[:, 0, :]
            else:
                batch_actions = actions
            next_obs = next_observations
            last_reward = rewards if rewards.ndim == 1 else rewards[:, 0]
            last_mask = masks if masks.ndim == 1 else masks[:, 0]

        # Compute target Q value
        with torch.no_grad():
            # Sample next actions using one-step policy
            if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
                noises = torch.randn(batch_size, horizon_length, action_dim, device=self.device)
            else:
                full_action_dim = action_dim * horizon_length if self.config.get('action_chunking', True) else action_dim
                noises = torch.randn(batch_size, full_action_dim, device=self.device)

            # Compute next actions using one-step policy
            next_actions = self.compute_onestep_actions(next_obs, noises)
            next_actions = torch.clamp(next_actions, -1, 1)

            # Flatten next_actions if needed for critic (critic expects flat actions)
            if next_actions.ndim == 3:
                next_actions_flat = next_actions.reshape(batch_size, -1)
            else:
                next_actions_flat = next_actions

            # Compute target Q values
            if self.critic_encoder is not None:
                # Encode next observations for critic
                next_obs_emb_critic = self.critic_encoder(next_obs)
                if next_obs_emb_critic.ndim == 3:
                    next_obs_emb_critic = next_obs_emb_critic.reshape(batch_size, -1)
                next_qs = self.target_critic(next_obs_emb_critic, next_actions_flat)  # (num_ensembles, B)
            else:
                # Flatten next_obs if needed
                if isinstance(next_obs, dict):
                    next_obs_flat = next_obs
                elif next_obs.ndim == 3:
                    next_obs_flat = next_obs.reshape(batch_size, -1)
                else:
                    next_obs_flat = next_obs
                next_qs = self.target_critic(next_obs_flat, next_actions_flat)  # (num_ensembles, B)

            if self.config.get('q_agg', 'mean') == 'min':
                next_q = next_qs.min(dim=0)[0]
            else:
                next_q = next_qs.mean(dim=0)

            # Compute TD target
            discount_factor = self.config['discount'] ** (horizon_length if self.config.get('action_chunking', True) else 1)
            target_q = last_reward + discount_factor * last_mask * next_q

        # Compute current Q values
        current_qs = self.critic(obs_emb, batch_actions)  # (num_ensembles, B)

        # Get valid mask (critical for action chunking - masks out cross-episode samples)
        if 'valid' in batch:
            valid = batch['valid']
            if valid.ndim == 2:
                # Use last timestep's validity for TD target
                last_valid = valid[:, -1]
            else:
                last_valid = valid
        else:
            last_valid = torch.ones(batch_size, device=self.device)

        # Compute critic loss with valid mask (MSE for each ensemble, then mean)
        # This is critical to avoid Q-value explosion from cross-episode samples
        td_errors = (current_qs - target_q.unsqueeze(0)) ** 2  # (num_ensembles, B)
        critic_loss = (td_errors * last_valid.unsqueeze(0)).mean()

        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': current_qs.mean().item(),
            'q_max': current_qs.max().item(),
            'q_min': current_qs.min().item(),
            'target_q_mean': target_q.mean().item(),
        }

        return critic_loss, info
    
    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute total loss (actor + critic)."""
        info = {}
        
        # Actor loss
        actor_loss, actor_info = self.actor_loss(batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Critic loss
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        info['total_loss'] = (actor_loss + critic_loss).item()
        
        return actor_loss, critic_loss, info
    
    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step."""
        # Compute losses
        actor_loss, critic_loss, info = self.total_loss(batch)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Apply gradient clipping (aligned with mip/config.py)
        grad_clip_norm = self.config.get('grad_clip_norm', 10.0)
        if grad_clip_norm > 0:
            # Collect all actor parameters for gradient clipping
            params_to_clip = list(self.actor.parameters()) + list(self.actor_onestep.parameters())
            if self.encoder is not None:
                params_to_clip += [p for p in self.encoder.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)

        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Apply gradient clipping to critic
        if grad_clip_norm > 0:
            critic_params_to_clip = list(self.critic.parameters())
            if self.critic_encoder is not None:
                critic_params_to_clip += [p for p in self.critic_encoder.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(critic_params_to_clip, grad_clip_norm)

        self.critic_optimizer.step()

        # Update target network
        self.target_update()

        self.step += 1

        return info
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step.
        
        Args:
            batch: Dictionary of batched data
        
        Returns:
            info: Dictionary of logging information
        """

        # Mark CUDA graph step boundary for torch.compile
        if self.use_compile and torch.cuda.is_available():
            torch.compiler.cudagraph_mark_step_begin()

        # Move batch to device if needed
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        return self._update(batch)
    
    def batch_update(
        self, 
        batches: Dict[str, torch.Tensor]
    ) -> Tuple['FQLAgent', Dict[str, float]]:
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
        horizon_length = self.config.get('horizon_length', 5)

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
    def compute_onestep_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noises: torch.Tensor,
    ):
        """Compute actions from the one-step distillation policy.

        Uses ODE sampling aligned with much-ado-about-noising/mip/samplers.py:
        - One-step: directly map from t=0 to t=1 using get_velocity

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

        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 5)

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

        # One-step: directly map from t=0 to t=1
        s = torch.zeros(batch_size, device=self.device)

        # Get velocity at t=0 and apply one-step update
        b_s = self.flow_map_onestep.get_velocity(s, actions, condition)
        actions = actions + b_s * 1.0  # dt = 1.0 for one-step

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
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Sample actions using the trained policy.

        Uses the one-step distillation policy for fast inference.

        Args:
            observations: Observations
            temperature: Sampling temperature (for noise scaling)

        Returns:
            actions: numpy array
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
        horizon_length = self.config.get('horizon_length', 5)

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies work with (B, T, act_dim)
            noises = torch.randn(batch_size, horizon_length, action_dim, device=self.device)
        else:
            # Start with noise in flat format
            full_action_dim = action_dim * horizon_length if self.config.get('action_chunking', True) else action_dim
            noises = torch.randn(batch_size, full_action_dim, device=self.device)

        if temperature != 1.0:
            noises = noises * temperature

        # Use one-step policy for fast inference
        actions = self.compute_onestep_actions(observations, noises)

        # Reshape to flat format if needed
        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit']:
            # (B, T, act_dim) -> (B, T * act_dim)
            actions = actions.reshape(batch_size, -1)

        return actions.cpu().numpy()

    def save(self, path: str):
        """Save agent checkpoint."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_onestep': self.actor_onestep.state_dict(),
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

        torch.save(save_dict, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_onestep.load_state_dict(checkpoint['actor_onestep'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.encoder is not None and 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if self.critic_encoder is not None and 'critic_encoder' in checkpoint:
            self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.step = checkpoint.get('step', 0)
        print(f"Agent loaded from {path} (step {self.step})")
    
    @classmethod
    def create(
        cls,
        observation_shape: Union[Tuple[int, ...], Dict],
        action_dim: int,
        config: Dict[str, Any],
    ) -> 'FQLAgent':
        """
        Create a new FQL agent.

        Args:
            observation_shape: Shape of observations
                - For state: (obs_dim,) tuple
                - For images: (C, H, W) tuple
                - For multi-image: shape_meta dict (from robomimic_image_utils)
            action_dim: Dimension of action space
            config: Agent configuration dictionary

        Returns:
            agent: Initialized FQL agent
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

        # ===== Create Policy Networks =====
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
            # One-step actor (same architecture)
            actor_onestep = MLP(
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
            actor_onestep = VanillaMLP(
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
            actor_onestep = ChiUNet(
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
            actor_onestep = ChiTransformer(
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
            actor_onestep = JannerUNet(
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
            actor_onestep = RNN(
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
            actor_onestep = VanillaRNN(
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
            actor_onestep = DiT(
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
        flow_map_onestep = FlowMap(actor_onestep)
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

        # Collect parameters for actor optimizer (includes both actors and encoder)
        actor_params = list(actor.parameters()) + list(actor_onestep.parameters())
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
            actor_onestep=actor_onestep,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=config,
            encoder=encoder,
            critic_encoder=critic_encoder,
            flow_map=flow_map,
            flow_map_onestep=flow_map_onestep,
            interpolant=interpolant,
        )


def get_config():
    """Get default configuration for FQL agent (ml_collections.ConfigDict)."""
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='fql',
            lr=1e-4,  # Learning rate (aligned with mip/config.py)
            batch_size=1024,  # Batch size (aligned with mip/config.py)
            weight_decay=1e-5,  # Weight decay (aligned with mip/config.py)

            # New parameters from mip/config.py
            grad_clip_norm=10.0,  # Gradient clipping norm
            ema_rate=0.995,  # EMA rate for model averaging
            loss_scale=100.0,  # Loss scaling factor
            norm_type='l2',  # 'l2' or 'l1'

            # Encoder configuration
            encoder='mlp',  # Encoder type: 'identity', 'mlp', 'image', 'impala'
            obs_type='state',  # Observation type: 'state', 'image', 'keypoint'
            emb_dim=256,  # Encoder output dimension
            encoder_hidden_dims=[256, 256],  # For MLP encoder
            encoder_dropout=0.25,  # Encoder dropout

            # Image encoder specific
            rgb_model_name='resnet18',  # ResNet model: 'resnet18', 'resnet34', 'resnet50'
            pretrained_encoder=True,  # Load pretrained ImageNet weights
            freeze_encoder=True,  # Freeze encoder parameters (only train MLP projection)
            resize_shape=None,  # Optional image resize (H, W)
            crop_shape=None,  # Optional crop for augmentation (H, W)
            random_crop=True,  # Enable random crop augmentation
            share_rgb_model=False,  # Share encoder across multiple cameras
            use_group_norm=True,  # Use GroupNorm in ResNet
            imagenet_norm=False,  # Use ImageNet normalization

            # Network configuration
            network_type='mlp',  # Network: 'mlp', 'vanillamlp', 'chiunet', 'chitransformer', 'jannerunet', 'rnn', 'vanillarnn', 'dit'
            n_layers=4,  # Number of layers (aligned with mip/config.py)
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            expansion_factor=4,  # Expansion factor for MLP (aligned with mip/config.py)

            # ChiUNet-specific
            model_dim=256,
            kernel_size=5,
            cond_predict_scale=True,
            obs_as_global_cond=True,
            dim_mult=[1, 2],

            # ChiTransformer-specific
            d_model=256,
            nhead=4,
            num_layers=8,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            n_cond_layers=0,

            # JannerUNet-specific
            unet_norm_type='groupnorm',
            attention=False,

            # RNN-specific
            rnn_type='LSTM',  # 'LSTM' or 'GRU'
            max_freq=100.0,

            # DiT-specific
            n_heads=6,

            # Time embedding
            time_encoder='sinusoidal',  # Time encoder: None, 'sinusoidal', 'fourier', 'positional'
            time_encoder_dim=64,  # Time embedding dimension
            timestep_emb_type='positional',  # For U-Net/Transformer networks
            timestep_emb_params=None,
            timestep_emb_dim=128,
            disable_time_embedding=False,
            use_fourier_features=False,  # Legacy parameter
            fourier_feature_dim=64,  # Legacy parameter

            # Training
            discount=0.99,
            tau=0.005,
            num_qs=2,
            flow_steps=10,

            # Action chunking
            horizon_length=16,
            action_chunking=True,
            obs_steps=1,  # Observation context length

            # Compilation and optimization
            use_compile=True,  # Enable torch.compile for faster training
            compile_mode='default',  # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
            use_dataloader=True,  # Use PyTorch DataLoader for multi-process data loading

            # FQL specific
            bc_weight=1.0,
            alpha=100.0,
            q_agg='mean',  # Q aggregation method (min or mean)

            # Flow matching
            interp_type='linear',  # Interpolation type: 'linear' or 'trig'
        )
    )
