"""Base network class for all policy networks.

All network architectures should inherit from BaseNetwork and implement
the forward method with a unified interface.

Reference: /home/xukainan/much-ado-about-noising/mip/networks/base.py
"""

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """Base class for all policy networks.
    
    Args:
        act_dim: Action dimension
        Ta: Action sequence length (horizon)
        obs_dim: Observation dimension (or embedding dimension after encoder)
        To: Observation sequence length
        emb_dim: Internal embedding dimension
        n_layers: Number of layers in the network
    """
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.emb_dim = emb_dim
        self.n_layers = n_layers

    def forward(self, x, s, t, condition):
        """Forward pass of the network.
        
        Args:
            x: Action tensor of shape (b, Ta, act_dim)
            s: Source time of shape (b,)
            t: Target time of shape (b,)
            condition: Observation condition (can be dict or tensor)
                - If dict: multi-modal observations (e.g., images + proprioception)
                - If tensor: shape (b, To, obs_dim)
        
        Returns:
            y: Output action tensor of shape (b, Ta, act_dim)
            scalar: Scalar output of shape (b, 1) (optional, can be None)
        """
        raise NotImplementedError("Subclasses must implement forward method")
