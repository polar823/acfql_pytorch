"""Timestep embeddings for flow matching networks.

This module provides various timestep embedding methods used in diffusion
and flow matching models, including positional, sinusoidal, and Fourier embeddings.

Reference: /home/xukainan/much-ado-about-noising/mip/embeddings.py
"""

import math

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures,
# from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    """Positional embedding with learnable frequencies.

    Args:
        dim: Embedding dimension (must be even)
        max_positions: Maximum number of positions for frequency scaling
        endpoint: Whether to include endpoint in frequency calculation
    """
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (...,) containing timesteps

        Returns:
            Embedded tensor of shape (..., dim)
        """
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class UntrainablePositionalEmbedding(nn.Module):
    """Untrainable positional embedding (fixed frequencies).

    Args:
        dim: Embedding dimension (must be even)
        max_positions: Maximum number of positions for frequency scaling
        endpoint: Whether to include endpoint in frequency calculation
    """
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (...,) containing timesteps

        Returns:
            Embedded tensor of shape (..., dim)
        """
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.einsum("...i,j->...ij", x, freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding (Transformer-style).

    Args:
        dim: Embedding dimension (must be even)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (...,) containing timesteps

        Returns:
            Embedded tensor of shape (..., dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum("...i,j->...ij", x, emb.to(x.dtype))
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------
# Timestep embedding with random Fourier features
class FourierEmbedding(nn.Module):
    """Fourier embedding with learnable MLP projection.

    Args:
        dim: Embedding dimension (must be divisible by 4)
        scale: Scale factor for random frequencies
    """
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (...,) containing timesteps

        Returns:
            Embedded tensor of shape (..., dim)
        """
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    """Untrainable Fourier embedding (no MLP projection).

    Args:
        dim: Embedding dimension (must be even)
        scale: Scale factor for random frequencies
    """
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (...,) containing timesteps

        Returns:
            Embedded tensor of shape (..., dim)
        """
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


# Dictionary mapping embedding types to classes
SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "sinusoidal": SinusoidalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
}


def get_timestep_embedding(emb_type: str, dim: int, **kwargs):
    """Factory function to create timestep embeddings.

    Args:
        emb_type: Type of embedding ('positional', 'sinusoidal', 'fourier', etc.)
        dim: Embedding dimension
        **kwargs: Additional arguments for the embedding class

    Returns:
        Timestep embedding module
    """
    if emb_type is None or emb_type == 'none':
        return None

    if emb_type not in SUPPORTED_TIMESTEP_EMBEDDING:
        raise ValueError(f"Unknown timestep embedding type: {emb_type}. "
                        f"Supported types: {list(SUPPORTED_TIMESTEP_EMBEDDING.keys())}")

    embedding_class = SUPPORTED_TIMESTEP_EMBEDDING[emb_type]
    return embedding_class(dim, **kwargs)
