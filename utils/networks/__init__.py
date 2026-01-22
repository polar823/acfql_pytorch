"""Network architectures for flow matching.

This module provides various network architectures for action prediction:
- MLP: Advanced MLP with LayerNorm and residual blocks (default)
- VanillaMLP: Simplified MLP without fancy features
- ChiUNet: U-Net architecture from Diffusion Policy
- ChiTransformer: Transformer architecture from Diffusion Policy
- JannerUNet: U-Net architecture from Janner et al.
- RNN: RNN/LSTM/GRU architecture with time embeddings
- VanillaRNN: Simplified RNN without fancy features
- DiT: Diffusion Transformer architecture
- Value: Critic network for Q-learning
"""

from utils.networks.base import BaseNetwork
from utils.networks.mlp import MLP, VanillaMLP
from utils.networks.chiunet import ChiUNet
from utils.networks.chitransformer import ChiTransformer
from utils.networks.jannerunet import JannerUNet
from utils.networks.rnn import RNN, VanillaRNN
from utils.networks.dit import DiT, SudeepDiT
from utils.networks.value import Value

__all__ = [
    "BaseNetwork",
    "MLP",
    "VanillaMLP",
    "ChiUNet",
    "ChiTransformer",
    "JannerUNet",
    "RNN",
    "VanillaRNN",
    "DiT",
    "SudeepDiT",
    "Value",
]
