"""Value/Critic networks for Q-learning.

Contains Value network for Q(s,a) or V(s) estimation with ensemble support.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP for Value network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (512, 512, 512, 512),
        activation: nn.Module = nn.GELU,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class Value(nn.Module):
    """Value/Critic Network.

    Can be used as V(s) network (when action_dim=0 or None)
    or as Q(s, a) network (when action_dim > 0).

    Features:
    - Contains num_ensembles independent MLP networks (Ensemble).
    - Output shape is (num_ensembles, batch_size) for easy min/mean computation.
    """

    def __init__(
        self,
        observation_dim,
        action_dim=None,
        hidden_dim=(512, 512, 512, 512),
        num_ensembles=2,
        encoder=None,
        layer_norm=True,
    ):
        super().__init__()

        self.num_ensembles = num_ensembles
        self.encoder = encoder

        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = observation_dim

        if action_dim is not None and action_dim > 0:
            self.input_dim += action_dim

        # Build Ensemble
        self.nets = nn.ModuleList([
            SimpleMLP(
                input_dim=self.input_dim,
                output_dim=1,
                hidden_dims=hidden_dim if isinstance(hidden_dim, (list, tuple)) else (hidden_dim,) * 4,
                activation=nn.GELU,
                layer_norm=layer_norm,
            )
            for _ in range(num_ensembles)
        ])

    def forward(self, observations, actions=None):
        """Forward pass.

        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim) [optional]

        Returns:
            values: (num_ensembles, batch_size)
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations

        if actions is not None:
            inputs = torch.cat([inputs, actions], dim=-1)

        outputs = []
        for net in self.nets:
            out = net(inputs)  # (batch_size, 1)
            outputs.append(out.squeeze(-1))  # (batch_size,)

        # [batch_size] * N -> [num_ensembles, batch_size]
        outputs = torch.stack(outputs, dim=0)

        return outputs
