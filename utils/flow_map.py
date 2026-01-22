"""FlowMap wrapper for neural networks with partial derivative computation.

Designed for stochastic interpolant framework. Wraps a neural network and provides
methods to compute Jacobian-vector products (JVPs) for flow matching.

Reference: /home/xukainan/much-ado-about-noising/mip/flow_map.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.interpolant import at_least_ndim


class FlowMap(nn.Module):
    """FlowMap wrapper for neural networks with partial derivative computation.
    
    Wraps a policy network and provides:
    - Forward flow mapping: x_st = x_s + (t-s) * f(x_s, s, t, condition)
    - Jacobian-vector products (JVPs) for efficient gradient computation
    - Velocity field extraction
    
    Args:
        net: Policy network (e.g., MLP, ChiUNet, etc.)
        reference_net: Optional reference network for distillation
    """

    def __init__(
        self,
        net: nn.Module,
        reference_net: nn.Module | None = None,
    ):
        super().__init__()
        self.net = net
        self.reference_net = reference_net

    def forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the flow map.
        
        Computes: x_st = x_s + (t-s) * f(x_s, s, t, label)
        
        Args:
            s: Source time (b,)
            t: Target time (b,)
            xs: Source state (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            x_st: Mapped state (b, Ta, act_dim)
        """
        f_xst, _ = self.net(xs, s, t, label)
        ss = at_least_ndim(s, xs.dim())
        ts = at_least_ndim(t, xs.dim())
        xst = xs + (ts - ss) * f_xst
        return xst

    def get_map_and_velocity(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both the map and velocity.
        
        Args:
            s: Source time (b,)
            t: Target time (b,)
            xs: Source state (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (b, Ta, act_dim)
            f_xst: Velocity field (b, Ta, act_dim)
        """
        f_xst, _ = self.net(xs, s, t, label)
        ss = at_least_ndim(s, xs.dim())
        ts = at_least_ndim(t, xs.dim())
        xst = xs + (ts - ss) * f_xst
        return xst, f_xst

    def forward_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for a single sample (unbatched).
        
        Args:
            s: Source time (scalar)
            t: Target time (scalar)
            xs: Source state (Ta, act_dim)
            label: Condition/observation (To, obs_dim) or dict
            
        Returns:
            x: Mapped state (Ta, act_dim)
        """
        s_batch = s.unsqueeze(0)
        t_batch = t.unsqueeze(0)
        xs_batch = xs.unsqueeze(0)
        label_batch = label.unsqueeze(0) if label is not None else None
        x = self.forward(s_batch, t_batch, xs_batch, label_batch)[0]
        return x

    def jvp_t_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobian-vector product with respect to t for a single sample.

        Uses jvp to compute the forward pass and partial derivative with respect to t
        at the same time, which can save computation.
        
        Args:
            s: Source time (scalar)
            t: Target time (scalar)
            xs: Source state (Ta, act_dim)
            label: Condition/observation (To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (Ta, act_dim)
            d_xst_dt: Partial derivative w.r.t. t (Ta, act_dim)
        """

        def f_single_wrapped(t_input):
            return self.forward_single(s, t_input, xs, label)

        return torch.func.jvp(
            f_single_wrapped,
            (t,),
            (torch.tensor(1.0, device=xs.device),),
        )

    def jvp_s_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobian-vector product with respect to s for a single sample.

        Uses jvp to compute the forward pass and partial derivative with respect to s
        at the same time, which can save computation.
        
        Args:
            s: Source time (scalar)
            t: Target time (scalar)
            xs: Source state (Ta, act_dim)
            label: Condition/observation (To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (Ta, act_dim)
            d_xst_ds: Partial derivative w.r.t. s (Ta, act_dim)
        """

        def f_single_wrapped(s_input):
            return self.forward_single(s_input, t, xs, label)

        return torch.func.jvp(
            f_single_wrapped,
            (s,),
            (torch.tensor(1.0, device=xs.device),),
        )

    def jvp_x_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        tangent: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobian-vector product with respect to x for a single sample.

        Uses jvp to compute the forward pass and partial derivative with respect to x
        at the same time, which can save computation.
        
        Args:
            s: Source time (scalar)
            t: Target time (scalar)
            xs: Source state (Ta, act_dim)
            tangent: Tangent vector (Ta, act_dim)
            label: Condition/observation (To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (Ta, act_dim)
            d_xst_dx: Jacobian-vector product (Ta, act_dim)
        """

        def f_single_wrapped(xs_input):
            return self.forward_single(s, t, xs_input, label)

        return torch.func.jvp(
            f_single_wrapped,
            (xs,),
            (tangent,),
        )

    def jvp_t(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched Jacobian-vector product with respect to t using vmap.
        
        Args:
            s: Source time (b,)
            t: Target time (b,)
            xs: Source state (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (b, Ta, act_dim)
            d_xst_dt: Partial derivative w.r.t. t (b, Ta, act_dim)
        """
        return torch.func.vmap(self.jvp_t_single, randomness="same")(s, t, xs, label)

    def jvp_s(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched Jacobian-vector product with respect to s using vmap.
        
        Args:
            s: Source time (b,)
            t: Target time (b,)
            xs: Source state (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (b, Ta, act_dim)
            d_xst_ds: Partial derivative w.r.t. s (b, Ta, act_dim)
        """
        return torch.func.vmap(self.jvp_s_single, randomness="same")(s, t, xs, label)

    def jvp_x(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        tangent: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched Jacobian-vector product with respect to x using vmap.
        
        Args:
            s: Source time (b,)
            t: Target time (b,)
            xs: Source state (b, Ta, act_dim)
            tangent: Tangent vector (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            xst: Mapped state (b, Ta, act_dim)
            d_xst_dx: Jacobian-vector product (b, Ta, act_dim)
        """
        return torch.func.vmap(self.jvp_x_single, randomness="same")(
            s, t, xs, tangent, label
        )

    def get_velocity(self, t, xs, label):
        """Get the velocity field of the flow.
        
        Args:
            t: Time (b,)
            xs: State (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            bt: Velocity field (b, Ta, act_dim)
        """
        bt, _ = self.net(xs, t, t, label)
        return bt

    def get_reference_velocity(self, t, xs, label):
        """Get the velocity field of the reference net.
        
        Args:
            t: Time (b,)
            xs: State (b, Ta, act_dim)
            label: Condition/observation (b, To, obs_dim) or dict
            
        Returns:
            bt: Velocity field (b, Ta, act_dim)
        """
        bt, _ = self.reference_net(xs, t, t, label)
        return bt
