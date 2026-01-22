"""Interpolant for stochastic interpolation with different types.

Supports linear and trigonometric interpolation for flow matching.

Reference: /home/xukainan/much-ado-about-noising/mip/interpolant.py
"""

import torch


def at_least_ndim(x, ndim: int, pad: int = 0):
    """Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Input tensor, int, or float
        ndim: Minimum number of dimensions
        pad: Padding direction. 0: pad in the last dimension, 1: pad in the first dimension

    Returns:
        Reshaped tensor or original value if int/float
    """
    if isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


class Interpolant:
    """Class for a stochastic interpolant with different types of interpolation.
    
    Supports two interpolation types:
    - linear: alpha(t) = 1-t, beta(t) = t
    - trig: alpha(t) = cos(t*π/2), beta(t) = sin(t*π/2)
    """

    def __init__(self, interp_type: str = "linear"):
        """Initialize an interpolant with the specified type.

        Args:
            interp_type: Type of interpolation ("linear" or "trig")
        """
        if interp_type == "linear":
            self.alpha = lambda t: 1.0 - t
            self.beta = lambda t: t
            self.alpha_dot = lambda _: -1.0
            self.beta_dot = lambda _: 1.0
        elif interp_type == "trig":
            self.alpha = lambda t: (
                torch.cos(torch.tensor(t) * torch.pi / 2)
                if isinstance(t, (int, float))
                else torch.cos(t * torch.pi / 2)
            )
            self.beta = lambda t: (
                torch.sin(torch.tensor(t) * torch.pi / 2)
                if isinstance(t, (int, float))
                else torch.sin(t * torch.pi / 2)
            )
            self.alpha_dot = (
                lambda t: -0.5
                * torch.pi
                * (
                    torch.sin(torch.tensor(t) * torch.pi / 2)
                    if isinstance(t, (int, float))
                    else torch.sin(t * torch.pi / 2)
                )
            )
            self.beta_dot = (
                lambda t: 0.5
                * torch.pi
                * (
                    torch.cos(torch.tensor(t) * torch.pi / 2)
                    if isinstance(t, (int, float))
                    else torch.cos(t * torch.pi / 2)
                )
            )
        else:
            raise NotImplementedError(
                f"Interpolant type '{interp_type}' not implemented."
            )

    def calc_It(
        self, t: float | torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the interpolant at time t.
        
        I(t) = alpha(t) * x0 + beta(t) * x1
        
        Args:
            t: Time value(s)
            x0: Source tensor
            x1: Target tensor
            
        Returns:
            Interpolated tensor
        """
        t = at_least_ndim(t, x0.dim())
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(
        self, t: float | torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the time derivative of the interpolant at time t.
        
        dI(t)/dt = alpha_dot(t) * x0 + beta_dot(t) * x1
        
        Args:
            t: Time value(s)
            x0: Source tensor
            x1: Target tensor
            
        Returns:
            Time derivative of interpolated tensor
        """
        t = at_least_ndim(t, x0.dim())
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1
