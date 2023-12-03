# https://gist.github.com/xmodar/ae2d94681a6fda39f3c4f3ac91eef7b7
# %%
from functools import partial
import torch


def sinusoidal(positions, features=16, periods=10000):
    """Encode `positions` using sinusoidal positional encoding

    Args:
        positions: tensor of positions
        features: half the number of features per position
        periods: used frequencies for the sinusoidal functions

    Returns:
        Positional encoding of shape `(*positions.shape, features, 2)`
    """
    dtype = positions.dtype if positions.is_floating_point() else None
    kwargs = dict(device=positions.device, dtype=dtype)
    omega = torch.logspace(0, 1 / features - 1, features, periods, **kwargs)
    fraction = omega * positions.unsqueeze(-1)
    return torch.stack((fraction.sin(), fraction.cos()), dim=-1)


def point_pe(points, low=0, high=1, steps=100, features=16, periods=10000):
    """Encode points in bounded space using sinusoidal positional encoding

    Args:
        points: tensor of points; typically of shape (*, C)
        low: lower bound of the space; typically of shape (C,)
        high: upper bound of the space; typically of shape (C,)
        steps: number of cells that split the space; typically of shape (C,)
        features: half the number of features per position
        periods: used frequencies for the sinusoidal functions

    Returns:
        Positional encoded points of the following shape:
        `(*points.shape[:-1], points.shape[-1] * features * 2)`
    """
    positions = (points - low).mul_(steps / (high - low))
    return sinusoidal(positions, features, periods).flatten(-3)


def point_position_encoding(points, max_steps=100, features=16, periods=10000):
    low = points.min(0).values
    high = points.max(0).values
    steps = high - low
    steps *= max_steps / steps.max()
    pe = point_pe(points, low, high, steps, features, periods)
    return pe


def test(num_points=1000, max_steps=100, features=32, periods=10000):
    """Test point_pe"""
    point_cloud = torch.rand(num_points, 3)
    low = point_cloud.min(0).values
    high = point_cloud.max(0).values
    steps = high - low
    steps *= max_steps / steps.max()
    # print(point_pe(point_cloud, low, high, steps).shape)
    pe = point_pe(point_cloud, low, high, steps, features=features, periods=periods)
    return pe


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_steps=100, features=32, periods=1000):
        super().__init__()
        self.pe = partial(
            point_position_encoding,
            max_steps=max_steps,
            features=features,
            periods=periods,
        )

    @torch.no_grad()
    def forward(self, x):
        return self.pe(x)