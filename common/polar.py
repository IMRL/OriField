import numpy as np


def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates to Polar coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    """Convert polar coordinates to Cartesian coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polar_sub(a, b):
    a_r, a_theta = cartesian_to_polar(a[..., 0], a[..., 1])
    b_r, b_theta = cartesian_to_polar(b[..., 0], b[..., 1])
    return a_theta - b_theta


def polar_add(a, b):
    a_r, a_theta = cartesian_to_polar(a[..., 0], a[..., 1])
    b_r, b_theta = cartesian_to_polar(b[..., 0], b[..., 1])
    return a_theta + b_theta


def cross_add(a, b):
    b_x, b_y = polar_to_cartesian(np.full(b.shape, 1.), b)
    b = np.stack([b_x, b_y], axis=-1)
    return polar_add(a, b)
