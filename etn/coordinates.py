"""
Sampling grids for 2D canonical coordinate systems.
Each coordinate system is defined by a pair of coordinates u, v.
Each grid function maps from a grid in (u, v) coordinates to a collection points in Cartesian coordinates.
"""

import tensorflow as tf
import numpy as np


def denormalize(grid, im_shape):
    # maps from [-1, 1] to [0, im_shape]
    im_shape = tf.cast(im_shape, dtype=tf.float32)
    im_shape = tf.reshape(im_shape, [1, 1, -1])
    grid = im_shape / 2. * (grid + 1)
    return grid


def _grid_prepare(output_size, ulim, vlim):
    """Prepares meshgrids for respective coordinate systems
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits

    Returns:
        tuple of tf.Tensor, type tf.float32, shape (output_size[0], output_size[1])
    """
    nv, nu = output_size[0], output_size[1]
    ulim = (tf.convert_to_tensor(ulim[0], dtype=tf.float32),
            tf.convert_to_tensor(ulim[1], dtype=tf.float32))
    vlim = (tf.convert_to_tensor(vlim[0], dtype=tf.float32),
            tf.convert_to_tensor(vlim[1], dtype=tf.float32))

    urange = tf.linspace(ulim[0], ulim[1], nu)
    vrange = tf.linspace(vlim[0], vlim[1], nv)
    vs, us = tf.meshgrid(vrange, urange, indexing='ij')
    return vs, us


def identity_grid(output_size, ulim=(-1, 1), vlim=(-1, 1)):
    """Cartesian coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    vs, us = _grid_prepare(output_size, ulim, vlim)
    xs = us
    ys = vs
    return tf.stack([xs, ys], 2)


def polar_grid(output_size, ulim=(0., np.sqrt(2.)), vlim=(-np.pi, np.pi)):
    """Polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        
    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    vs, us = _grid_prepare(output_size, ulim, vlim)
    xs = us * tf.cos(vs)
    ys = us * tf.sin(vs)
    return tf.stack([xs, ys], 2)


def logpolar_grid(output_size, ulim=(None, np.log(2.) / 2.), vlim=(-np.pi, np.pi)):
    """Log-polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    if ulim[0] is None:
        ulim = (-tf.math.log(tf.cast(output_size[1], dtype=tf.float32)), ulim[1])
    vs, us = _grid_prepare(output_size, ulim, vlim)
    rs = tf.exp(us)
    xs = rs * tf.cos(vs)
    ys = rs * tf.sin(vs)
    return tf.stack([xs, ys], 2)


def shearx_grid(output_size, ulim=(-1, 1), vlim=(-5, 5.)):
    """Horizontal shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian y-coordinate limits
        vlim: (float, float), x/y ratio limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    vs, us = _grid_prepare(output_size, ulim, vlim)
    ys = us
    xs = us * vs
    return tf.stack([xs, ys], 2)


def sheary_grid(output_size, ulim=(-1, 1), vlim=(-5, 5)):
    """Vertical shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), y/x ratio limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    vs, us = _grid_prepare(output_size, ulim, vlim)
    xs = us
    ys = us * vs
    return tf.stack([xs, ys], 2)


def scalex_grid(output_size, ulim=(None, 0), vlim=(-1, 1)):
    """Horizontal scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-tf.math.log(tf.cast(nu, dtype=tf.float32) / 2.), ulim[1])
    vs, us = _grid_prepare((nv // 2, nu), ulim, vlim)
    
    xs = tf.exp(us)
    ys = vs
    
    if nv % 2 == 0:
        xs = tf.concat([xs, -xs], axis=0)
        ys = tf.concat([ys, ys], axis=0)
    else:
        xs = tf.concat([xs, xs[-1:], -xs], axis=0)
        ys = tf.concat([ys, ys[-1:], ys], axis=0)
    return tf.stack([xs, ys], 2)


def scaley_grid(output_size, ulim=(None, 0), vlim=(-1, 1)):
    """Vertical scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic y-coordinate limits
        vlim: (float, float), Cartesian x-coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-tf.math.log(tf.cast(nu, dtype=tf.float32) / 2.), ulim[1])
    vs, us = _grid_prepare((nv // 2, nu), ulim, vlim)
    
    ys = tf.exp(us)
    xs = vs
    
    if nv % 2 == 0:
        xs = tf.concat([xs, xs], axis=0)
        ys = tf.concat([ys, -ys], axis=0)
    else:
        xs = tf.concat([xs, xs[-1:], xs], axis=0)
        ys = tf.concat([ys, ys[-1:], -ys], axis=0)
    return tf.stack([xs, ys], 2)


def hyperbolic_grid(output_size, ulim=(-np.sqrt(0.5), np.sqrt(0.5)), vlim=(-np.log(6.), np.log(6.))):
    """Hyperbolic coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), hyperbolic angular coordinate limits
        vlim: (float, float), hyperbolic log-radial coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    nv, nu = output_size
    vs, us = _grid_prepare((nv // 2, nu), ulim, vlim)
    
    rs = tf.exp(vs)
    xs = us * rs
    ys = us / rs
    
    if nv % 2 == 0:
        xs = tf.concat([xs, xs], axis=0)
        ys = tf.concat([ys, -ys], axis=0)
    else:
        xs = tf.concat([xs, xs[-1:], xs], axis=0)
        ys = tf.concat([ys, ys[-1:], -ys], axis=0)
    return tf.stack([xs, ys], 2)


def perspectivex_grid(output_size, ulim=(1, 8), vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2)):
    """Horizontal perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), x^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    nv, nu = output_size
    vs, us = _grid_prepare((nv // 2, nu), ulim, vlim)
    
    xl = -tf.math.reciprocal(tf.reverse(us, [1]))
    xr = tf.math.reciprocal(us)
    yl = -xl * tf.tan(vs)
    yr = xr * tf.tan(vs)
    
    if nv % 2 == 0:
        xs = tf.concat([xl, xr], axis=0)
        ys = tf.concat([yl, yr], axis=0)
    else:
        xs = tf.concat([xl, xl[-1:], xr], axis=0)
        ys = tf.concat([yl, yl[-1:], yr], axis=0)
    return tf.stack([xs, ys], 2)


def perspectivey_grid(output_size, ulim=(1, 8), vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2)):
    """Vertical perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), y^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    nv, nu = output_size
    vs, us = _grid_prepare((nv // 2, nu), ulim, vlim)
    
    yl = -tf.math.reciprocal(tf.reverse(us, [1]))
    yr = tf.math.reciprocal(us)
    xl = -yl * tf.tan(vs)
    xr = yr * tf.tan(vs)
    
    if nv % 2 == 0:
        xs = tf.concat([xl, xr], axis=0)
        ys = tf.concat([yl, yr], axis=0)
    else:
        xs = tf.concat([xl, xl[-1:], xr], axis=0)
        ys = tf.concat([yl, yl[-1:], yr], axis=0)
    return tf.stack([xs, ys], 2)


def spherical_grid(output_size, ulim=(-np.pi / 4, np.pi / 4), vlim=(-np.pi / 4, np.pi / 4)):
    """Spherical coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), latitudinal coordinate limits
        vlim: (float, float), longitudinal coordinate limits

    Returns:
        tf.Tensor, type tf.float32, shape (output_size[0], output_size[1], 2),
        tensor where entry (i, j) gives the (x, y) coordinate of the grid point.
    """
    vs, us = _grid_prepare(output_size, ulim, vlim)
    
    su, cu = tf.sin(us), tf.cos(us)
    sv, cv = tf.sin(vs), tf.cos(vs)
    xs = cu * sv / (np.sqrt(2.) - cu * cv)
    ys = su / (np.sqrt(2.) - cu * cv)
    return tf.stack([xs, ys], 2)