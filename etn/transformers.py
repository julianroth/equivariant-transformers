"""
Transformer modules.
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from .coordinates import *


class Transformer(tf.keras.Model):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=identity_grid,
                 ulim=None, vlim=None,
                 return_u=True, return_v=True,
                 periodic_u=False, periodic_v=False,
                 rescale=True,
                 padding_mode='border',
                 **kwargs):
        """Transformer module base class.

        Args:
            predictor_cls: Callable, instantiates an nn.Module instance for predicting
                pose parameters.
            input_shape: int, Size of channel dimension of input tensor.
            nf: int, Number of filters for instantiating pose predictor.
            coords: Callable, coordinate transformation
            ulim: (float, float), limits of u coordinate
            vlim: (float, float), limits of v coordinate
            return_u: bool, whether to return a prediction for the u coordinate.
            return_v: bool, whether to return a prediction for the v coordinate.
            periodic_u: bool, whether the u coordinate is periodic.
            periodic_v: bool, whether the v coordinate is periodic.
            rescale: bool, whether to scale the predicted u and v by half the range
                of ulim and vlim. Useful for pose predictors that return values in [-1, 1].
        """
        super().__init__()
        self.coords = coords
        self.ulim = ulim
        self.vlim = vlim
        self.return_u = return_u
        self.return_v = return_v
        self.periodic_u = periodic_u
        self.periodic_v = periodic_v
        self.rescale = rescale
        
        self.num_outputs = 2 if (return_u and return_v) else 1
        self.predictor = predictor_cls(
            input_shape=input_shape,
            nf=nf,
            periodic_u=periodic_u,
            periodic_v=periodic_v,
            return_u=return_u,
            return_v=return_v,
            num_outputs=self.num_outputs,
            **kwargs)
    
    def transform_from_params(self, *params):
        """Returns a transformation function from the given parameters"""
        return NotImplemented
    
    def call(self, inputs, training=None, mask=None):
        if not isinstance(inputs, list):
            x = inputs
            grid_size = None
            transform = None
        else:
            x, grid_size, transform = inputs
        if grid_size is None:
            grid_size = tf.shape(x)[-3:-1]
        
        grid = self.coords(grid_size, ulim=self.ulim, vlim=self.vlim)
        n = tf.shape(x)[0]
        grid = tf.tile(tf.expand_dims(grid, axis=0), [n, 1, 1, 1])
        
        grid = (grid + 1.) * (tf.reshape(tf.cast(grid_size, dtype=tf.float32), [1, 1, 1, 2]) / 2.)
        x_tf = tfa.image.resampler(x, grid)
        
        coords, heatmaps = self.predictor(x_tf, training=training)
        
        if self.rescale:
            urad = (self.ulim[1] - self.ulim[0]) / 2.
            vrad = (self.vlim[1] - self.vlim[0]) / 2.
        else:
            urad = 1.
            vrad = 1.
        
        if self.return_u and self.return_v:
            u, v = coords
            u = urad * u
            v = vrad * v
            params = (u, v)
        elif self.return_u:
            u, v = coords, None
            u = urad * u
            params = (u, )
        else:
            u, v = None, coords
            v = vrad * v
            params = (v, )
        
        new_transform = self.transform_from_params(*params)
        if transform is not None:
            new_transform = projective_grid_compose(transform, new_transform)

        return {
            'transform': new_transform,
            'params': params,
            'maps': heatmaps,
        }


class TransformerSequence(tf.keras.Model):
    def __init__(self, *transformers):
        """A container class representing a sequence of Transformer modules to be applied iteratively.

        Args:
            transformers: a sequence of Transformer modules.
        """
        super().__init__()
        self.transformers = transformers
        self.num_outputs = sum(t.num_outputs for t in self.transformers)
    
    def call(self, inputs, training=None, mask=None):
        if not isinstance(inputs, list) or len(inputs) == 1:
            x = inputs
            grid_size = None
            transform = None
        else:
            x, grid_size, transform = inputs
        if grid_size is None:
            grid_size = tf.shape(x)[-3:-1]
        
        transforms = []
        params = []
        heatmaps = []

        # fold over projective modules
        for i, transformer in enumerate(self.transformers):
            out_dict = transformer([x, grid_size, transform])
            transform = out_dict['transform']
            transforms.append(transform)
            params.append(out_dict['params'])
            heatmaps.append(out_dict['maps'])
        
        return {
            'transform': transforms,
            'params': params,
            'maps': heatmaps,
        }
        

# TODO: TransformerParallel

def projective_grid_transform(transformer_mat, grid):
    """Perform the transformation on a grid.

    Args:
        transformer_mat: tf.Tensor, a tensor with dimensions [batch, 3, 3] representing a collection
            of projective transformations.
        grid: tf.Tensor, tensor of shape [batch, height, width, 2] denoting
            the (x, y) coordinates of each of the height x width grid points
            for each grid in the batch.

    Returns:
        A tensor of shape [batch, height, width, 2] representing the transformed grid.
    """
    grid_shape = tf.shape(grid)
    n, h, w = grid_shape[0], grid_shape[1], grid_shape[2]
    ones = tf.ones((n, h, w, 1), dtype=tf.float32)
    coords = tf.concat([grid, ones], axis=-1)
    coords = tf.reshape(coords, [n, h*w, 3])
    transformer_mat = tf.transpose(transformer_mat, [0, 2, 1])
    coords = coords @ transformer_mat
    coords = tf.reshape(coords, [n, h, w, 3])
    grid_tf = coords[..., :2] / (coords[..., -1:] + 1e-8)
    return grid_tf


def projective_grid_compose(transformer_mat, other_transformer_mat):
    """Compose with another transformation"""
    return transformer_mat @ other_transformer_mat

class Translation(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=identity_grid,
                 ulim=(-1, 1),
                 vlim=(-1, 1),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         **kwargs)
    
    def transform_from_params(self, *params):
        tx, ty = params
        
        ones = tf.ones_like(tx)
        zeros = tf.zeros_like(tx)
        # transformation matrix
        # 1  0  tx
        # 0  1  ty
        # 0  0  1
        mat = tf.stack([ones, zeros, tx,
                         zeros, ones, ty,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class Rotation(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=polar_grid,
                 ulim=(0., np.sqrt(2.)),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         periodic_v=True,
                         **kwargs)
    
    def transform_from_params(self, *params):
        angle = params[0]
        ca, sa = tf.cos(angle), tf.sin(angle)

        ones = tf.ones_like(angle)
        zeros = tf.zeros_like(angle)
        # transformation matrix
        # ca  -sa  0
        # sa   ca  0
        # 0    0   1
        mat = tf.stack([ca, -sa, zeros,
                         sa, ca, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class Scale(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=logpolar_grid,
                 ulim=(-np.log(10.), np.log(2.) / 2.),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         periodic_v=True,
                         **kwargs)
    
    def transform_from_params(self, *params):
        scale = tf.exp(params[0])

        ones = tf.ones_like(scale)
        zeros = tf.zeros_like(scale)
        # transformation matrix
        # scale 0      0
        # 0     scale  0
        # 0     0      1
        mat = tf.stack([scale, zeros, zeros,
                         zeros, scale, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class RotationScale(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=logpolar_grid,
                 ulim=(-np.log(10.), np.log(2.) / 2.),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         periodic_v=True,
                         **kwargs)
    
    def transform_from_params(self, *params):
        scale, angle = params
        scale = tf.exp(scale)
        ca, sa = tf.cos(angle), tf.sin(angle)

        ones = tf.ones_like(scale)
        zeros = tf.zeros_like(scale)
        # transformation matrix
        # scale*ca  -scale*sa  0
        # scale*sa   scale*ca  0
        # 0          0         1
        mat = tf.stack([scale*ca, -scale*sa, zeros,
                         scale*sa, scale*ca, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class ShearX(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=shearx_grid,
                 ulim=(-1, 1),
                 vlim=(-5, 5),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        shear = params[0]

        ones = tf.ones_like(shear)
        zeros = tf.zeros_like(shear)
        # transformation matrix
        # 1  shear  0
        # 0  1      0
        # 0  0      1
        mat = tf.stack([ones, shear, zeros,
                         zeros, ones, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class ShearY(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=sheary_grid,
                 ulim=(-1, 1),
                 vlim=(-5, 5),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        shear = params[0]

        ones = tf.ones_like(shear)
        zeros = tf.zeros_like(shear)
        # transformation matrix
        # 1     0  0
        # shear 1  0
        # 0     0  1
        mat = tf.stack([ones, zeros, zeros,
                         shear, ones, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class ScaleX(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=scalex_grid,
                 ulim=(-np.log(10.), 0),
                 vlim=(-1, 1),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        scale = tf.exp(params[0])

        ones = tf.ones_like(scale)
        zeros = tf.zeros_like(scale)
        # transformation matrix
        # scale  0  0
        # 0      1  0
        # 0      0  1
        mat = tf.stack([scale, zeros, zeros,
                         zeros, ones, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class ScaleY(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=scalex_grid,
                 ulim=(-np.log(10.), 0),
                 vlim=(-1, 1),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        scale = tf.exp(params[0])

        ones = tf.ones_like(scale)
        zeros = tf.zeros_like(scale)
        # transformation matrix
        # 1  0      0
        # 0  scale  0
        # 0  0      1
        mat = tf.stack([ones, zeros, zeros,
                         zeros, scale, zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class HyperbolicRotation(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=hyperbolic_grid,
                 ulim=(-np.sqrt(0.5), np.sqrt(0.5)),
                 vlim=(-np.log(6.), np.log(6.)),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        scale = tf.exp(params[0])

        ones = tf.ones_like(scale)
        zeros = tf.zeros_like(scale)
        # transformation matrix
        # scale  0        0
        # 0      1/scale  0
        # 0      0        1
        mat = tf.stack([scale, zeros, zeros,
                         zeros, tf.math.reciprocal(scale), zeros,
                         zeros, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class PerspectiveX(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=perspectivex_grid,
                 ulim=(1, 7),
                 vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        perspective = params[0]

        ones = tf.ones_like(perspective)
        zeros = tf.zeros_like(perspective)
        # transformation matrix
        # 1            0  0
        # 0            1  0
        # perspective  0  1
        mat = tf.stack([ones, zeros, zeros,
                         zeros, ones, zeros,
                         perspective, zeros, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat


class PerspectiveY(Transformer):
    def __init__(self, predictor_cls, input_shape, nf,
                 coords=perspectivey_grid,
                 ulim=(1, 7),
                 vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         input_shape=input_shape,
                         nf=nf,
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
    
    def transform_from_params(self, *params):
        perspective = params[0]

        ones = tf.ones_like(perspective)
        zeros = tf.zeros_like(perspective)
        # transformation matrix
        # 1  0           0
        # 0  1           0
        # 0  perspective 1
        mat = tf.stack([ones, zeros, zeros,
                         zeros, ones, zeros,
                         zeros, perspective, ones], axis=1)
        mat = tf.reshape(mat, [-1, 3, 3])
        
        return mat
