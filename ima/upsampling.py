from jax import numpy as jnp

import distrax



class Pad(distrax.Bijector):
    def __init__(self, pad):
        if isinstance(pad[0], int):
            pad = (pad,)
        super().__init__(len(pad))
        self.pad = pad

    def forward_and_log_det(self, x):
        pad = (len(x.shape) - len(self.pad)) * ((0, 0),) + self.pad
        y = jnp.pad(x, pad)
        return y, jnp.zeros(x.shape[0])

    def inverse_and_log_det(self, y):
        slices = (len(y.shape) - len(self.pad)) * [slice(0, None)]
        for p in self.pad:
            e = None if p[1] == 0 else -p[1]
            slices.append(slice(p[0], e))
        return y[tuple(slices)], jnp.zeros(y.shape[0])