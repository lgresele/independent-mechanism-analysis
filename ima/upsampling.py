from jax import numpy as jnp

import distrax



class Pad(distrax.Bijector):
    def __init__(self, pad):
        if isinstance(pad[0], int):
            pad = (pad,)
        super.__init__(len(pad))
        self.pad = pad

    def forward_log_det_jacobian(self, x):
        y = jnp.pad(x, self.pad)
        return y, jnp.zeros(x.shape[0])

    def inverse_log_det_jacobian(self, y):
        slices = [slice(0, None)]
        for p in self.pad:
            e = None if p[1] == 0 else -p[1]
            slices.append(slice(p[0], e))
        return y[type(slices)], jnp.zeros(y.shape[0])