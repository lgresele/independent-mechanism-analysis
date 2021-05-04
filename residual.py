import distrax
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

class TriangularResidual(distrax.Bijector):

    def __init__(self, hidden_units, zeros=True, brute_force_log_det=True,
                 act='elu', name='residual'):
        super().__init__(1)
        self.net = mlp(hidden_units, zeros, act, name)
        self.brute_force_log_det = brute_force_log_det
        if self.brute_force_log_det:
            self.net_jac = jax.vmap(jax.jacfwd(self.net))

    def forward_and_log_det(self, x):
        y = self._inverse_fix_point(x)
        logdet = -self._logdetgrad(y)
        return y, logdet

    def forward(self, x):
        return self._inverse_fix_point(x)

    def inverse_and_log_det(self, y):
        x = y + self.net(y)
        logdet = self._logdetgrad(y)
        return x, logdet

    def inverse(self, y):
        return y + self.net(y)

    def _inverse_fix_point(self, y, atol=1e-5, rtol=1e-5, max_iter=1000):
        """
        Iterative inverse of residual net operation y = x + net(x)
        :param y: y to be inverted
        :param atol: Absolute tolerance
        :param rtol: Relative tolerance
        :param max_iter: Maximum number of iterations to apply
        :return: Inverse, i.e. x
        """
        x, x_prev = y - self.net(y), y
        i = jnp.array(0)
        tol = atol + jnp.abs(y) * rtol
        def fix_point_update(arg):
            i, x, x_prev, tol = arg
            x, x_prev = y - self.net(x), x
            tol = atol + jnp.abs(y) * rtol
            i += 1
            return [i, x, x_prev, tol]
        def fix_point_cond(arg):
            i, x, x_prev, tol = arg
            return jnp.logical_or(i < max_iter, jnp.all((x - x_prev) ** 2 / tol > 1))
        i, x, x_prev, tol = jax.lax.while_loop(fix_point_cond, fix_point_update,
                                               [i, x, x_prev, tol])
        return x

    def _logdetgrad(self, x):
        if self.brute_force_log_det:
            jac = self.net_jac(x)
            det = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) \
                  - jac[:, 0, 1] * jac[:, 1, 0]
            log_det = jnp.log(jnp.abs(det))
        else:
            log_det = 0
        return log_det


def mlp(hidden_units, zeros=True, act='lipswish', name=None) -> hk.Sequential:
    """
    Returns an haiku MLP with relu nonlinearlties and a number
    of hidden units specified by an array
    :param hidden_units: Array containing number of hidden units of each layer
    :param zeros: Flag, if true weights and biases of last layer are
    initialized as zero
    :param act: Activation function, can be relu, elu or lipswish
    :param name: Name of hidden layer
    :return: MLP as hk.Sequential
    """
    if act == 'relu':
        act_fn = jax.nn.relu
    elif act == 'lipswish':
        act_fn = lambda x: jax.nn.swish(x) / 1.1
    elif act == 'elu':
        act_fn = jax.nn.elu
    else:
        raise NotImplementedError('The activation function ' + act
                                  + ' is not implemented.')
    layers = []
    if name is None:
        prefix = ''
    else:
        prefix = name + '_'
    for i in range(len(hidden_units) - 1):
        layer_name = prefix + 'linear_' + str(i)
        layers += [hk.Linear(hidden_units[i], name=layer_name), act_fn]
    layer_name = prefix + 'linear_' + str(i + 1)
    if zeros:
        layers += [hk.Linear(hidden_units[-1], w_init=jnp.zeros, b_init=jnp.zeros,
                             name=layer_name)]
    else:
        layers += [hk.Linear(hidden_units[-1], name=layer_name)]
    return hk.Sequential(layers)


def masks_triangular_weights(hidden_units):
    total_hu = jnp.sum(jnp.array(hidden_units))
    ndims = len(hidden_units)
    # Create masks
    mask0 = np.ones((ndims, total_hu))
    mask1 = np.ones((total_hu, total_hu))
    mask2 = np.ones((total_hu, ndims))
    # Make masks block triangular
    thu = 0
    hu = hidden_units[0]
    for i in range(1, ndims):
        thu += hu
        hu_ = hidden_units[i]
        mask1[thu:(thu + hu_), :thu] = np.zeros((hu_, thu))
        mask0[i, :thu] = np.zeros(thu)
        mask2[thu:, i - 1] = np.zeros(total_hu - thu)
        hu = hu_
    return [jnp.array(mask0), jnp.array(mask1), jnp.array(mask2)]


def make_weights_triangular(params, masks, keywords=['residual', 'linear']):
    params_new = {}
    for key, item in params.items():
        if np.all([keyw in key for keyw in keywords]):
            params_new[key] = {}
            s = item['w'].shape
            if s == masks[1].shape:
                params_new[key]['w'] = item['w'] * masks[1]
            elif s == masks[0].shape:
                params_new[key]['w'] = item['w'] * masks[0]
            else:
                params_new[key]['w'] = item['w'] * masks[2]
            params_new[key]['b'] = item['b']
            params_new[key] = hk.data_structures.to_immutable_dict(params_new[key])
        else:
            params_new[key] = item
    return hk.data_structures.to_immutable_dict(params_new)


def spectral_norm_init(params, rng_key):
    uv = {}
    for key, item in params.items():
        uv[key] = {}
        rng_key, rng_subkey = jax.random.split(rng_key)
        s = params[key]['w'].shape
        u = jax.random.normal(rng_subkey, (s[1],))
        uv[key]['u'] = u / jnp.linalg.norm(u)
        rng_key, rng_subkey = jax.random.split(rng_key)
        v = jax.random.normal(rng_subkey, (s[0],))
        uv[key]['v'] = v / jnp.linalg.norm(v)
    return uv

def spectral_normalization(params, uv, coeff=0.97, max_iter=100, atol=1e-3, rtol=1e-3):
    params_new = {}
    # Prepare functions for power iteration
    def power_iter(arg):
        i, u, v, w = arg
        v_ = jnp.matmul(w, u)
        u_ = jnp.matmul(v, w)
        i, v, v_ = jax.lax.cond(jnp.linalg.norm(v_) > 0,
                                lambda arg: (arg[0], arg[2] / jnp.linalg.norm(arg[2]), arg[1]),
                                lambda arg: (max_iter, arg[1], arg[2]),
                                [i, v, v_])
        i, u, u_ = jax.lax.cond(jnp.linalg.norm(u_) > 0,
                                lambda arg: (arg[0], arg[2] / jnp.linalg.norm(arg[2]), arg[1]),
                                lambda arg: (max_iter, arg[1], arg[2]),
                                [i, u, u_])
        err_u_ = jnp.concatenate([jnp.linalg.norm(u - u_)[None],
                                  jnp.linalg.norm(u + u_)[None]])
        err_u = jnp.min(err_u_) / (len(u) ** 0.5)
        err_v_ = jnp.concatenate([jnp.linalg.norm(v - v_)[None],
                                  jnp.linalg.norm(v + v_)[None]])
        err_v = jnp.min(err_v_) / (len(v) ** 0.5)
        tol_u = atol + rtol * jnp.max(jnp.abs(u))
        tol_v = atol + rtol * jnp.max(jnp.abs(v))
        i = jax.lax.cond(jnp.logical_and(err_u < tol_u, err_v < tol_v),
                         lambda arg: max_iter, lambda arg: arg, i)
        i += 1
        return [i, u, v, w]
    def power_iter_cond(arg):
        return arg[0] < max_iter

    # Do spectral normalization
    for key, item in params.items():
        params_new[key] = {}
        u = uv[key]['u']
        v = uv[key]['v']
        w = item['w']
        # Power iteration
        i = jnp.array(0)
        i, u, v, w = jax.lax.while_loop(power_iter_cond, power_iter, [i, u, v, w])
        # Update variables
        sigma = jnp.abs(jnp.dot(v, jnp.matmul(w, u)))[None]
        uv[key]['u'] = u
        uv[key]['v'] = v
        factor = jnp.min(jnp.concatenate([coeff / sigma, jnp.ones_like(sigma)]))
        params_new[key]['w'] = item['w'] * factor
        params_new[key]['b'] = item['b']
        params_new[key] = hk.data_structures.to_immutable_dict(params_new[key])
    return hk.data_structures.to_immutable_dict(params_new), uv