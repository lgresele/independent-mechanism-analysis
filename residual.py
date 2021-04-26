import distrax
import jax
import jax.numpy as jnp
import haiku as hk

class TriangularResidual(distrax.Bijector):

    def __init__(self, hidden_units, zeros=True, brute_force_log_det=False):
        super().__init__(1)
        self.net = mlp(hidden_units, zeros)
        self.brute_force_log_det = brute_force_log_det
        if self.brute_force_log_det:
            self.net_jac = jax.jacfwd(self.net)

    def forward_and_log_det(self, x):
        y = x + self.net(x)
        logdet = self._logdetgrad(x)
        return y, logdet

    def inverse_and_log_det(self, y):
        # Optional. Can be omitted if inverse methods are not needed.
        x = self._inverse_fix_point(y)
        logdet = -self._logdetgrad(x)
        return x, logdet

    def _inverse_fix_point(self, y, atol=1e-5, rtol=1e-5, max_iter=1000):
        """
        Iterative inverse of residual net operation y = x + net(x)
        :param y: y to be inverted
        :param atol: Absolute tolerance
        :param rtol: Relative tolerance
        :param max_iter: Maximum number of iterations to apply
        :return: Inverse, i.e. x
        """
        x, x_prev= y - self.net(y), y
        i = 0
        tol = atol + jnp.abs(y) * rtol
        while not jnp.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.net(x), x
            i += 1
            if i > max_iter:
                break
        return x

    def _logdetgrad(self, x):
        if self.brute_force_log_det:
            jac = self.net_jac(x)
            det = jax.abs((jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1)
                          - jac[:, 0, 1] * jac[:, 1, 0])
            log_det = jax.log(det).reshape(-1, 1)
        else:
            log_det = 0
        return log_det


def mlp(hidden_units, zeros=True) -> hk.Sequential:
    """
    Returns an haiku MLP with relu nonlinearlties and a number
    of hidden units specified by an array
    :param hidden_units: Array containing number of hidden units of each layer
    :param zeros: Flag, if true weights and biases of last layer are
    initialized as zero
    :return: MLP as hk.Sequential
    """
    layers = []
    for i in range(len(hidden_units) - 1):
        layers += [hk.Linear(hidden_units[i]), jax.nn.relu]
    layers += [hk.Linear(hidden_units[-1], w_init=jnp.zeros, b_init=jnp.zeros)]
    return hk.Sequential(layers)