import distrax
import jax
import jax.numpy as jnp
import haiku as hk

class TriangularResidual(distrax.Bijector):

    def __init__(self, net):
        super().__init__(1)
        self.net = net

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
        return 0


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