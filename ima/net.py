import jax
import jax.numpy as jnp
import haiku as hk


def mlp(hidden_units, initializer='None', act='lipswish', name=None) -> hk.Sequential:
    """
    Returns an haiku MLP with relu nonlinearlties and a number
    of hidden units specified by an array
    :param hidden_units: Array containing number of hidden units of each layer
    :param initializer: if 'None' weights and biases of last layer are
    initialized as zero; else, they are initialized with initializer, from hk.initializers
    :param act: Activation function, can be relu, elu or lipswish
    :param name: Name of hidden layer
    :return: MLP as hk.Sequential
    """
    if act == 'relu':
        act_fn = jax.nn.relu
    elif act == 'lipswish':
        act_fn = None
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
        if act == 'lipswish':
            act_fn = LipSwish(name=prefix + 'lipswish_' + str(i))
        layer_name = prefix + 'linear_' + str(i)
        layers += [hk.Linear(hidden_units[i], name=layer_name), act_fn]
    layer_name = prefix + 'linear_' + str(i + 1)
    if initializer=='None':
        layers += [hk.Linear(hidden_units[-1], w_init=jnp.zeros, b_init=jnp.zeros,
                             name=layer_name)]
    else:
        layers += [hk.Linear(hidden_units[-1], w_init=initializer, b_init=initializer,
                             name=layer_name)]
    return hk.Sequential(layers)

class LipSwish(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        beta = hk.get_parameter('beta', shape=[1], dtype=x.dtype, init=jnp.zeros)
        return jax.nn.swish(jax.nn.softplus(beta + 0.5) * x) / 1.1