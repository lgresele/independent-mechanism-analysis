import jax
import jax.numpy as jnp
import haiku as hk
import distrax

from . import net


class VAE:
    def __init__(self, prior, hidden_units_encoder, hidden_units_decoder,
                 act='relu'):
        """
        Constructor
        :param prior: Prior of the VAE, must be able to compute log_prob
        and sample from
        :param hidden_units_encoder: List of hidden units of encoder
        :param hidden_units_decoder: List of hidden units of decoder
        :param act: Activation function used in encoder and decoder,
        for options see net.mlp
        """
        self.encoder = net.mlp(hidden_units_encoder, act=act)
        self.decoder = net.mlp(hidden_units_decoder, act=act)


