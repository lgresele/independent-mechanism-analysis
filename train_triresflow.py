import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import distrax
import haiku as hk
from residual import TriangularResidual, spectral_norm_init, spectral_normalization, masks_triangular_weights, make_weights_triangular, LipSwish
from utils import get_config

from jax.experimental.optimizers import adam

import argparse
import os

# Parse input arguments
parser = argparse.ArgumentParser(description='Train a triangular residual flow')

parser.add_argument('--config', type=str, default='./config.yaml',
                    help='Path config file specifying model '
                         'architecture and training procedure')

args = parser.parse_args()


# Load config
config = get_config(args.config)

# Set seed
key = jax.random.PRNGKey(config['training']['jax_seed'])
np.random.seed(config['training']['np_seed'])

# Make directories
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
data_dir = os.path.join(root, 'data')


# Data generation
N = config['data']['n']
D = config['data']['dim']

# Generate the samples
key, subkey = jax.random.split(key)
S = jax.random.uniform(subkey, shape=(2 * N, D), minval=0.0, maxval=1.0)
S -= 0.5

# Train test split
S_train = S[:N, :]
S_test = S[N:, :]

# Save data
jnp.save(os.path.join(data_dir, 'sources_train.npy'), S_train)
jnp.save(os.path.join(data_dir, 'sources_test.npy'), S_test)

from plotting import cart2pol, scatterplot_variables

_, colors_train = cart2pol(S_train[:, 0], S_train[:, 1])
_, colors_test = cart2pol(S_test[:, 0], S_test[:, 1])

# Plot the sources
scatterplot_variables(S_train, 'Sources (train)', colors=colors_train, savefig=True,
                      fname=os.path.join(plot_dir, 'data_sources_train.png'))
plt.close()
scatterplot_variables(S_test, 'Sources (test)', colors=colors_train, savefig=True,
                      fname=os.path.join(plot_dir, 'data_sources_test.png'))
plt.close()

from mixing_functions import build_moebius_transform
# Generate a random orthogonal matrix
from scipy.stats import ortho_group # Requires version 0.18 of scipy
A = ortho_group.rvs(dim=D)
A = jnp.array(A)
# Scalar
alpha = 1.0
# Two vectors with data dimensionality
a = []
while len(a) < D:
    s = np.random.randn()
    if np.abs(s) > 0.5:
        a = a + [s]
a = jnp.array(a) # a vector in \RR^D
b = jnp.zeros(D) # a vector in \RR^D
# Save data
jnp.save(os.path.join(data_dir, 'moebius_transform_A.npy'), A)
jnp.save(os.path.join(data_dir, 'moebius_transform_a.npy'), a)

mixing, _ = build_moebius_transform(alpha, A, a, b, epsilon=2)
mixing_batched = jax.vmap(mixing)

X_train = mixing_batched(S_train)
X_test = mixing_batched(S_test)
X_train -= jnp.mean(X_train, axis=0)
X_train /= jnp.std(X_train, axis=0)
X_test -= jnp.mean(X_train, axis=0)
X_test /= jnp.std(X_train, axis=0)

scatterplot_variables(X_train, 'Observations (train)', colors=colors_train, savefig=True,
                      fname=os.path.join(plot_dir, 'data_observations_train.png'))
plt.close()
scatterplot_variables(X_test, 'Observations (test)', colors=colors_train, savefig=True,
                      fname=os.path.join(plot_dir, 'data_observations_test.png'))
plt.close()