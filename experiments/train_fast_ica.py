import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import distrax
import haiku as hk
from ima.utils import get_config
from ima.metrics import observed_data_likelihood, jacobian_amari_distance
from ima.mixing_functions import build_moebius_transform
from ima.solve_hungarian import SolveHungarian

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
ckpt_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
data_dir = os.path.join(root, 'data')
# Create dirs if not existent
for dir in [ckpt_dir, plot_dir, log_dir, data_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)


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

# Plot the sources
if D == 2:
    from ima.plotting import cart2pol, scatterplot_variables

    _, colors_train = cart2pol(S_train[:, 0], S_train[:, 1])
    _, colors_test = cart2pol(S_test[:, 0], S_test[:, 1])

    scatterplot_variables(S_train, 'Sources (train)', colors=colors_train, savefig=True,
                          fname=os.path.join(plot_dir, 'data_sources_train.png'), show=False)
    plt.close()
    scatterplot_variables(S_test, 'Sources (test)', colors=colors_test, savefig=True,
                          fname=os.path.join(plot_dir, 'data_sources_test.png'), show=False)
    plt.close()

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
epsilon = config['data']['epsilon']

mixing, unmixing = build_moebius_transform(alpha, A, a, b, epsilon=epsilon)
mixing_batched = jax.vmap(mixing)

X_train = mixing_batched(S_train)
X_test = mixing_batched(S_test)

#mean_train = jnp.mean(X_train, axis=0)
#std_train = jnp.std(X_train, axis=0)
#X_train -= mean_train
#X_test -= mean_train

#b = b - mean_train

mixing, unmixing = build_moebius_transform(alpha, A, a, b, epsilon=epsilon)
unmixing_batched = jax.vmap(unmixing)
jac_mixing_batched = jax.vmap(jax.jacfwd(mixing))


# Save parameters of Moebius transformation
jnp.save(os.path.join(data_dir, 'moebius_transform_params.npy'),
         {'A': A, 'a': a, 'b': b})

# Compute true log probability
true_log_p_fn = jax.vmap(lambda arg: observed_data_likelihood(arg, jax.jacfwd(unmixing)))
true_log_p_train = true_log_p_fn(X_train)
true_log_p_test = true_log_p_fn(X_test)

# Correct for scaling
true_log_p_train_avg = jnp.mean(true_log_p_train)
true_log_p_test_avg = jnp.mean(true_log_p_test)

if D == 2:
    scatterplot_variables(X_train, 'Observations (train)', colors=colors_train, savefig=True,
                          fname=os.path.join(plot_dir, 'data_observations_train.png'), show=False)
    plt.close()
    scatterplot_variables(X_test, 'Observations (test)', colors=colors_test, savefig=True,
                          fname=os.path.join(plot_dir, 'data_observations_test.png'), show=False)
    plt.close()

# Save data
jnp.save(os.path.join(data_dir, 'observation_train.npy'), X_train)
jnp.save(os.path.join(data_dir, 'observation_test.npy'), X_test)
#jnp.save(os.path.join(data_dir, 'observation_mean_std.npy'),
#         {'mean': mean_train, 'std': std_train})


# Do ICA
from sklearn.decomposition import FastICA
ica = FastICA(n_components=D, whiten=True)

S_rec_train = ica.fit_transform(X_train)
A_ = ica.mixing_
B_ = ica.components_

# cima functions
if D == 2:
    from ima.metrics import cima
else:
    from ima.metrics import cima_higher_d as cima


# Logging & plotting
loss_hist = np.zeros((0, 2))
cima_train_hist = np.zeros((0, 2))
cima_test_hist = np.zeros((0, 2))
amari_train_hist = np.zeros((0, 2))
amari_test_hist = np.zeros((0, 2))
mcc_train_hist = np.zeros((0, 3))
mcc_test_hist = np.zeros((0, 3))

if D == 2:
    npoints = 300
    plot_max = X_train.max(axis=0)
    plot_min = X_train.min(axis=0)
    x, y = jnp.linspace(plot_min[0], plot_max[0], npoints), jnp.linspace(plot_min[1], plot_max[1], npoints)
    xx, yy = jnp.meshgrid(x, y)
    zz = jnp.column_stack([xx.reshape(-1), yy.reshape(-1)])



# Measures
it = 0

jac_fn_eval = jax.vmap(jax.jacfwd(lambda y: jnp.dot(y, jnp.asarray(B_).T)))
c = jnp.mean(cima(X_train, jac_fn_eval))
cima_append = np.array([[it + 1, c.item()]])
cima_train_hist = np.concatenate([cima_train_hist, cima_append])
np.savetxt(os.path.join(log_dir, 'cima_train.csv'), cima_train_hist,
           delimiter=',', header='it,cima', comments='')

c = jnp.mean(cima(X_test, jac_fn_eval))
cima_append = np.array([[it + 1, c.item()]])
cima_test_hist = np.concatenate([cima_test_hist, cima_append])
np.savetxt(os.path.join(log_dir, 'cima_test.csv'), cima_test_hist,
           delimiter=',', header='it,cima', comments='')

amari = jacobian_amari_distance(X_train, jac_fn_eval, jac_mixing_batched,
                                unmixing_batched)
amari_append = np.array([[it + 1, amari.item()]])
amari_train_hist = np.concatenate([amari_train_hist, amari_append])
np.savetxt(os.path.join(log_dir, 'amari_train.csv'), amari_train_hist,
           delimiter=',', header='it,amari', comments='')

amari = jacobian_amari_distance(X_test, jac_fn_eval, jac_mixing_batched,
                                unmixing_batched)
amari_append = np.array([[it + 1, amari.item()]])
amari_test_hist = np.concatenate([amari_test_hist, amari_append])
np.savetxt(os.path.join(log_dir, 'amari_test.csv'), amari_test_hist,
           delimiter=',', header='it,amari', comments='')

av_corr_spearman, _, _ = SolveHungarian(recov=S_rec_train[::10, :], source=S_train[::10, :],
                                        correlation='Spearman')
av_corr_pearson, _, _ = SolveHungarian(recov=S_rec_train[::10, :], source=S_train[::10, :],
                                       correlation='Pearson')
mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_pearson.item()]])
mcc_train_hist = np.concatenate([mcc_train_hist, mcc_append])
np.savetxt(os.path.join(log_dir, 'mcc_train.csv'), mcc_train_hist,
           delimiter=',', header='it,spearman,pearson', comments='')

S_rec_test = ica.transform(X_test)
av_corr_spearman, _, _ = SolveHungarian(recov=S_rec_test[::10, :], source=S_test[::10, :],
                                        correlation='Spearman')
av_corr_pearson, _, _ = SolveHungarian(recov=S_rec_test[::10, :], source=S_test[::10, :],
                                       correlation='Pearson')
mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_pearson.item()]])
mcc_test_hist = np.concatenate([mcc_test_hist, mcc_append])
np.savetxt(os.path.join(log_dir, 'mcc_test.csv'), mcc_test_hist,
           delimiter=',', header='it,spearman,pearson', comments='')

# Plots
if D == 2:
    scatterplot_variables(S_rec_train, 'Reconstructed sources (train)',
                          colors=colors_train, savefig=True, show=False,
                          fname=os.path.join(plot_dir, 'rec_sources_train_%06i.png' % (it + 1)))
    plt.close()

    scatterplot_variables(S_rec_test, 'Reconstructed sources (test)',
                          colors=colors_test, savefig=True, show=False,
                          fname=os.path.join(plot_dir, 'rec_sources_test_%06i.png' % (it + 1)))
    plt.close()


# Save transform
jnp.save(os.path.join(ckpt_dir, 'A_%06i.npy' % (it + 1)), A_)
jnp.save(os.path.join(ckpt_dir, 'B_%06i.npy' % (it + 1)), B_)