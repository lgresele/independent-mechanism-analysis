import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import distrax
import haiku as hk
from residual import TriangularResidual, Scaling, spectral_norm_init, spectral_normalization, masks_triangular_weights, make_weights_triangular
from utils import get_config
from metrics import observed_data_likelihood
from mixing_functions import build_moebius_transform

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
    from plotting import cart2pol, scatterplot_variables

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
# Save data
jnp.save(os.path.join(data_dir, 'moebius_transform_params.npy'), {'A': A, 'a': a})

mixing, unmixing = build_moebius_transform(alpha, A, a, b, epsilon=2)
mixing_batched = jax.vmap(mixing)

X_train = mixing_batched(S_train)
X_test = mixing_batched(S_test)

# Compute true log probability
true_log_p_fn = jax.vmap(lambda arg: observed_data_likelihood(arg, jax.jacfwd(unmixing)))
true_log_p_train = true_log_p_fn(X_train)
true_log_p_test = true_log_p_fn(X_test)

mean_train = jnp.mean(X_train, axis=0)
std_train = jnp.std(X_train, axis=0)
X_train -= mean_train
X_test -= mean_train

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
jnp.save(os.path.join(data_dir, 'observation_mean_std.npy'),
         {'mean': mean_train, 'std': std_train})


# Setup model
n_layers = config['model']['flow_layers']
hidden_units = config['model']['nn_layers'] * [config['model']['nn_hidden_units']]

# Define model functions
def log_prob(x):
    base_dist = distrax.Independent(distrax.Normal(loc=jnp.zeros(D), scale=jnp.ones(D)),
                                    reinterpreted_batch_ndims=1)
    flows = distrax.Chain([TriangularResidual(hidden_units + [D], name='residual_' + str(i))
                           for i in range(n_layers)] + [Scaling(D)])
    model = distrax.Transformed(base_dist, flows)
    return model.log_prob(x)

def inv_map_fn(x):
    flows = distrax.Chain([TriangularResidual(hidden_units + [D], name='residual_' + str(i))
                           for i in range(n_layers)] + [Scaling(D)])
    return flows.inverse(x)

# Init model
logp = hk.transform(log_prob)
key, subkey = jax.random.split(key)
params = logp.init(subkey, jnp.array(np.random.randn(5, D)))
inv_map = hk.transform(inv_map_fn)

# Initialize scale
params_ = hk.data_structures.to_mutable_dict(params)
params_['~']['Scaling_log_scale'] = jnp.log(std_train)
params = hk.data_structures.to_immutable_dict(params_)

# Make triangular
hu_masks = [hidden_units[0] // D for _ in range(D)]
remainder = hidden_units[0] - np.sum(hu_masks)
for ind in range(remainder):
    hu_masks[-(ind + 1)] = hu_masks[-(ind + 1)] + 1
masks = masks_triangular_weights(hu_masks)
params = make_weights_triangular(params, masks)

# Apply spectral normalization
key, subkey = jax.random.split(key)
uv = spectral_norm_init(params, subkey)
params, uv = spectral_normalization(params, uv)


# Prepare training

# Performance measures
def loss(params, x):
    ll = logp.apply(params, None, x)
    return -jnp.mean(ll)

# cima functions
if D == 2:
    from metrics import cima
else:
    from metrics import cima_higher_d as cima

# Optimizer
num_iter = config['training']['num_iter']
log_iter = config['training']['log_iter']
ckpt_iter = config['training']['checkpoint_iter']
batch_size = config['training']['batch_size']
lr = config['training']['lr']

opt_init, opt_update, get_params = adam(step_size=lr)
opt_state = opt_init(params)

# Logging & plotting
loss_hist = np.zeros((0, 2))
log_p_train_hist = np.zeros((0, 2))
log_p_test_hist = np.zeros((0, 2))
cima_train_hist = np.zeros((0, 2))
cima_test_hist = np.zeros((0, 2))
kld_train_hist = np.zeros((0, 2))
kld_test_hist = np.zeros((0, 2))

if D == 2:
    npoints = 300
    plot_max = X_train.max(axis=0)
    plot_min = X_train.min(axis=0)
    x, y = jnp.linspace(plot_min[0], plot_max[0], npoints), jnp.linspace(plot_min[1], plot_max[1], npoints)
    xx, yy = jnp.meshgrid(x, y)
    zz = jnp.column_stack([xx.reshape(-1), yy.reshape(-1)])

# Iteration
@jax.jit
def step(it_, opt_state_, uv_, x_):
    params_ = get_params(opt_state_)
    params_ = make_weights_triangular(params_, masks) # makes Jacobian triangular
    params_, uv_ = spectral_normalization(params_, uv_)
    params_flat = jax.tree_util.tree_flatten(params_)[0]
    for ind in range(len(params_flat)):
        opt_state.packed_state[ind][0] = params_flat[ind]
    value, grads = jax.value_and_grad(loss, 0)(params_, x_)
    opt_state_ = opt_update(it_, grads, opt_state_)
    return value, opt_state_, uv_


# Training
for it in range(num_iter):
    x = X_train[np.random.choice(N, batch_size)]
    loss_val, opt_state, uv = step(it, opt_state, uv, x)

    loss_append = np.array([[it + 1, loss_val.item()]])
    loss_hist = np.concatenate([loss_hist, loss_append])

    if (it + 1) % log_iter == 0:
        # Save loss
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
                   delimiter=',', header='it,loss', comments='')

        # Get params
        params_eval = get_params(opt_state)
        params_eval = make_weights_triangular(params_eval, masks)
        params_eval, _ = spectral_normalization(params_eval, uv)

        # Measures
        log_p = -loss(params_eval, X_train)
        log_p_append = np.array([[it + 1, log_p.item()]])
        log_p_train_hist = np.concatenate([log_p_train_hist, log_p_append])
        np.savetxt(os.path.join(log_dir, 'log_p_train.csv'), log_p_train_hist,
                   delimiter=',', header='it,log_p', comments='')

        kld = true_log_p_train_avg - log_p
        kld_append = np.array([[it + 1, kld.item()]])
        kld_train_hist = np.concatenate([kld_train_hist, kld_append])
        np.savetxt(os.path.join(log_dir, 'kld_train.csv'), kld_train_hist,
                   delimiter=',', header='it,kld', comments='')

        log_p = -loss(params_eval, X_test)
        log_p_append = np.array([[it + 1, log_p.item()]])
        log_p_test_hist = np.concatenate([log_p_test_hist, log_p_append])
        np.savetxt(os.path.join(log_dir, 'log_p_test.csv'), log_p_test_hist,
                   delimiter=',', header='it,log_p', comments='')

        kld = true_log_p_test_avg - log_p
        kld_append = np.array([[it + 1, kld.item()]])
        kld_test_hist = np.concatenate([kld_test_hist, kld_append])
        np.savetxt(os.path.join(log_dir, 'kld_test.csv'), kld_test_hist,
                   delimiter=',', header='it,kld', comments='')

        jac_fn_eval = jax.vmap(jax.jacfwd(lambda y: inv_map.apply(params_eval, None, y)))
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

        # Plots
        if D == 2:
            S_rec = inv_map.apply(params_eval, None, X_train)
            S_rec_uni = jnp.column_stack([jax.scipy.stats.norm.cdf(S_rec[:, 0]),
                                          jax.scipy.stats.norm.cdf(S_rec[:, 1])])
            S_rec_uni -= 0.5
            scatterplot_variables(S_rec_uni, 'Reconstructed sources (train)',
                                  colors=colors_train, savefig=True, show=False,
                                  fname=os.path.join(plot_dir, 'rec_sources_train_%06i.png' % (it + 1)))
            plt.close()

            S_rec = inv_map.apply(params_eval, None, X_test)
            S_rec_uni = jnp.column_stack([jax.scipy.stats.norm.cdf(S_rec[:, 0]),
                                          jax.scipy.stats.norm.cdf(S_rec[:, 1])])
            S_rec_uni -= 0.5
            scatterplot_variables(S_rec_uni, 'Reconstructed sources (test)',
                                  colors=colors_test, savefig=True, show=False,
                                  fname=os.path.join(plot_dir, 'rec_sources_test_%06i.png' % (it + 1)))
            plt.close()

            prob = jnp.exp(logp.apply(params_eval, None, zz))
            plt.pcolormesh(np.array(xx), np.array(yy), np.array(prob.reshape(npoints, npoints)))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(plot_dir, 'pdf_%06i.png' % (it + 1)))
            plt.close()


    if (it + 1) % ckpt_iter == 0:
        params_save = get_params(opt_state)
        params_save = make_weights_triangular(params_save, masks)
        params_save, uv_save = spectral_normalization(params_save, uv)

        jnp.save(os.path.join(ckpt_dir, 'model_%06i.npy' % (it + 1)),
                 hk.data_structures.to_mutable_dict(params_save))

        jnp.save(os.path.join(ckpt_dir, 'uv_%06i.npy' % (it + 1)), uv_save)