import numpy as np
import jax
from jax import numpy as jnp
import distrax
import haiku as hk
from ima.upsampling import Pad
from jax.experimental.optimizers import adam
from tqdm import tqdm
from matplotlib import pyplot as plt
from jax.lib import xla_bridge
from ima.plotting import cart2pol 
import argparse
import os
from ima.utils import get_config
from ima.solve_hungarian import SolveHungarian
import statsmodels.api as sm


# Parse input arguments
parser = argparse.ArgumentParser(description='Train for swiss roll dataset')

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
        os.makedirs(dir)

key, subkey = jax.random.split(key)
N = 1600
base_samples = jax.random.uniform(subkey, shape=(2*N, 2), minval=-0.5, maxval=0.5)


x_train = base_samples[:N,0]
x_test = base_samples[N:,0]
y_train = base_samples[:N,1]
y_test = base_samples[N:,1]

pi = np.pi
obs_train = jnp.stack((x_train*jnp.cos(1/2*pi*(x_train)+pi/2), y_train, x_train*jnp.sin(1/2*pi*(x_train)+pi/2)), axis=1)
obs_test = jnp.stack((x_test*jnp.cos(1/2*pi*(x_test)+pi/2), y_test, x_test*jnp.sin(1/2*pi*(x_test)+pi/2)), axis=1)



# Save data
jnp.save(os.path.join(data_dir, 'sourcesx_train.npy'), x_train)
jnp.save(os.path.join(data_dir, 'sourcesx_test.npy'), x_test)

jnp.save(os.path.join(data_dir, 'sourcesy_train.npy'), y_train)
jnp.save(os.path.join(data_dir, 'sourcesy_test.npy'), y_test)

jnp.save(os.path.join(data_dir, 'mani_train.npy'), obs_train)
jnp.save(os.path.join(data_dir, 'mani_test.npy'), obs_test)

d = 2
D = obs_train.shape[1]
N = obs_train.shape[0]

# Define Real NVP flow with Distrax
def mk_flow(K = 16, nl = 2, hu = 256):
    
    layers_lowdim =[]
    for i in range(K):
        mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear(d, w_init=jnp.zeros, b_init=jnp.zeros)])
        
        def bij_fn_1(params):
            bij = distrax.ScalarAffine(shift=params[..., :d // 2], log_scale=jax.nn.sigmoid(params[..., :d // 2]))
            return distrax.Block(bij, 1)
        
        def bij_fn_2(params):
            if (d % 2):
                bij = distrax.ScalarAffine(shift=params[..., (d // 2) + 1:], log_scale=jax.nn.sigmoid(params[..., (d // 2) + 1:]))
            else:
                bij = distrax.ScalarAffine(shift=params[..., (d // 2):], log_scale=jax.nn.sigmoid(params[..., (d // 2):]))
            return distrax.Block(bij, 1)
        
        if (bool(i%2) == True):
            layers_lowdim.append(distrax.SplitCoupling(d // 2, 1, mlp, bij_fn_2, swap=True))
        else:
            layers_lowdim.append(distrax.SplitCoupling(d // 2, 1, mlp, bij_fn_1, swap=False))
        
        
    flow_lowdim = distrax.Chain(layers_lowdim)
    pad = Pad((0, D - d))
    layers = []
    W = []
    b = []
    for i in range(K):
        
        mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)])
        
        W.append(hk.get_parameter("W"+ str(i), [D, D], init = jnp.zeros))
        b.append(hk.get_parameter("b"+ str(i), [D], init = jnp.zeros))
        W[i] = W[i] - jnp.diag(jnp.diag(W[i])) + jnp.diag(jnp.exp(jnp.diag(W[i])))
        
        
        
        def bij_fn_1(params):
            bij = distrax.ScalarAffine(shift=params[..., :D // 2], log_scale=params[..., :D // 2])
            return distrax.Block(bij, 1)
        
        def bij_fn_2(params):
            if (D % 2):
                bij = distrax.ScalarAffine(shift=params[..., (D // 2) + 1:], log_scale=params[..., (D // 2) + 1:])
            else:
                bij = distrax.ScalarAffine(shift=params[..., D // 2:], log_scale=params[..., D // 2:])
            return distrax.Block(bij, 1)
        
        if (bool(i%2) == True):
            layers.append(distrax.SplitCoupling(D // 2, 1, mlp, bij_fn_2, swap=True))
        else:
            layers.append(distrax.SplitCoupling(D // 2, 1, mlp, bij_fn_1, swap=False))
        layers.append(distrax.LowerUpperTriangularAffine(matrix = W[i], bias = b[i]))
        
        
    flow = distrax.Chain(layers)
    return (flow_lowdim, pad, flow)

def fwd_(x):
    flow_lowdim, pad, flow = mk_flow()
    x = flow_lowdim.forward(x)
    x = pad.forward(x)
    return flow.forward(x)

def inv_(x):
    flow_lowdim, pad, flow = mk_flow()
    
    x = flow.inverse(x)
    x = pad.inverse(x)
    return flow_lowdim.inverse(x)

key, subkey = jax.random.split(key)
fwd = hk.transform(fwd_)
inv = hk.transform(inv_)
params = fwd.init(subkey, jnp.array(np.random.randn(5, d)))

# Loss function

def loss_(args):


    x, lam, beta, gamma = args
    flow_lowdim, pad, flow = mk_flow()
    
    fwd = lambda y: flow.forward(pad.forward(flow_lowdim.forward(y)))
    inv = lambda y: flow_lowdim.inverse(pad.inverse(flow.inverse(y)))
    fn_inv = lambda y: pad.inverse(flow.inverse(y))
    
    #base_dist = distrax.Independent(distrax.Normal(loc=jnp.zeros(d), scale=jnp.ones(d)),
    #                                reinterpreted_batch_ndims=1)
    base_dist = distrax.Independent(distrax.Logistic(loc=jnp.zeros(d), scale=jnp.ones(d)),
                                    reinterpreted_batch_ndims=1)
    #base_dist = distrax.Independent(distrax.Uniform(0., 1.))
    
    jac_fn = jax.vmap(jax.jacfwd(fwd))
    
    z = inv(x)
    fn_inv_obs = fn_inv(x)
    jac = jac_fn(z)
    
    jj = jax.lax.batch_matmul(jnp.transpose(jac, (0, 2, 1)), jac) 
    eye = 0.1*jnp.repeat(jnp.eye(jnp.shape(jj)[1])[jnp.newaxis, :, :], jnp.shape(jj)[0], axis=0)
    jj = jj + eye
     
    chol = jax.vmap(jax.scipy.linalg.cholesky)(jj)
    log_det = jnp.sum(jnp.log(jax.vmap(jnp.diag)(chol)), -1)
    
    
    
    log_det_diag = jnp.sum(jnp.log(jax.vmap(jnp.diag)(jj)), -1)
    
    
    
    
    cima = 0.5*(log_det_diag) - log_det
    diff = jnp.mean((x - fwd(z)) ** 2)
    
    #diff = jnp.mean(jax.vmap(jnp.sqrt)((x - fwd(z)) ** 2))
    
    #import pdb; pdb.set_trace()
    
    return jnp.mean(-lam * (base_dist.log_prob(z) - log_det ) + beta * diff + gamma*cima ), (jnp.mean(-lam * (base_dist.log_prob(z) - log_det)), beta * diff, jnp.mean(gamma*cima)
                                                                                                                    )
    

key, subkey = jax.random.split(key)
loss = hk.transform(loss_)
params = loss.init(subkey, (jnp.array(np.random.randn(5, D)), 1., 1., 1.))
b = jnp.array(np.random.randn(5, D))

@jax.jit
def step(it_, opt_state_, x_, lam_, beta_, gamma_):
    params_ = get_params(opt_state_)
    
    
    value, grads = jax.value_and_grad(loss.apply, 0, has_aux = True)(params_, None, (x_, lam_, beta_, gamma_))
    opt_state_ = opt_update(it_, grads, opt_state_)
    
    return value, opt_state_, grads, params_

num_iter = config['training']['num_iter']
batch_size = config['training']['batch_size']
lr = 1e-4 #config['training']['lr']
log_iter = config['training']['log_iter']
checkpoint_iter = 0

beta = config['training']['mani_value']
lam_value = config['training']['loglike_value']
gamma_value =  config['training']['cima_value']

lam_introduced = config['training']['loglike_int']
gamma_introduced = config['training']['cima_int']

lam_int = [lam_introduced, lam_introduced+1]
gamma_int = [gamma_introduced, gamma_introduced+1]

loss_hist = np.zeros((0,2))
loss_manifold_hist = np.zeros((0,2))
loss_dist_hist = np.zeros((0,2))
loss_cima_hist = np.zeros((0,2))
mcc_log_train_hist = np.zeros((0,4))
mcc_log_test_hist = np.zeros((0,4))
mcc_emp_train_hist = np.zeros((0,4))
mcc_emp_test_hist = np.zeros((0,4))




opt_init, opt_update, get_params = adam(step_size=lr)
opt_state = opt_init(params)


observations = obs_train
for it in np.arange(num_iter):
    x = observations[np.random.choice(N, batch_size)]
    
    # Need to warm up lambda due to stability issues
    lam = np.interp(it, lam_int, [0, lam_value])
    gam = np.interp(it, gamma_int, [0, gamma_value])
    
    loss_val, opt_state, grads, params_learned = step(it, opt_state, x, lam, beta, gam)
    
    
    loss_append = np.array([[it + 1, loss_val[0].item()]])
    loss_manifold_append = np.array([[it + 1, loss_val[1][1].item()]])
    loss_dist_append = np.array([[it + 1, loss_val[1][0].item()]])
    loss_cima_append = np.array([[it + 1, loss_val[1][2].item()]])

    loss_hist = np.concatenate([loss_hist, loss_append])
    loss_manifold_hist = np.concatenate([loss_manifold_hist, loss_manifold_append])
    loss_dist_hist = np.concatenate([loss_dist_hist, loss_dist_append])
    loss_cima_hist = np.concatenate([loss_cima_hist, loss_cima_append])
    
    if (it == lam_introduced):
        params_manifold = params_learned
        jnp.save(os.path.join(ckpt_dir, 'params_manifold.npy'),
                 hk.data_structures.to_mutable_dict(params_manifold))
        sources_in = inv.apply(params_manifold, None, obs_train)
        mani_out = fwd.apply(params_manifold, None, sources_in)
        jnp.save(os.path.join(ckpt_dir, 'sources_manifold.npy'), sources_in)
        jnp.save(os.path.join(ckpt_dir, 'manifold_manifold.npy'), mani_out)
        
    if (it == gamma_introduced):
        params_dist = params_learned
        jnp.save(os.path.join(ckpt_dir, 'params_dist.npy'),
                 hk.data_structures.to_mutable_dict(params_dist))
        sources_in = inv.apply(params_dist, None, obs_train)
        mani_out = fwd.apply(params_dist, None, sources_in)
        jnp.save(os.path.join(ckpt_dir, 'sources_dist.npy'), sources_in)
        jnp.save(os.path.join(ckpt_dir, 'manifold_dist.npy'), mani_out)

    if (it == num_iter-1):
        jnp.save(os.path.join(ckpt_dir, 'params_cima.npy'),
                 hk.data_structures.to_mutable_dict(params_learned))
        sources_in = inv.apply(params_learned, None, obs_train)
        mani_out = fwd.apply(params_learned, None, sources_in)
        jnp.save(os.path.join(ckpt_dir, 'sources_cima.npy'), sources_in)
        jnp.save(os.path.join(ckpt_dir, 'manifold_cima.npy'), mani_out)
        
    



    if (it + 1) % log_iter == 0:

        # Save loss
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
               delimiter=',', header='it,loss', comments='')
        np.savetxt(os.path.join(log_dir, 'loss_mani.csv'), loss_manifold_hist,
               delimiter=',', header='it,loss', comments='')
        np.savetxt(os.path.join(log_dir, 'loss_dist.csv'), loss_dist_hist,
               delimiter=',', header='it,loss', comments='')
        np.savetxt(os.path.join(log_dir, 'loss_cima.csv'), loss_cima_hist,
               delimiter=',', header='it,loss', comments='')

        S_train = np.vstack((x_train,y_train)).T
        S_rec = inv.apply(params_learned, None, obs_train)
        S_rec_uni = jax.scipy.stats.logistic.cdf(S_rec)
        S_rec_uni_train = S_rec_uni - 0.5
        av_corr_spearman, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_spearman_uni, _, _ = SolveHungarian(recov=S_rec_uni_train[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_pearson_uni, _, _ = SolveHungarian(recov=S_rec_uni_train[::10, :], source=S_train[::10, :],
                                                   correlation='Pearson')
        av_corr_pearson, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                   correlation='Pearson')
        mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_spearman_uni.item() av_corr_pearson.item(),av_corr_pearson_uni.item()]])
        mcc_log_train_hist = np.concatenate([mcc_log_train_hist, mcc_append])
        np.savetxt(os.path.join(log_dir, 'mcc_log_train.csv'), mcc_log_train_hist,
                       delimiter=',', header='it,spearman,pearson', comments='')

        S_test = np.vstack((x_test,y_test)).T
        S_rec = inv.apply(params_learned, None, obs_test)
        S_rec_uni = jax.scipy.stats.logistic.cdf(S_rec)
        S_rec_uni_test = S_rec_uni - 0.5
        av_corr_spearman, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_test[::10, :],
                                                    correlation='Spearman')
        av_corr_spearman_uni, _, _ = SolveHungarian(recov=S_rec_uni_test[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_pearson_uni, _, _ = SolveHungarian(recov=S_rec_uni_test[::10, :], source=S_test[::10, :],
                                                   correlation='Pearson')
        av_corr_pearson, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_test[::10, :],
                                                   correlation='Pearson')
        mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_spearman_uni.item(), av_corr_pearson.item(), av_corr_pearson_uni.item()]])
        mcc_log_test_hist = np.concatenate([mcc_log_test_hist, mcc_append])
        np.savetxt(os.path.join(log_dir, 'mcc_log_test.csv'), mcc_log_test_hist,
                       delimiter=',', header='it,spearman,pearson', comments='')




        S_train = np.vstack((x_train,y_train)).T
        S_rec = inv.apply(params_learned, None, obs_train)
        ecdfx, ecdfy = sm.distributions.ECDF(S_rec[:,0]), sm.distributions.ECDF(S_rec[:,1])
        S_rec_emp_trainx, S_rec_emp_trainy = ecdfx(S_rec[:,0])-0.5, ecdfy(S_rec[:,1])-0.5
        S_rec_emp_train = np.vstack((S_rec_emp_trainx,S_rec_emp_trainy)).T
        av_corr_spearman, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_spearman_emp, _, _ = SolveHungarian(recov=S_rec_emp_train[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_pearson_emp, _, _ = SolveHungarian(recov=S_rec_emp_train[::10, :], source=S_train[::10, :],
                                                   correlation='Pearson')
        av_corr_pearson, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                   correlation='Pearson')
        mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_spearman_emp.item(), av_corr_pearson.item(),av_corr_pearson_emp.item()]])
        mcc_emp_train_hist = np.concatenate([mcc_emp_train_hist, mcc_append])
        np.savetxt(os.path.join(log_dir, 'mcc_emp_train.csv'), mcc_emp_train_hist,
                       delimiter=',', header='it,spearman,pearson', comments='')

        S_test = np.vstack((x_test,y_test)).T
        S_rec = inv.apply(params_learned, None, obs_test)
        ecdfx, ecdfy = sm.distributions.ECDF(S_rec[:,0]), sm.distributions.ECDF(S_rec[:,1])
        S_rec_emp_testx, S_rec_emp_testy = ecdfx(S_rec[:,0])-0.5, ecdfy(S_rec[:,1])-0.5
        S_rec_emp_test = np.vstack((S_rec_emp_testx,S_rec_emp_testy)).T
        av_corr_spearman, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_test[::10, :],
                                                    correlation='Spearman')
        av_corr_spearman_emp, _, _ = SolveHungarian(recov=S_rec_emp_test[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_pearson_emp, _, _ = SolveHungarian(recov=S_rec_emp_test[::10, :], source=S_test[::10, :],
                                                   correlation='Pearson')
        av_corr_pearson, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_test[::10, :],
                                                   correlation='Pearson')
        mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_spearman_emp.item(), av_corr_pearson.item(),av_corr_pearson_emp.item()]])
        mcc_emp_test_hist = np.concatenate([mcc_emp_test_hist, mcc_append])
        np.savetxt(os.path.join(log_dir, 'mcc_emp_test.csv'), mcc_emp_test_hist,
                       delimiter=',', header='it,spearman,pearson', comments='')




        jnp.save(os.path.join(ckpt_dir, 'model_%06i.npy' % (it + 1)),
                 hk.data_structures.to_mutable_dict(params_learned))

        


        

        
            
    
