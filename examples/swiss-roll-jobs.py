import numpy as np
import jax
from jax import numpy as jnp
import distrax
import haiku as hk
from ima.upsamling import Pad
from jax.experimental.optimizers import adam
from tqdm import tqdm
from matplotlib import pyplot as plt
from jax.config import config
from jax.lib import xla_bridge
from ima.plotting import cart2pol 
import argparse
import os
from ima.utils import get_config
from ima.solve_hungarian import SolveHungarian
import statsmodels.api as sm

config.update("jax_enable_x64",True)

# Parse input arguments
parser = argparse.ArgumentParser(description='Train for swiss roll dataset')

parser.add_argument('--config', type=str, default='./config.yaml',
                    help='Path config file specifying model '
                         'architecture and training procedure')

args = parser.parse_args()


# Load config
configs = get_config(args.config)

# Set seed
key = jax.random.PRNGKey(configs['data']['jax_seed'])
np.random.seed(configs['data']['np_seed'])

# Make directories
root = configs['training']['save_root']
ckpt_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
data_dir = os.path.join(root, 'data')
# Create dirs if not existent
for dir in [ckpt_dir, plot_dir, log_dir, data_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

key, subkey = jax.random.split(key)
N = configs['data']['N']

base_samples = jax.random.uniform(subkey, shape=(2*N, 2), minval=0, maxval=1)

x_train = base_samples[:N,0]
x_test = base_samples[N:,0]
y_train = base_samples[:N,1]
y_test = base_samples[N:,1]


pi = np.pi
hpi = (configs['data']['hpi'])*pi
obs = jnp.stack((hpi*x_train*jnp.cos(hpi*(x_train)), y_train, hpi*x_train*jnp.sin(hpi*(x_train))), axis=1)
obs = (1.8)*(obs - jnp.min(obs,axis=0))/(jnp.max(obs,axis=0) - jnp.min(obs,axis=0)) - 0.9



# Save data
jnp.save(os.path.join(data_dir, 'sourcesx_train.npy'), x_train)
jnp.save(os.path.join(data_dir, 'sourcesx_test.npy'), x_test)

jnp.save(os.path.join(data_dir, 'sourcesy_train.npy'), y_train)
jnp.save(os.path.join(data_dir, 'sourcesy_test.npy'), y_test)

jnp.save(os.path.join(data_dir, 'mani_train.npy'), obs)

d = 2
D = obs.shape[1]
N = obs.shape[0]

key = jax.random.PRNGKey(configs['training']['jax_seed'])
np.random.seed(configs['training']['np_seed'])
# Define Neural Spline with permutation each layer flow with Distrax
#def mk_flow(K = 8-16, nl = 2, hu = 256):
def mk_flow(K = 16, nl = 2, hu = 256,key=key):
    
    num_bins = 8
    range_val = 1
    bs = 'identity'
    
    layers_lowdim =[]
    for i in range(K):
        key, subkey = jax.random.split(key)
        
        def bij_fn_1(params):
            new_d = params.shape[-1]//(3*num_bins+1)
            if (len(params.shape)) == 1:
                params_reshaped = params.reshape((new_d,3*num_bins+1))
            else:
                params_reshaped = params.reshape((params.shape[0],new_d,3*num_bins+1))
            bij = distrax.RationalQuadraticSpline(params=params_reshaped, range_min=-range_val, range_max=range_val,boundary_slopes=bs)
            return distrax.Block(bij, 1)
        
        if i%2:
            shape_dim_params_d = (d+1)//2
            mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear((shape_dim_params_d)*(3*num_bins+1), w_init=jnp.zeros, b_init=jnp.zeros)])
            layers_lowdim.append(distrax.Lambda(lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x: jnp.ones((x.shape)),event_ndims_in=1))
            layers_lowdim.append(distrax.SplitCoupling(d // 2, 1, mlp, bij_fn_1, swap=False))
        else:
            shape_dim_params_d = d//2
            mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear((shape_dim_params_d)*(3*num_bins+1), w_init=jnp.zeros, b_init=jnp.zeros)])
            layers_lowdim.append(distrax.Lambda(lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x: jnp.ones((x.shape)),event_ndims_in=1))
            layers_lowdim.append(distrax.SplitCoupling(d // 2, 1, mlp, bij_fn_1, swap=True))

        
    flow_lowdim = distrax.Chain(layers_lowdim)
    pad = Pad((0, D - d))
    layers = []
    W = []
    b = []
    for i in range(K):
        key, subkey = jax.random.split(key)
        #shape_dim_params_D = (D+(i%2))//2
        #mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
        #                     hk.Linear((shape_dim_params_D)*(3*num_bins+1), w_init=jnp.zeros, b_init=jnp.zeros)])
        
        #W.append(hk.get_parameter("W"+ str(i), [D, D], init = jnp.zeros))
        #b.append(hk.get_parameter("b"+ str(i), [D], init = jnp.zeros))
        #W[i] = W[i] - jnp.diag(jnp.diag(W[i])) + jnp.diag(jnp.exp(jnp.diag(W[i])))
        
        
        
        def bij_fn_2(params):
            new_D = params.shape[-1]//(3*num_bins+1)
            if (len(params.shape)) == 1:
                params_reshaped = params.reshape((new_D,3*num_bins+1))
            else:
                params_reshaped = params.reshape((params.shape[0],new_D,3*num_bins+1))
            bij = distrax.RationalQuadraticSpline(params=params_reshaped, range_min=-range_val,range_max=range_val,boundary_slopes=bs)
            return distrax.Block(bij, 1)
                                 
        if i%2:
            shape_dim_params_D = (D+1)//2
            mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear((shape_dim_params_D)*(3*num_bins+1), w_init=jnp.zeros, b_init=jnp.zeros)])
            #layers.append(distrax.Lambda(lambda x: x[:,jax.random.permutation(key, jnp.arange(D))]
            layers.append(distrax.Lambda(lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x: jnp.ones((x.shape)),event_ndims_in=1))
            layers.append(distrax.SplitCoupling(D // 2, 1, mlp, bij_fn_2, swap=False))
        else:
            shape_dim_params_D = D//2
            mlp = hk.Sequential([hk.nets.MLP(nl * (hu,), activate_final=True),
                             hk.Linear((shape_dim_params_D)*(3*num_bins+1), w_init=jnp.zeros, b_init=jnp.zeros)])
            layers.append(distrax.Lambda(lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x,key=key: jax.random.permutation(key,x.T).T if x.size > 1 else x, 
                                         lambda x: jnp.ones((x.shape)),event_ndims_in=1))
            layers.append(distrax.SplitCoupling(D // 2, 1, mlp, bij_fn_2, swap=True))

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

import cloudpickle, pickle

with open(os.path.join(data_dir, 'fwd.pickle'), 'wb') as handle:
	cloudpickle.dump(fwd, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
with open(os.path.join(data_dir, 'inv.pickle'), 'wb') as handle:
	cloudpickle.dump(inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Loss function

def loss_(args):


    x, lam, beta, gamma = args
    flow_lowdim, pad, flow = mk_flow()
    
    fwd = lambda y: flow.forward(pad.forward(flow_lowdim.forward(y)))
    inv = lambda y: flow_lowdim.inverse(pad.inverse(flow.inverse(y)))
    fn_inv = lambda y: pad.inverse(flow.inverse(y))
    
    #base_dist = distrax.Independent(distrax.Normal(loc=jnp.zeros(d), scale=jnp.ones(d)),
    #                                reinterpreted_batch_ndims=1)
    #base_dist = distrax.Independent(distrax.Logistic(loc=jnp.zeros(d), scale=jnp.ones(d)),
    #                                reinterpreted_batch_ndims=1)
    distr = distrax.Uniform(-1, 1)
    distr._batch_shape = (d,)
    base_dist = distrax.Independent(distr,
                                    reinterpreted_batch_ndims=1)
    
    jac_fn = jax.vmap(jax.jacfwd(fwd))
    
    #print(x)
    #print(a._split(x))
    #print(b._split(x))
    
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
    
    return jnp.mean(-lam * (base_dist.log_prob(z) - log_det ) + beta * diff + gamma*cima ), (jnp.mean(-lam * (base_dist.log_prob(z) - log_det)), beta * diff, jnp.mean(gamma*cima),jnp.mean((base_dist.log_prob(z) - log_det)), diff, jnp.mean(cima))
    

arr = jnp.array(np.random.randn(5, D))
arr = arr/np.max(arr)
key, subkey = jax.random.split(key)
loss = hk.transform(loss_)
params = loss.init(subkey, (arr, 1., 1., 1.))

with open(os.path.join(data_dir, 'loss.pickle'), 'wb') as handle:
	cloudpickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)



@jax.jit
def step(it_, opt_state_, x_, lam_, beta_, gamma_):
    params_ = get_params(opt_state_)
    
    
    value, grads = jax.value_and_grad(loss.apply, 0, has_aux = True)(params_, None, (x_, lam_, beta_, gamma_))
    opt_state_ = opt_update(it_, grads, opt_state_)
    
    return value, opt_state_, grads, params_

num_iter = configs['training']['num_iter']
batch_size = configs['training']['batch_size']
lr = 1e-4 #configs['training']['lr']
log_iter = configs['training']['log_iter']
checkpoint_iter = 0

beta = configs['training']['mani_value']
lam_value_start = configs['training']['loglike_value_start']
gamma_value_start =  configs['training']['cima_value_start']

lam_value_stop = configs['training']['loglike_value_stop']
gamma_value_stop =  configs['training']['cima_value_stop']

lam_introduced = configs['training']['loglike_int']
gamma_introduced = configs['training']['cima_int']

lam_constant = configs['training']['loglike_con']
gamma_constant = configs['training']['cima_con']

lam_int = [lam_introduced, lam_introduced+1, lam_constant]
gamma_int = [gamma_introduced, gamma_introduced+1,gamma_constant]

loss_hist = np.zeros((0,2))
loss_manifold_hist = np.zeros((0,2))
loss_dist_hist = np.zeros((0,2))
loss_cima_hist = np.zeros((0,2))
loglike_hist = np.zeros((0,2))
mani_hist = np.zeros((0,2))
cima_hist = np.zeros((0,2))
mcc_hist = np.zeros((0,3))




opt_init, opt_update, get_params = adam(step_size=lr)
opt_state = opt_init(params)


observations = obs
for it in np.arange(num_iter):
    x = observations[np.random.choice(N, batch_size)]
    
    # Need to warm up lambda due to stability issues
    lam = np.interp(it, lam_int, [0, lam_value_start, lam_value_stop])
    gam = np.interp(it, gamma_int, [0, gamma_value_start, gamma_value_stop])
    
    loss_val, opt_state, grads, params_learned = step(it, opt_state, x, lam, beta, gam)
    
    loss_append = np.array([[it + 1, loss_val[0].item()]])
    loss_manifold_append = np.array([[it + 1, loss_val[1][1].item()]])
    loss_dist_append = np.array([[it + 1, loss_val[1][0].item()]])
    loss_cima_append = np.array([[it + 1, loss_val[1][2].item()]])
	
    loglike_append = np.array([[it + 1, loss_val[1][3].item()]])
    mani_append = np.array([[it + 1, loss_val[1][4].item()]])
    cima_append = np.array([[it + 1, loss_val[1][5].item()]])

    loss_hist = np.concatenate([loss_hist, loss_append])
    loss_manifold_hist = np.concatenate([loss_manifold_hist, loss_manifold_append])
    loss_dist_hist = np.concatenate([loss_dist_hist, loss_dist_append])
    loss_cima_hist = np.concatenate([loss_cima_hist, loss_cima_append])
	
    loglike_hist = np.concatenate([loglike_hist, loglike_append])
    mani_hist = np.concatenate([mani_hist, mani_append])
    cima_hist = np.concatenate([cima_hist, cima_append])
    
    if (it == lam_introduced):
        params_manifold = params_learned
        jnp.save(os.path.join(ckpt_dir, 'params_manifold.npy'),
                 hk.data_structures.to_mutable_dict(params_manifold))
        sources_in = inv.apply(params_manifold, None, observations)
        mani_out = fwd.apply(params_manifold, None, sources_in)
        jnp.save(os.path.join(ckpt_dir, 'sources_manifold.npy'), sources_in)
        jnp.save(os.path.join(ckpt_dir, 'manifold_manifold.npy'), mani_out)
        
    if (it == gamma_introduced):
        params_dist = params_learned
        jnp.save(os.path.join(ckpt_dir, 'params_dist.npy'),
                 hk.data_structures.to_mutable_dict(params_dist))
        sources_in = inv.apply(params_dist, None, observations)
        mani_out = fwd.apply(params_dist, None, sources_in)
        jnp.save(os.path.join(ckpt_dir, 'sources_dist.npy'), sources_in)
        jnp.save(os.path.join(ckpt_dir, 'manifold_dist.npy'), mani_out)

    if (it == num_iter-1):
        jnp.save(os.path.join(ckpt_dir, 'params_cima.npy'),
                 hk.data_structures.to_mutable_dict(params_learned))
        sources_in = inv.apply(params_learned, None, observations)
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
		
        np.savetxt(os.path.join(log_dir, 'loglike.csv'), loglike_hist,
               delimiter=',', header='it,val', comments='')
        np.savetxt(os.path.join(log_dir, 'mani.csv'), mani_hist,
               delimiter=',', header='it,val', comments='')
        np.savetxt(os.path.join(log_dir, 'cima.csv'), cima_hist,
               delimiter=',', header='it,val', comments='')

        S_train = np.vstack((x_train,y_train)).T
        S_rec = inv.apply(params_learned, None, observations)

        av_corr_spearman, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                    correlation='Spearman')
        av_corr_pearson, _, _ = SolveHungarian(recov=S_rec[::10, :], source=S_train[::10, :],
                                                    correlation='Pearson')

        mcc_append = np.array([[it + 1, av_corr_spearman.item(), av_corr_pearson.item()]])
        mcc_hist = np.concatenate([mcc_hist, mcc_append])
        np.savetxt(os.path.join(log_dir, 'mcc_log.csv'), mcc_hist,
                       delimiter=',', header='it,spearman,pearson', comments='')

        jnp.save(os.path.join(ckpt_dir, 'model_%06i.npy' % (it + 1)),
                 hk.data_structures.to_mutable_dict(params_learned))

        


        

        
            
    
