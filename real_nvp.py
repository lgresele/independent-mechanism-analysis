'''
A Jax implementation of RealNVP
Based on:
https://github.com/glouppe/flowing-with-jax
'''

from jax import numpy as np
from jax.experimental.stax import serial, Dense, Relu

def real_nvp_and_inv(D):
    
    d = D // 2
    
    def realnvp(x, s, s_params, t, t_params):
        x1, x2 = x[:d], x[d:]
        sx1 = s(s_params, x1)
        tx1 = t(t_params, x1)
        z1, z2 = x1, x2 * np.exp(sx1) + tx1
        z = np.concatenate([z1, z2], axis=-1)
        return z

    def realnvp_inv(z, s, s_params, t, t_params): 
        z1, z2 = z[:d], z[d:]
        x1 = z1
        sx1 = s(s_params, x1)
        tx1 = t(t_params, x1)  
        x2 = (z2 - tx1) * np.exp(-sx1)
        x = np.concatenate([x1, x2], axis=-1)
        return x
    
    return realnvp, realnvp_inv

'''
Make a sequential transformation from a map of a bunch of them
'''

def nf_init(n_steps, rng, d):
    nf_nn, nf_params = [], []
    
    for i in range(n_steps):
        s_init, s = serial(Dense(1024), Relu, Dense(1024), Relu, Dense(1))
        _, s_params = s_init(rng, (d,))
        
        t_init, t = serial(Dense(1024), Relu, Dense(1024), Relu, Dense(1))
        _, t_params = t_init(rng, (d,))
    
        nf_nn.append((s, t))
        nf_params.append((s_params, t_params))
        
    return nf_nn, nf_params 

def nf_forward(x, nf_nn, nf_params, forward, return_jacobians=False, jacobian_function=None):
    z = x
    jacobians = []
    
    for i, ((s, t), (s_params, t_params)) in enumerate(zip(nf_nn, nf_params)):
#         Uncomment this if you _do_ want reverse among the layers!
#         Plus, only works in _batch_ mode!
        if i % 2 == 0:
#             z = z[:, ::-1]
#             Maybe written as below works both in Batch mode and not?
            z = np.flip(z, -1)
            
        if return_jacobians:
            J = jacobian_function(z, s, s_params, t, t_params)
            jacobians.append(J)
        
        z = forward(z, s, s_params, t, t_params)
        
    if return_jacobians:
        return z, jacobians
    else:
        return z

def nf_backward(z, nf_nn, nf_params, backward):
    x = z
    
    nf_nn = nf_nn[::-1]
    nf_params = nf_params[::-1]
    
    for i, ((s, t), (s_params, t_params)) in enumerate(zip(nf_nn, nf_params)):
        x = backward(x, s, s_params, t, t_params)
        
        # Uncomment this if you _do_ want reverse among the layers!
#         Plus, only works in _batch_ mode!
        if (len(nf_params) - i - 1) % 2 == 0:
#             x = x[:, ::-1]
#             Maybe written as below works both in Batch mode and not?
            x = np.flip(x, -1)
            
    return x
