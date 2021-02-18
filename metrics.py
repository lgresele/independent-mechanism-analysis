from jax import numpy as np 
    
def aDM(Jf, s):
    '''
    anti Darmois Metric
    
    Input:
    Jf: batched Jacobian (function)
    Applied to s, which has shape (N,D), returns a Jacobian with shape (N,D,D)
    
    s: a collection of samples of (true or reconstructed) sources, which has shape (N,D)
    
    Output:
    
    aDM metric: a scalar
    '''
    
    # Get shapes and dimensions
    N = s.shape[0]
    
    # Compute the Jacobian
    Jacf = Jf(s)

    # Compute the norm of its columns
    grad_norms = np.linalg.norm(Jacf, axis=1)
    log_grad_norms = np.log(grad_norms)

    # Sum the norms
    sum_log_norms = np.sum(log_grad_norms, axis = 1)

    # Compute the determinants of the Jacobians
    # N.B. Jacf needs to be of shape (..., M, M)
    
    # Just use slogdet here!
    jac_log_dets = np.linalg.slogdet(Jacf)[1]
    
    return np.sum(sum_log_norms - jac_log_dets)/ N