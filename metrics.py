from jax import numpy as jnp 
import jax
    
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

    # Compute the norm of its rows
#     grad_norms = jnp.linalg.norm(Jacf, axis=2)
#     # Compute the norm of its columns
    grad_norms = jnp.linalg.norm(Jacf, axis=1)
    log_grad_norms = jnp.log(grad_norms)

    # Sum the norms
    sum_log_norms = jnp.sum(log_grad_norms, axis = -1)

    # Compute the determinants of the Jacobians
    # N.B. Jacf needs to be of shape (..., M, M)
    
    # Just use slogdet here!
    jac_log_dets = jnp.linalg.slogdet(Jacf)[1]
    
    return jnp.sum(sum_log_norms - jac_log_dets)/ N


################################################################################################################################################################################################
##
################################################################################################################################################################################################

def amari_distance_average(W, A):
    """
    Computes the Amari distance between the average of the absolute value of the products of two collections of matrices W and A.
    It cancels when the average of the absolute value of WA is a permutation and scale matrix.
    
    Based on the implementation of Amari distance in:
    https://github.com/pierreablin/picard/blob/master/picard/_tools.py
    
    Parameters
    ----------
    W : ndarray, shape (n_samples, n_features, n_features)
        Input collection of matrices
    A : ndarray, shape (n_samples, n_features, n_features)
        Input collection of matrices
    Returns
    -------
    d : ndarray, shape (n_samples, )
        The Amari distances between the average of the products of W and A.
    """
    
#     P = jnp.matmul(W, A)
    P = jnp.mean(jnp.abs(jnp.matmul(W, A)), axis=0)

    def s(r):
        return jnp.sum(jnp.sum(r ** 2, axis=-1) / jnp.max(r ** 2, axis=-1) - 1, axis=-1)
    return (s(jnp.abs(P)) + s(jnp.abs(P.T))) / (2 * P.shape[1])
#     return (s(jnp.abs(P)) + s(jnp.abs(jnp.transpose(P, (0, 2, 1))))) / (2 * P.shape[1])

def jacobian_amari_distance(x, jac_r_unmix, jac_t_mix, unmixing_batched):
    """
    Computes the Amari distance between the average of the products of two collections of matrices W and A.
    It cancels when the average of WA is a permutation and scale matrix.
    
    Based on the implementation of Amari distance in:
    https://github.com/pierreablin/picard/blob/master/picard/_tools.py
    
    Parameters
    ----------
    x : ndarray, shape (n_features, n_features)
        Input collection of datapoints
    jac_r_unmix : jacobian function for the reconstructed unmixing (not batched)
    jac_t_mix : jacobian function for the true unmixing (not batched)
    unmixing_batched : true unmixing (batched)
    
    Returns
    -------
    d: scalar
        Amari distance between the average of the products of the two collections of Jacobians, evaluated in x and unmixing_batched(x) respectively.
    """
    J_r_unmix = jax.vmap(jac_r_unmix)(x)
    J_t_mix = jax.vmap(jac_t_mix)(unmixing_batched(x))
    
    return amari_distance_average(J_r_unmix, J_t_mix)

################################################################################################################################################################################################
##
################################################################################################################################################################################################

def observed_data_likelihood(x, unmixing, base_log_pdf="Uniform"):
    '''
    Computes the log-likelihood of an observation given the unmixing and a base log-pdf
    '''
    Jac = jax.jacfwd(unmixing)
    jac = Jac(x)
    log_det_jac = jnp.linalg.slogdet(jac)[1]
    if base_log_pdf=="Uniform":
        log_pdf = 0
    else:
        log_pdf = base_log_pdf(x)
    return log_det_jac + log_pdf