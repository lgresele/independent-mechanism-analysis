from jax import numpy as jnp 
import numpy as np
import jax
from sympy.utilities.iterables import multiset_permutations

'''
Unless otherwise specified, whenever a function is required as input, the **batched** version has to be given.
'''

def cima(x, jac_fn):
    '''
    Computes the C_IMA (Independent Mechanism Analysis contrast) **for 2-dimensional observations**
    Based on **observations and the unmixing function** rather than on the mixing and sources.
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, **n_features=2**)
        Input collection of datapoints --- **observations**
    jac_fn : function
        Jacobian of the **unmixing** function, **batched**
    Returns
    -------
    out : ndarray, shape (n_samples, )
        cima, pointwise (take the mean for the global quantity)
    '''
    J = jac_fn(x)

    logdetJ = jnp.log(jnp.abs( J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]))
    # Log norms
    # Here taken over the rows
    sum_log_norm = jnp.sum(jnp.log(jnp.linalg.norm(J, axis=2)), axis=1)

    out = sum_log_norm - logdetJ
    return out

def cima_higher_d(x, jac_fn):
    '''
    Computes the C_IMA (Independent Mechanism Analysis contrast) for observations of any dimensionality
    Based on **observations and the unmixing function** rather than on the mixing and sources.
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input collection of datapoints --- **observations**
    jac_fn : function
        Jacobian of the **unmixing** function, **batched**
    Returns
    -------
    out : ndarray, shape (n_samples, )
        cima, pointwise (take the mean for the global quantity)
    '''
    J = jac_fn(x)
    logdetJ = jnp.linalg.slogdet(J)[1]
    
    inv_J = jnp.linalg.inv(J)
    
    # Log norms
    # axis = 1 -- norm over columns
    # axis = 2 -- norm over rows    
    sum_log_norms = jnp.sum(jnp.log(jnp.linalg.norm(inv_J, axis=1)), axis=-1)
    
    out = sum_log_norms + logdetJ
    return out

# def aDM(Jf, s):
#     '''
#     anti Darmois Metric
    
#     Input:
#     Jf: batched Jacobian (function)
#     Applied to s, which has shape (N,D), returns a Jacobian with shape (N,D,D)
    
#     s: a collection of samples of (true or reconstructed) sources, which has shape (N,D)
    
#     Output:
    
#     aDM metric: a scalar
#     '''
    
#     # Get shapes and dimensions
#     N = s.shape[0]
    
#     # Compute the Jacobian
#     Jacf = Jf(s)

#     # Compute the norm of its rows
# #     grad_norms = jnp.linalg.norm(Jacf, axis=2)
# #     # Compute the norm of its columns
#     grad_norms = jnp.linalg.norm(Jacf, axis=1)
#     log_grad_norms = jnp.log(grad_norms)

#     # Sum the norms
#     sum_log_norms = jnp.sum(log_grad_norms, axis = -1)

#     # Compute the determinants of the Jacobians
#     # N.B. Jacf needs to be of shape (..., M, M)
    
#     # Just use slogdet here!
#     jac_log_dets = jnp.linalg.slogdet(Jacf)[1]
    
#     return jnp.sum(sum_log_norms - jac_log_dets)/ N


################################################################################################################################################################################################
##
################################################################################################################################################################################################

def amari_distance(W, A):
    """
    Computes the Amari distance between the products of two collections of matrices W and A.
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
        The Amari distances between the average of absolute values of the products of W and A.
    """
    
    P = jnp.matmul(W, A)
#     P = jnp.mean(jnp.abs(jnp.matmul(W, A)), axis=0)

    def s(r):
        return jnp.sum(jnp.sum(r ** 2, axis=-1) / jnp.max(r ** 2, axis=-1) - 1, axis=-1)
    return (s(jnp.abs(P)) + s(jnp.abs(jnp.transpose(P, (0, 2, 1))))) / (2 * P.shape[1])
#     return (s(jnp.abs(P)) + s(jnp.abs(P.T))) / (2 * P.shape[1])

def jacobian_amari_distance(x, jac_r_unmix, jac_t_mix, unmixing_batched):
    """
    Computes the average of the Amari distance between the products of two collections of **batched** Jacobians evaluated at points x.
    It cancels when the composition of the functions whose Jacobians we multiply is a scalar function times a permutation.
    
    Based on the implementation of Amari distance in:
    https://github.com/pierreablin/picard/blob/master/picard/_tools.py
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input collection of datapoints
    jac_r_unmix : jacobian function for the reconstructed unmixing (**batched**)
    jac_t_mix : jacobian function for the true unmixing (**batched**)
    unmixing_batched : true unmixing (**batched**)
    
    Returns
    -------
    d: scalar
        Average of the Amari distance between the products of the two collections of Jacobians, evaluated in x and unmixing_batched(x) respectively.
    """
#     J_r_unmix = jax.vmap(jac_r_unmix)(x)
#     J_t_mix = jax.vmap(jac_t_mix)(unmixing_batched(x))
    J_r_unmix = jac_r_unmix(x)
    J_t_mix = jac_t_mix(unmixing_batched(x))
    
    return jnp.mean(amari_distance(J_r_unmix, J_t_mix))


################################################################################################################################################################################################
##
################################################################################################################################################################################################


def observed_data_likelihood(x, jac_unmixing, base_log_pdf="Uniform"):
    '''
    Computes the log-likelihood of an observation given the unmixing and a base log-pdf
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input collection of datapoints
    jac_r_unmix : jacobian function for the reconstructed unmixing (**batched**)
    base_log_pdf : default -- Uni
    
    Returns
    -------
    d: 
    '''
#     Jac = jax.jacfwd(unmixing)
    jac = jac_unmixing(x)
    log_det_jac = jnp.linalg.slogdet(jac)[1]
    if base_log_pdf=="Uniform":
        log_pdf = 0
    else:
        log_pdf = base_log_pdf(x)
    return log_det_jac + log_pdf

# ################################################################################################################################################################################################
# ##
# ################################################################################################################################################################################################


# # Below the distance based on bruteforce search on the permutation matrices -- not using it for now as too much variability when bad mixing.

# def amari_inspired_distance(W, A, perm):
#     """
#     Computes the Amari distance between the products of two collections of matrices W and A.
#     It cancels when the average of the absolute value of WA is a permutation and scale matrix.
    
#     Based on the implementation of Amari distance in:
#     https://github.com/pierreablin/picard/blob/master/picard/_tools.py
    
#     Parameters
#     ----------
#     W : ndarray, shape (n_samples, n_features, n_features)
#         Input collection of matrices
#     A : ndarray, shape (n_samples, n_features, n_features)
#         Input collection of matrices
#     perm : ndarray, shape (n_features, n_features)
#         A permutation matrix
#     Returns
#     -------
#     d : ndarray, shape (n_samples, )
#         The Amari inspired distance between the average of absolute values of the products of W and A (and P).
#         This measures whether the product of W, A and P is diagonal.
#     """
    
#     P = jnp.matmul(W, A)
#     P = jnp.matmul(P, perm)
# #     P = jnp.mean(jnp.abs(jnp.matmul(W, A)), axis=0)
    
#     p = jnp.diagonal(P, axis1=1, axis2=2) # Here only the diagonal elements
# #     print(p.shape)
#     def s(r, p):
#         return jnp.sum(jnp.sum(r ** 2, axis=-1) / p ** 2 - 1, axis=-1)
#     return (s(jnp.abs(P), p) + s(jnp.abs(jnp.transpose(P, (0, 2, 1))), p)) / (2 * P.shape[1])
# #     return (s(jnp.abs(P)) + s(jnp.abs(P.T))) / (2 * P.shape[1])

# def jacobian_amari_inspired_distance(x, jac_r_unmix, jac_t_mix, unmixing_batched):
#     """
#     Computes an Amari-inspired distance between the products of two collections of **batched** Jacobians evaluated at points x.
#     It cancels when the composition of the functions whose Jacobians we multiply is a scalar function times a permutation.
#     It requires a bruteforce permutation through all permutation matrices.
    
#     Based on the implementation of Amari distance in:
#     https://github.com/pierreablin/picard/blob/master/picard/_tools.py
    
#     Parameters
#     ----------
#     x : ndarray, shape (n_samples, n_features)
#         Input collection of datapoints
#     jac_r_unmix : jacobian function for the reconstructed unmixing (**batched**)
#     jac_t_mix : jacobian function for the true unmixing (**batched**)
#     unmixing_batched : true unmixing (**batched**)
    
#     Returns
#     -------
#     d: scalar
#         Average of the Amari distance between the products of the two collections of Jacobians, evaluated in x and unmixing_batched(x) respectively.
#     """
# #     J_r_unmix = jax.vmap(jac_r_unmix)(x)
# #     J_t_mix = jax.vmap(jac_t_mix)(unmixing_batched(x))
#     J_r_unmix = jac_r_unmix(x)
#     J_t_mix = jac_t_mix(unmixing_batched(x))
    
#     minimum = 10**20
    
#     # find all of the permutation matrices
#     a = jnp.arange(x.shape[1])
#     for p in multiset_permutations(a):
#         perm = np.zeros((x.shape[1], x.shape[1]))
#         perm[a, p]=1.0
#         ad1 = jnp.mean(amari_inspired_distance(J_r_unmix, J_t_mix, perm))    
#         if ad1<minimum:
#             minimum = ad1
#     return minimum
