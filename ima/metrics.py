from jax import numpy as jnp

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

def cima_higher_d_fwd(s, jac_fn):
    '''
    Computes the C_IMA (Independent Mechanism Analysis contrast) for observations of any dimensionality
    Based on **sources and the mixing function**.
    
    Parameters
    ----------
    s : ndarray, shape (n_samples, n_features)
        Input collection of datapoints --- **sources**
    jac_fn : function
        Jacobian of the **mixing** function, **batched**
    Returns
    -------
    out : ndarray, shape (n_samples, )
        cima, pointwise (take the mean for the global quantity)
    '''
    J = jac_fn(s)
    logdetJ = jnp.linalg.slogdet(J)[1]
    
    # Log norms
    # axis = 1 -- norm over columns
    # axis = 2 -- norm over rows    
    sum_log_norms = jnp.sum(jnp.log(jnp.linalg.norm(J, axis=-2)), axis=-1)
    out = sum_log_norms - logdetJ
    return out

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

    def s(r):
        return jnp.sum(jnp.sum(r ** 2, axis=-1) / jnp.max(r ** 2, axis=-1) - 1, axis=-1)
    return (s(jnp.abs(P)) + s(jnp.abs(jnp.transpose(P, (0, 2, 1))))) / (2 * P.shape[1])

def jacobian_amari_distance(x, jac_r_unmix, jac_t_mix, unmixing_batched=None, sources=None):
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
    sources : true sources
    
    Returns
    -------
    d: scalar
        Average of the Amari distance between the products of the two collections of Jacobians, evaluated in x and unmixing_batched(x) respectively.
    """
    J_r_unmix = jac_r_unmix(x)
    if unmixing_batched == None:
    	J_t_mix = jac_t_mix(sources)
    elif sources == None:
    	J_t_mix = jac_t_mix(unmixing_batched(x))
    else: raise Exception('Missing input arguments (unimixing_batched/sources)')
    
    return jnp.mean(amari_distance(J_r_unmix, J_t_mix))

def observed_data_likelihood(x, jac_unmixing=None, jac_mixing=None, base_log_pdf="Uniform"):
    '''
    Computes the log-likelihood of an observation given the unmixing and a base log-pdf
    or given the mixing and a base log-pdf
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input collection of datapoints
    jac_unmixing : jacobian function for the true unmixing (**batched**)
    jac_mixing : jacobian function for the true mixing (**batched**)
    base_log_pdf : default -- Uni
    
    Returns
    -------
    d: 
    '''
    if jac_mixing == None:
    	jac = jac_unmixing(x)
    	log_det_jac = jnp.linalg.slogdet(jac)[1]
    elif jac_unmixing == None:
    	jac = jac_mixing(x)
    	log_det_jac = -jnp.linalg.slogdet(jac)[1]
    else: raise Exception('Missing input arguments (jac_unmixing/jac_mixing)')
    	
    if base_log_pdf=="Uniform":
        log_pdf = 0
    else:
        log_pdf = base_log_pdf(x)
    return log_det_jac + log_pdf

