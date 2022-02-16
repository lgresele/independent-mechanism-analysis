'''
Implementation of various mixing/unmixing functions.
'''

from jax.scipy import special
from jax import numpy as jnp

'''
Mixing functions inspired by "Nonlinear independent component analysis: Existence and uniqueness results",

https://www.cs.helsinki.fi/u/ahyvarin/papers/NN99.pdf

'''

def f_1(s):
    '''
    "Moderately nonlinear mixing"
    '''
    f0 = jnp.tanh(4*s[0] - 2) + s[0] + s[1]/2
    f1 = jnp.tanh(4*s[1] - 2) + s[1] + s[0]/2
    return jnp.array([f0, f1])

def f_2(s):
    '''
    "Rather nonlinear mixing"
    '''
    f0 = jnp.tanh(s[1])/2 + s[0] + s[0]**2/2
    f1 = s[0]**3 - s[0] + jnp.tanh(s[1])
    return jnp.array([f0, f1])

def f_3(s):
    '''
    "Non-bijective nonlinear mixing"
    '''
    f0 = s[1]**3 + s[0]
    f1 = jnp.tanh(s[1]) + s[0]**3
    return jnp.array([f0, f1])

'''
Post-nonlinear mixing and unmixing
'''

def post_nonlinear_model(A, nonlinearity='cube'):
    '''
    Returns a post-nonlinear mixing and unmixing,
    based on a nonlinearity (default: cube) and a mixing matrix.
    '''
    def mixing(x):
        y = A @ x
        y = y**3
        return y
    
    A_inv = jnp.linalg.inv(A) 
        
    def unmixing(x):
        y = jnp.cbrt(x)
        y = A_inv @ y
        return y

    return mixing, unmixing

'''
Ground truth forward function, upon starting with Uniform random variables
'''

def f_g_unl(A):
    '''
    Returns a function turning a Uniform random variable into Normal which is then Linearly mixed (UNL), and its inverse function.
    Note that this corresponds to the same operations in a PNL model, but in reversed order.
    '''
    
    def f(x):
        y = special.erfinv(x*2.0-1.0)
        y = A @ y
        return y
    
    A_inv = jnp.linalg.inv(A) 
    
    def f_inv(x):
        y = A_inv @ x
        y = 0.5*(special.erf(y)+1)
        return y
    
    return f, f_inv


def f_lin(A):
    '''
    Returns a function performing a linear mixing, and its inverse
    '''
    
    def f(x):
        return A @ x
    
    A_inv = jnp.linalg.inv(A) 
    
    def f_inv(x):
        return A_inv @ x
    
    return f, f_inv


def darmois_linear_gaussian_2d(A):
    '''
    Returns the closed form Darmois construction (and its inverse) for the case of linearly mixed Gaussian sources in 2d
    '''
    sigma_0 = jnp.sqrt(A[0,0]**2 + A[0,1]**2) 
    sigma_1 = jnp.sqrt(A[1,0]**2 + A[1,1]**2) 
    rho_01 = (A[0,0]*A[1,0] + A[0,1]*A[1,1])/(sigma_0*sigma_1)
    c_1_given_0 = rho_01*sigma_1/sigma_0
    
    def darmois(x):
        y_0 = 0.5*(1.0 + special.erf(x[0]/(sigma_0*jnp.sqrt(2.0))))
        y_1 = 0.5*(1.0 + special.erf( (x[1] - c_1_given_0* x[0]) /jnp.sqrt( 2 * ( 1.0 - rho_01**2) * sigma_1**2 )))
        return jnp.array([y_0, y_1])

    def inv_darmois(y):
        s_0 = sigma_0*jnp.sqrt(2)*special.erfinv(2.0*y[0]-1.0)
        s_1 = jnp.sqrt( 2 * ( 1.0 - rho_01**2) * sigma_1**2 ) * special.erfinv(2.0*y[1]-1.0) + c_1_given_0 * s_0
        return jnp.array([s_0, s_1])    
    
    return darmois, inv_darmois


def build_conformal_map(nonlinearity):
    '''
    Build conformal map in 2d starting from a given nonlinearity,
    by turning the sources into complex numbers, applying the nonlinearity and
    separating real and imaginary part in the observations.
    '''
    
    def conformal_map(x):
#         x-= 0.5
        z = x[0] + x[1]*1j
        y = nonlinearity(z)
        re = jnp.real(y)
        imag = jnp.imag(y)
        return jnp.array([re, imag])
    
    def conformal_map_gridplot(x_0, x_1):
#         x_0-= 0.5
#         x_1-= 0.5
        z = x_0 + x_1*1j
        y = nonlinearity(z)
        re = jnp.real(y)
        imag = jnp.imag(y)
        return re, imag
    
    return conformal_map, conformal_map_gridplot


def build_moebius_transform(alpha, A, a, b, epsilon=2):
    '''
    Implements Möbius transformations for D>=2, based on:
    https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)
    
    alpha: a scalar
    A: an orthogonal matrix
    a, b: vectors in \RR^D (dimension of the data)
    '''
    def mixing_moebius_transform(x):
        if epsilon==2:
            frac = jnp.sum((x-a)**2) #is this correct?
            frac = frac**(-1)
        else:
            diff = jnp.abs(x-a)
            
            frac = 1.0
        return b + frac * alpha * A @ (x - a)
    
    B = jnp.linalg.inv(A)
    
    def unmixing_moebius_transform(y):
        numer = 1/alpha * (y - b)
        if epsilon==2:
            denom = jnp.sum((numer)**2)
        else:
            denom = 1.0
        return a + 1.0/denom * B @ numer
    
    return mixing_moebius_transform, unmixing_moebius_transform



'''
Building measure preserving automorphisms based on appendix D.1 in the paper:
https://arxiv.org/abs/1907.04809
'''

def build_automorphism(A):
    '''
    Takes an orthogonal matrix A, returns a measure preserving automorphism
    On the unit square (cube?) and its inverse
    '''
    def measure_preserving(z):
        # apply inverse cdf transform
        z_gauss = special.erfinv(2*z - 1.0)
        # apply rotation
        z_gauss = A @ z_gauss
        # apply cdf transform
        z_modified = 0.5*(1.0 + special.erf(z_gauss))
        return z_modified
    
    A_inv = jnp.linalg.inv(A) 
    
    def measure_preserving_inv(z):
        # apply cdf transform
        z_gauss = special.erfinv(2*z - 1.0)
        # apply (inverse) rotation
        z_gauss = A_inv @ z_gauss
        # apply inverse cdf transform
        z_modified = 0.5*(1.0 + special.erf(z_gauss))
        return z_modified
    
    return measure_preserving, measure_preserving_inv

'''
Build polar to cartesian transformation and inverse transformation
Only in two dimensions.
'''

def build_radial_map(add, rescale):
    def pol2cart_mixing(S):
        '''
        From cartesian to polar coordinates
        '''
        S = S @ rescale
        S += add
        rho,phi = S[0], S[1]
        x = rho * jnp.cos(phi)
        y = rho * jnp.sin(phi)
        return jnp.asarray([x, y])
    
    rescale_inv = jnp.linalg.inv(rescale)
    
    def cart2pol_unmixing(X):
        '''
        From polar to cartesian coordinates
        '''
        x,y = X[0], X[1]
        rho = jnp.sqrt(x**2 + y**2)
        phi = jnp.arctan2(y, x)
        S = jnp.asarray([rho, phi])
        S -= add
        S = S @ rescale_inv
        return S
    
    return pol2cart_mixing, cart2pol_unmixing