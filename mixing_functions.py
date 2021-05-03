'''
Implementation of various mixing/unmixing functions.
'''

from jax import numpy as np
from jax.scipy import special

'''
Mixing functions inspired by "Nonlinear independent component analysis: Existence and uniqueness results",

https://www.cs.helsinki.fi/u/ahyvarin/papers/NN99.pdf

'''

def f_1(s):
    '''
    "Moderately nonlinear mixing"
    '''
    f0 = np.tanh(4*s[0] - 2) + s[0] + s[1]/2
    f1 = np.tanh(4*s[1] - 2) + s[1] + s[0]/2
    return np.array([f0, f1])

def f_2(s):
    '''
    "Rather nonlinear mixing"
    '''
    f0 = np.tanh(s[1])/2 + s[0] + s[0]**2/2
    f1 = s[0]**3 - s[0] + np.tanh(s[1])
    return np.array([f0, f1])

def f_3(s):
    '''
    "Non-bijective nonlinear mixing"
    '''
    f0 = s[1]**3 + s[0]
    f1 = np.tanh(s[1]) + s[0]**3
    return np.array([f0, f1])

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
    
    A_inv = np.linalg.inv(A) 
        
    def unmixing(x):
        y = np.cbrt(x)
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
    
    A_inv = np.linalg.inv(A) 
    
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
    
    A_inv = np.linalg.inv(A) 
    
    def f_inv(x):
        return A_inv @ x
    
    return f, f_inv

'''
Closed form Darmois construction for the linear Gaussian case
'''

def darmois_linear_gaussian_2d(A):
    '''
    Returns the Darmois construction (and its inverse) for 2d Gaussian sources
    '''
    sigma_0 = np.sqrt(A[0,0]**2 + A[0,1]**2) 
    sigma_1 = np.sqrt(A[1,0]**2 + A[1,1]**2) 
    rho_01 = (A[0,0]*A[1,0] + A[0,1]*A[1,1])/(sigma_0*sigma_1)
    c_1_given_0 = rho_01*sigma_1/sigma_0
    
    def darmois(x):
        y_0 = 0.5*(1.0 + special.erf(x[0]/(sigma_0*np.sqrt(2.0))))
        y_1 = 0.5*(1.0 + special.erf( (x[1] - c_1_given_0* x[0]) /np.sqrt( 2 * ( 1.0 - rho_01**2) * sigma_1**2 )))
        return np.array([y_0, y_1])

    def inv_darmois(y):
        s_0 = sigma_0*np.sqrt(2)*special.erfinv(2.0*y[0]-1.0)
        s_1 = np.sqrt( 2 * ( 1.0 - rho_01**2) * sigma_1**2 ) * special.erfinv(2.0*y[1]-1.0) + c_1_given_0 * s_0
        return np.array([s_0, s_1])    
    
    return darmois, inv_darmois

def darmois_linear_gaussian(A):
    '''
    Returns a Darmois-like construction (and its inverse) for Gaussian sources
    in arbitrary dimension D
    
    N.B. This is still rather heuristic!
    '''
    A_inv = np.linalg.inv(A)
    _, R = np.linalg.qr(A_inv)
    R_inv = np.linalg.inv(R)
    
    def darmois(x):
        y = R @ x
        return 0.5*(1.0 + special.erf(y))

    def inv_darmois(y):
        y = special.erfinv(2*y - 1.0)
        return R_inv @ y
    
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
        re = np.real(y)
        imag = np.imag(y)
        return np.array([re, imag])
    
    def conformal_map_gridplot(x_0, x_1):
#         x_0-= 0.5
#         x_1-= 0.5
        z = x_0 + x_1*1j
        y = nonlinearity(z)
        re = np.real(y)
        imag = np.imag(y)
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
            frac = np.sum((x-a)**2) #is this correct?
            frac = frac**(-1)
        else:
            diff = np.abs(x-a)
            
            frac = 1.0
        return b + frac * alpha * A @ (x - a)
    
    def unmixing_moebius_transform(x):
        # TO DO: implement inverse of the above transformation
        return 
    
    return mixing_moebius_transform, unmixing_moebius_transform