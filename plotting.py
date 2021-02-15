'''
Various plotting functions I am using over and over
'''

from jax import numpy as np
import matplotlib.pyplot as plt

def cart2pol(x, y):
    '''
    From cartesian to polar coordinates
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def scatterplot_variables(X, title, colors='None', cmap='hsv'):
    '''
    Scatterplot of 2d variables, can be used both for the mixing and the unmixing
    '''
    if colors=='None':
        plt.scatter(X[:,0], X[:,1], color='r', s=30)
    else:
        plt.scatter(X[:,0], X[:,1], c=colors, s=30, alpha=0.75, cmap=cmap)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()