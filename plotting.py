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
    X : (N,D) array -- N samples, D dimensions (D=2).ss
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
    
def plot_histograms(hist_values, labels, xlabel):    
    '''
    1d Histograms
    hist_values : a list of arrays for which we want a histogram 
    labels : a number of labels for the legend -- same length as hist_values
    xlabel : what we want to be written under the x axis
    '''
    plt.hist(hist_values, 
             histtype='step', 
             density=True, 
             label= labels
            )
    plt.legend(prop={'size': 10})
    plt.xlabel(xlabel)
    plt.show