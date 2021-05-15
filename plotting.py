'''
Various plotting functions I am using over and over
'''

from jax import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as onp

def cart2pol(x, y):
    '''
    From cartesian to polar coordinates
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    '''
    From polar to cartesian coordinates
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def scatterplot_variables(X, title, colors='None', cmap='hsv', savefig=False,
                          fname="scatterplot", show=True):
    '''
    Scatterplot of 2d variables, can be used both for the mixing and the unmixing
    X : (N,D) array -- N samples, D dimensions (D=2).ss
    '''
    if colors=='None':
        plt.scatter(X[:,0], X[:,1], color='r', s=30)
    else:
        plt.scatter(X[:,0], X[:,1], c=colors, s=30, alpha=0.75, cmap=cmap)
    if title=="Sources":
        plt.xlabel('s_1')
        plt.ylabel('s_2')
    elif title=="Observations":
        plt.xlabel('x_1')
        plt.ylabel('x_2')
    elif title=="Reconstructed" or title=="Reconstructions":
        plt.xlabel('y_1')
        plt.ylabel('y_2')
    else:
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    if savefig:
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    if show:
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

'''
Various functions for grid plots
'''
    
def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = onp.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    
def show_grid_plot(f, multi_argument=False, extremes=(0,1), savefig=False, fname="grplot"):
    '''
    Plots how a regularly spaced grid in a 2d space is distorted under the action of the function f
    
    f: A mixing function
    multi_argument: A Boolean variable; checks whether f takes a (N,2) array as input, or two (N,) arrays.
                    In the latter case, internally builds a version of f which takes two (N,) arrays as input.
    '''
    
    if multi_argument==False:
        def f_grid(x, y):
            z = np.array([x, y])
            z_ = f(z)
            return z_[0], z_[1]
    else:
        f_grid = f

    bottom, top = extremes
    
    fig, ax = plt.subplots()

    grid_x,grid_y = np.meshgrid(onp.linspace(bottom,top,20),onp.linspace(bottom,top,20))
    plot_grid(grid_x,grid_y, ax=ax,  color="lightgrey")

    distx, disty = f_grid(grid_x,grid_y)
    plot_grid(distx, disty, ax=ax, color="C0")
 
    plt.gca().set_aspect('equal', adjustable='box')
    if savefig==True:
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    plt.show()