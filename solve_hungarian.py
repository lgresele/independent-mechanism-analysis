import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

def SolveHungarian( recov, source, correlation='Pearson' ):
    """
    compute maximum correlations between true indep components and estimated components 
    """
    Ncomp = source.shape[1]
    if correlation == 'Pearson':
        CorMat = (np.abs(np.corrcoef( recov.T, source.T ) ) )[:Ncomp, Ncomp:]
    elif correlation == 'Spearman':
        rho, _ = np.abs(spearmanr( recov, source ) )
        CorMat = rho[:Ncomp, Ncomp:]
    ii = linear_sum_assignment( -1*CorMat )

    return CorMat[ii].mean(), CorMat, ii