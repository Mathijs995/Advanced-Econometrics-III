# -*- coding: utf-8 -*-
"""
Created on Wed Mar  20 15:37:22 2019

@author: Jakob
"""

###########################################################
### Imports
import pandas as pd
import numpy as np
import scipy as scipy
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib

# Print options
np.set_printoptions(precision=4, threshold=10000, linewidth=150, suppress=True)


def regressors_given_p(fd_X, X, p):
    """Purpose: Returns the regressor matrix mX and the regressand vector vY
    for a given number of lags p (now added a constant)
    Inputs: fd_X, X, p
    Returns: vY,mX
    """
    ones = np.ones((len(X)-p-1))
    ones = ones.reshape((ones.shape[0],1))
    vY = fd_X[p:]
    mX = np.concatenate((ones, X[p:-1], np.concatenate([fd_X[p - i - 1: - i - 1] for i in range(p)], axis=1)), axis=1)
    
#    print("fd_X=\n{}\n".format(fd_X[0:5,]))
#    print("X=\n{}\n".format(X[0:5,]))
#    print("vY =\n{}\n".format(vY[0:5,]))
#    print("mX =\n{}\n".format(mX[0:5,]))
    
    return vY, mX


def ols(vY,mX):
    """Purpose: Returns the ols estimate vector vB
    Inputs: vY,mX
    Returns: vB
    """
    vB=np.linalg.inv(mX.T@mX)@mX.T@vY
    
    return vB

def bic(vY,mX, vB):
    """Purpose: value of BIC criterion given data and parameters
    Inputs: vY,mX,vB
    Returns: BIC (scalar)
    """
    n=len(vY)
    k=len(mX.T)
    resid = vY - mX@vB
    sse = sum(resid**2)
    BIC = n*np.log(sse/n) + k*np.log(n)
    
    return BIC

def tstat(vY,mX,vB, par, h0):
    """Purpose: Returns the t-statistic for the given parameter par and h0
    Inputs: vY,mX,vB, par
    Returns: t
    """
    n=len(vY)
    k=len(mX.T)
    resid = vY - mX@vB
    sse = sum(resid**2)
    mCovhat = np.linalg.inv(mX.T@mX)*sse/(n-k) # Estimate covariance matrix
    t = (vB[par]-h0)/np.sqrt(np.diag(mCovhat)[par])
    
    return t

###########################################################
### main
def main():
    # Inputs
    data = pd.read_csv("C:/Users/Jakob/Documents/adv ectrics III/VIX.csv")
    X = np.log(data["VIX"].values)
    fd_X = np.diff(X)
    X = X.reshape((X.shape[0],1))
    fd_X = fd_X.reshape((fd_X.shape[0], 1))
    
    MAX_LAG = 50
    T_DELTA = -2.86
    

    # Initialisation
    
    plot_acf(X, lags=range(1, MAX_LAG + 1))
    plt.show()
    plot_acf(fd_X, lags=range(1, MAX_LAG + 1))
    plt.show()
    
    #Finding the optimal number of lags p
    p = 1
    for i in range(10):
        vY, mX = regressors_given_p(fd_X, X, p)
        vB = ols(vY,mX)
        BIC = bic(vY,mX, vB)
        print('BIC for ', p, ' lags is ', BIC)
        p += 1
        
    # THE OPTIMAL NUMBER OF LAGS IS 4! -> SET p=4 subsequently
    p = 4
    
    # Now do Dickey-Fuller test:
    vY, mX = regressors_given_p(fd_X, X, p)
    vB = ols(vY,mX)
    par = 1 # We are interested in the second parameter delta (first is the constant)
    h0 = 0 # We test h0: X is integrated of order one so then delta is 0
    t=tstat(vY,mX,vB, par, h0)
    print('The point estimate for delta is ', vB[par])
    print('The t-statistic for testing whether the process is integrated of ',
          'order one is', t)
# =============================================================================
# We get a high t statistic b/c even though delta hat is close to 0,
# it is very precisely estimated (n=6800). In conclusion, h0 is rejected and the 
# process is NOT integrated of order one.
# =============================================================================
    
    # Estimate spectral density
    f,Pxx  = scipy.signal.periodogram(fd_X, window='bartlett')

 

###########################################################
### start main
if __name__ == "__main__":
    main()
