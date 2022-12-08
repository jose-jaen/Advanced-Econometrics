# Import libraries
import numpy as np
import pandas as pd

def iv_estimator(data, target, endo_vars, exo_vars, instr):
    """ Computes IV estimator

    - Parameters:
        - data = Pandas Dataframe with data
        - target = Index of dependent variable
        - endo_vars = List with names of endogenous regressors
        - exo_vars = List with names of exogenous regressors
        - instr = List with names of instrumental variables

    - Output:
        - iv_estim = IV Estimator
    """
    # Check if the equation is underidentified
    if len(exo_vars) + len(instr) + 1 < len(endo_vars):
        raise ValueError('Equation is underindentified, try again')

    # Check if the equation is overidentified
    elif len(exo_vars) + len(instr) + 1 > len(endo_vars):
        raise ValueError('Equation is overidentified, try the GMM estimator!')

    # If K = L then calculate IV Estimator
    else:
        # Include an intercept to the dataset
        constant = [1]*data.shape[0]
        data.insert(0, column='intercept', value=constant)
        exo_vars.insert(0, 'intercept')

        # Define the matrix of exogenous features
        Z = data.loc[:, exo_vars + instr]
        Z = Z.to_numpy()

        # Matrix of endogenous regressors
        X = data.loc[:, endo_vars]
        X = X.to_numpy()

        # Feature vector
        y = data.loc[:, target]
        y = y.to_numpy()

        # Cross-product matrix between exogenous and endogenous variables
        S_zx = np.matmul(Z.T, X)

        # Dot product of exogenous regressors and target variable
        S_zy = np.matmul(Z.T, y)

        # Compute the inverse (or pseudo-inverse)
        try:
            S_zx_inv = np.lingalg.inv(S_zx)
        except:
            S_zx_inv = np.linalg.pinv(S_zx)

        # Retrieve final estimator vector
        iv_estim = np.matmul(S_zx_inv, S_zy)
    return iv_estim

def gmm_estimator(data, target, endo_vars, exo_vars, instr):
    """ Computes efficient GMM estimator

    - Parameters:
        - data = Pandas Dataframe with data
        - target = Index of dependent variable
        - endo_vars = List with names of endogenous regressors
        - exo_vars = List with names of exogenous regressors
        - instr = List with names of instrumental variables

    - Output:
        - iv_estim = GMM Estimator
    """
    # Check if the equation is underidentified
    if len(exo_vars) + len(instr) + 1 < len(endo_vars):
        raise ValueError('Equation is underindentified, try again')

    # Check if the equation is exactly identified
    elif len(exo_vars) + len(instr) + 1 == len(endo_vars):
        raise ValueError('Equation is exactly identified, try the IV estimator!')

    # If K > L then calculate GMM Estimator
    else:
        # Include an intercept to the dataset
        constant = [1]*data.shape[0]
        data.insert(0, column='intercept', value=constant)
        try:
             exo_vars.insert(0, 'intercept')
        except:
            exo_vars = ['intercept']
            
        # Define the matrix of exogenous features
        Z = data.loc[:, exo_vars + instr]
        Z = Z.to_numpy()

        # Matrix of endogenous regressors
        X = data.loc[:, endo_vars]
        X = X.to_numpy()

        # Feature vector
        y = data.loc[:, target]
        y = y.to_numpy()

        # Cross-product matrix between exogenous and endogenous variables
        S_zx = np.matmul(Z.T, X)

        # Dot product of exogenous regressors and target variable
        S_zy = np.matmul(Z.T, y)

        # Cross product of instruments
        S_zz = np.matmul(Z.T, Z)

        # Initialize weighting matrix
        try:
            W = np.linalg.inv(S_zz)
        except:
            W = np.linalg.pinv(S_zz)

        # Compute the first iteration of GMM estimator
        lhs = np.linalg.inv(np.matmul(np.matmul(S_zx.T, W), S_zx))
        rhs = np.matmul(np.matmul(S_zx.T, W), S_zy)
        gmm_estim = np.matmul(lhs, rhs)

        # Residual vector
        resid = y - np.dot(X, gmm_estim)

        # Calculate variance of g_i
        S = np.dot(np.sum(resid**2), S_zz)
        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.linalg.pinv(S)

        # Retrieve final GMM estimator
        lhs = np.linalg.inv(np.matmul(np.matmul(S_zx.T, S_inv), S_zx))
        rhs = np.matmul(np.matmul(S_zx.T, S_inv), S_zy)
        gmm_estim = np.matmul(lhs, rhs)
    return gmm_estim
