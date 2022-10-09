# Import required packages
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

def OLS_estimator(feature, target, intercept: bool):
    """ Retrieves the OLS estimator for some dataframe

    - Parameters:
        - feature = Pandas dataframe containing feature matrix
        - taget = Pandas dataframe containing target vector
        - intercept = Whether to include an intercept or not
    
    - Output:
        - b = OLS estimator
    """
    # Include intercept if specified
    if intercept:
        feature.insert(loc=0, column='intercept', value=1)
    
    # Transform to numpy array for computational efficiency
    features = feature.to_numpy()
    target = target.values

    # Compute inverse of cross-product matrix X'X 
    try:
        inverse = inv(np.matmul(features.T, features))
    except:
        # If X'X is singular, calculate the Moore-Penrose inverse
        inverse = pinv(np.matmul(features.T, features))

    # Define X'y
    covariance = np.matmul(features.T, target)
    b = np.matmul(inverse, covariance)
    return b


def sigma_estimator(feature, target, intercept: bool):
    """ Retrieves the variance estimator for an OLS coefficient vector

    - Parameters:
        - feature = Pandas dataframe containing feature matrix
        - taget = Pandas dataframe containing target vector
        - intercept = Whether to include an intercept or not
    
    - Output:
        - s2 = OLS variance estimator
    """
    # Retrieve OLS estimator
    b = OLS_estimator(feature, target, intercept)

    # Transform to numpy array for computational efficiency
    features = feature.to_numpy()
    target = target.values

    # Store matrix dimensions
    n, k = len(target), features.shape[1]

    # Compute inverse of cross-product matrix X'X 
    try:
        inverse = inv(np.matmul(features.T, features))
    except:
        # If X'X is singular, calculate the Moore-Penrose inverse
        inverse = pinv(np.matmul(features.T, features))

    # Calculate squared residual vector e'e
    y_hat = np.matmul(features, b)
    residuals = target - y_hat
    resid2 = np.dot(residuals.T, residuals)

    # Degrees of freedom correction
    resid2 /= n-k
    
    # Define S^2
    s2 = resid2*inverse
    return s2
