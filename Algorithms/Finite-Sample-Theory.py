# Import required packages
import numpy as np
from scipy import stats
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
    try:
        if intercept:
            feature.insert(loc=0, column='intercept', value=1)
    except:
        features = feature
    
    # Transform to numpy array for computational efficiency
    features = feature.to_numpy()
    y = target.values

    # Compute inverse of cross-product matrix X'X 
    try:
        inverse = inv(np.matmul(features.T, features))
    except:
        # If X'X is singular, calculate the Moore-Penrose inverse
        inverse = pinv(np.matmul(features.T, features))

    # Define X'y
    covariance = np.matmul(features.T, y)
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
    y = target.values

    # Store matrix dimensions
    n, k = len(y), features.shape[1]

    # Compute inverse of cross-product matrix X'X 
    try:
        inverse = inv(np.matmul(features.T, features))
    except:
        # If X'X is singular, calculate the Moore-Penrose inverse
        inverse = pinv(np.matmul(features.T, features))

    # Calculate squared residual vector e'e
    y_hat = np.matmul(features, b)
    residuals = y - y_hat
    resid2 = np.dot(residuals.T, residuals)

    # Degrees of freedom correction
    resid2 /= n-k
    
    # Define variance estimator
    s2 = resid2*inverse
    return s2

def individual_test(coef, value, feature, target, intercept: bool, sign: str):
    """ Tests the statistical significance of regression coefficients

    - Parameters:
        - coef = Index of the coefficient to test (0 is intercept)
        - value = Coefficient value to test
        - feature = Pandas dataframe containing feature matrix
        - taget = Pandas dataframe containing target vector
        - intercept = Whether to include an intercept or not
        - sign = Hypothesis test sign (two.sided, less, great)

    - Output:
        - Message with results and p-value
    """
    # Get matrix dimensions for degrees of freedom
    features = feature.to_numpy()
    y = target.values
    n, k = len(y), features.shape[1]
    df = n - k

    # Retrieve OLS estimator and variance estimator
    b = OLS_estimator(feature, target, intercept)
    s2 = sigma_estimator(feature, target, intercept)

    # Compute test statistic
    t = (b[coef] - value)/(np.sqrt(s2[coef, coef]))
    
    # Print results depending on sign
    if sign == 'two.sided':
        pvalue = 2*stats.t.cdf(t, df)
    elif sign == 'less':
        pvalue = stats.t.cdf(t, df)
    else:
        pvalue = 1 - stats.t.cdf(t, df)
    
    # Check if H0 is rejected
    if pvalue <= 0.05:
        print(f'H0 is rejected with p-value {pvalue}')
    else:
        print(f'Failed to reject H0 with p-value {pvalue}')
