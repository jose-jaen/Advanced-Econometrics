# Import required packages
import pandas as pd
import numpy as np
from scipy.stats import t

class OLS:
    """ Retrieves OLS estimator, standard errors and confidence intervals

    - Initialization parameters:
        - target = Pandas dataframe with target variable
        - regressors = Pandas dataframe with predictors
        - intercept = Whether or not to include a constant

    - Methods:
        - data_matrices = Target, regressors and inverted cross-matrix data
        - ols_estimator = Calculates OLS estimator vector
        - ols_se = Computes standard errors of OLS estimator
        - ols_t = Retrieves t-statistic for individual significance
        - ols_F = Obtains F-statistic for joint significance
        - ols_ci = 5% significance level confidence interval of OLS estimator
        - ols_summary = Data matrix with all previous methods' results
    """
    def __init__(self, target, regressors, intercept=True):
        # Assignments
        self.target = target
        self.regressors = regressors

        # Include intercept in data matrix
        if intercept and 'intercept' not in self.regressors.columns:
            constant = [1]*len(self.target)
            self.regressors.insert(loc=0, column='intercept', value=constant)

    def data_matrices(self):
        """" Sets up data matrices for modeling
        """
        regressors = self.regressors
        target = self.target
        regressors, target = regressors.to_numpy(), target.to_numpy()
        cross_matrix = np.matmul(regressors.T, regressors)

        # Invert cross matrix if it is nonsingular
        try:
            cross_matrix_inv = np.linalg.inv(cross_matrix)
        except:
            cross_matrix_inv = np.linalg.pinv(cross_matrix)

        return regressors, target, cross_matrix_inv

    def ols_estimator(self):
        """ Retrieves OLS estimator vector
        """
        regressors, target, cross_matrix_inv = self.data_matrices()
        dot_prod = np.dot(regressors.T, target)
        b = np.matmul(cross_matrix_inv, dot_prod)
        b = [i.round(4) for i in b]
        return b

    def ols_se(self):
        """ Retrieves OLS estimator standard errors (assumes homoskedasticity)
        """
        b = self.ols_estimator()
        regressors, target, cross_matrix_inv = self.data_matrices()
        n, k = len(target), len(b)
        resid = target - np.matmul(regressors, b)
        s2 = sum(resid**2)/(n-k)
        se = np.sqrt(s2*np.diag(cross_matrix_inv))
        return se

    def ols_t(self):
        """ Retrieves t-statistic for individual significance
        """
        b = self.ols_estimator()
        se = self.ols_se()
        t_stat = np.divide(b, se)
        n, k = len(target), len(b)
        return t_stat, 2*t.sf(abs(t_stat), n - k).round(3)

    def ols_F(self):
        """ Calculates F-statistic for joint significance
        """
        b = self.ols_estimator()
        regressors, target, cross_matrix_inv = self.data_matrices()
        n, k = len(target), len(b)
        resid_u = target - np.matmul(regressors, b)
        SSR_U = sum(resid_u**2)
        resid_r = target - [np.mean(target)]*n
        SSR_R = sum(resid_r**2)
        F = ((SSR_R - SSR_U)/(k-1))/(SSR_U/(n - k))
        return F

    def ols_ci(self):
        """ Computes onfidence intervals with 5% significance level
        """
        b = self.ols_estimator()
        se = self.ols_se()
        lb, ub = b - se*1.96, b + se*1.96
        lb = [i.round(3) for i in lb]
        ub = [j.round(3) for j in ub]
        return lb, ub

    def ols_summary(self):
        """ Provides summary of OLS estimation
        """
        cols = self.regressors.columns
        b = self.ols_estimator()
        se = self.ols_se()
        t, p_value = self.ols_t()
        lb, ub = self.ols_ci()
        summary = pd.DataFrame(data=np.column_stack((
            cols, b, se, t, p_value, lb, ub)),
            columns=['features', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]'])
        return summary


class OLS_tests(OLS):
    def __init__(self, target, regressors, intercept=True):
        super().__init__(
            target, regressors, intercept
        )

    def individual_test(self, coef: str, value: float, sign: str):
        """ Returns outcome of individual significance test
        """
        assert sign in ['two.sided', 'less', 'great'], f'Sign argument must be either ' \
            f"'two.sided', 'less' or 'great' but '{sign}' was introduced"
        self.coef = coef
        self.value = value
        self.sign = sign

        # Set up data matrices
        regressors, target = self.regressors, self.target

        # Retrieve OLS estimator and variance estimator
        b = self.ols_estimator()
        s2 = self.ols_se()

        # Compute degrees of freedom
        n, k = len(target), len(b)
        df = n - k

        # Obtain OLS estimate
        cols = list(regressors.columns)
        idx = cols.index(self.coef)
        b_k = b[idx]

        # Calculate test-statistic
        t_stat = (b_k - self.value)/s2[idx]

        # Print results depending on sign
        if self.sign == 'two.sided':
            pvalue = 2*t.sf(abs(t_stat), n - k)
        elif self.sign == 'less':
            pvalue = 1 - t.cdf(t_stat, df)
        else:
            pvalue = t.cdf(t_stat, df)

        # Check if H0 is rejected
        if pvalue <= 0.05:
            print(f'H0 is rejected with p-value {pvalue.round(3)}')
        else:
            print(f'Failed to reject H0 with p-value {pvalue.round(3)}')
