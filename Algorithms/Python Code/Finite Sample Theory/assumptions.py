from typing import Union, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.stats import chi2, norm
from statsmodels.stats.diagnostic import lilliefors

from sklearn.linear_model import LinearRegression


class AssumptionChecker:
    def __init__(
            self,
            covariates: pd.DataFrame,
            response: Union[pd.Series, np.array]
    ):
        self.covariates = covariates
        self.response = response
        self.continuous: List[Optional[str]] = []

    def _fit_intercept(self) -> None:
        """Include a constant in the dataset if not already present."""
        have_intercept = 0
        for covariate in self.covariates.columns:
            unique_values = self.covariates[covariate].unique()
            if len(unique_values) < 2 and unique_values == 1:
                have_intercept = 1

        # Include intercept
        if not have_intercept:
            total_columns = list(self.covariates.columns)
            self.covariates['intercept'] = 1
            self.covariates = self.covariates.loc[:, ['intercept'] + total_columns]

    def _get_continuous(self, response_name: str):
        """Identify continuous variables in the dataset."""
        for col in self.covariates.columns:
            unique_values = len(self.covariates[col].unique()) > 20
            is_continuous = unique_values and self.covariates[col].dtype != 'object'
            if is_continuous and col != response_name:
                self.continuous.append(col)

    def check_linearity(self, response_name: str) -> None:
        """Display scatterplots to visually analyze linearity."""
        self._get_continuous(response_name=response_name)

        # Show graphs
        for covariate in self.continuous:
            fig, ax = plt.subplots()
            ax.scatter(self.covariates[covariate], self.response)

            # Include a straight line
            line = mlines.Line2D([0, 1], [0, 1], color='red')
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)

            # Add labels
            plt.xlabel(covariate)
            plt.ylabel('Target')
            plt.title('Linearity Assumption')
            plt.show()

    def check_multicollinearity(self) -> int:
        """Test no multicollinearity assumption and retrieve matrix rank."""
        self._fit_intercept()

        # Check container for continuous regressors
        if not self.continuous:
            self._get_continuous(response_name='retailvalue')

        # Calculate rank of the covariates matrix
        rank = np.linalg.matrix_rank(self.covariates)
        if rank == self.covariates.shape[1]:
            print('A.3 is satisfied: covariates are linearly independent')
        else:
            print('A.3 is not satisfied: covariates are linearly dependent')

        # Visual representation
        correlations = np.corrcoef(
            self.response,
            self.covariates.loc[:, self.continuous],
            rowvar=False
        )
        correlations = correlations[0][1:]
        correlation_data = pd.DataFrame(
            {
                'Variable': self.continuous,
                'Correlation': correlations
            }
        )

        # Take absolute values and sort
        correlation_data['AbsCorrelation'] = correlation_data['Correlation'].abs()
        sorted_data = correlation_data.sort_values(
            by='AbsCorrelation',
            ascending=False
        )

        # Plot the sorted data
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_data['Variable'], sorted_data['AbsCorrelation'])
        plt.xlabel('Variables')
        plt.ylabel(f"Absolute Correlation with response")
        plt.title(f"Covariates Sorted by Absolute Correlation with response")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Warn about high multicollinearity
        high_corr = correlations[np.where(abs(correlations) >= 0.9)]
        abs_high_corr = [abs(j) for j in high_corr]
        condition = sorted_data['AbsCorrelation'].isin(abs_high_corr)
        suspects = sorted_data.loc[condition, 'Variable'].tolist()

        # Warning message
        if len(suspects) > 1:
            suspects = ', '.join(suspects)
            print(
                f"""Covariates {suspects} are highly correlated! """
                """It's advisable to ditch them out!"""
            )
        elif len(suspects) == 1:
            print(
                f"""Covariates {suspects[0]} is highly correlated! """
                """It's advisable to ditch it out!"""
            )
        return rank

    def check_homoskedasticity(
            self,
            resid: Union[pd.Series, np.array]
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """Perform Breusch-Pagan homoskedasticity test.

        Args:
            resid: Vector of residuals from the main regression

        Return:
            lm (float): Test statistic / Lagrange Multiplier
            p-value (float): Associaded p-value with test statistic
        """
        variance_auxiliary = np.sum(resid ** 2) / len(resid)
        target_auxiliary = np.divide(resid ** 2, variance_auxiliary)

        # Estimate new auxiliary regression
        include_constant = False if 'intercept' in self.covariates.columns else True
        regression_aux = LinearRegression(fit_intercept=include_constant)
        regression_aux.fit(self.covariates, target_auxiliary)
        coefs_aux = regression_aux.coef_

        # Calculate test statistic
        tss = np.sum(np.power(target_auxiliary - 1, 2))
        ssr = np.sum(np.power(target_auxiliary - self.covariates @ coefs_aux, 2))
        lm = 0.5 * (tss - ssr)

        # Retrieve p-value
        p_value = 1 - chi2.cdf(lm, df=len(coefs_aux) - 1)

        # Output result
        if p_value <= 0.05:
            print('A.4 is not satisfied: reject that residuals are homoskedastic')
        else:
            print('A.4 is satisfied: cannot reject that residuals are homoskedastic')

        # Visual representation
        main_regression = LinearRegression(fit_intercept=include_constant)
        main_regression.fit(self.covariates, self.response)
        coefficients = main_regression.coef_
        fitted = self.covariates @ coefficients
        plt.scatter(fitted, resid)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Homoskedasticity Assumption Check')
        plt.show()
        return lm, p_value

    @staticmethod
    def check_normality(
            resid: Union[pd.Series, np.array]
    ) -> Tuple[float, float]:
        """Perform Liellifors variant of Kolmogorov-Smirnov test."""
        kstat, p_value = lilliefors(resid, dist='norm', pvalmethod='table')

        # Output result
        if p_value <= 0.05:
            print(
                'A.5 is not satisfied: reject that residuals follow a normal distribution'
            )
        else:
            print(
                'A.5 is satisfied: cannot reject that residuals follow a normal distribution'
            )

        # Plot histogram of the data
        plt.hist(resid, bins=8, density=True, alpha=0.6, color='g', edgecolor='black')

        # Fit a normal distribution to the data
        mu, std = np.mean(resid), np.std(resid)

        # Density function
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=2)

        # Labels and title
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Distribution of residuals with Normal Fit')

        # Show the plot
        plt.show()
        return round(kstat, 4), round(p_value, 4)
