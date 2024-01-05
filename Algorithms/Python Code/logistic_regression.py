from typing import Union, Optional, Any, Tuple, List

import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(
            self,
            tol: float = 0.0001,
            l2_penalty: Union[int, float] = 0,
            fit_intercept: bool = True,
            random_state: Optional[int] = 42,
            solver: str = 'bfgs',
            max_iter: int = 150,
            learning_rate: Union[int, float] = 0.1,
            beta: float = 0.9,
            batch_size: int = 256,
            verbose: bool = False
    ):
        self.tol = tol
        self.l2_penalty = l2_penalty
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.verbose = verbose
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[Union[int, float]] = None
        self.grad_size: float = float('inf')
        self.has_intercept: bool = False
        self.inv_hessian: Union[int, np.ndarray] = 0
        self.n_iter: int = 0

    @staticmethod
    def _check_type(
            att_name: str,
            att: Any,
            right_type: Union[type, Tuple[type, type]]
    ) -> None:
        """Check if attributes have the correct type.

        Args:
            att_name: Attribute name
            att: Attribute to be checked
            right_type: Expected datatype
        """
        att_type = type(att).__name__
        if not isinstance(att, right_type):
            raise TypeError(
                f"'{att_name}' must be '{right_type.__name__}' but got '{att_type}'"
            )

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        self._check_type(att_name='tol', att=tol, right_type=float)
        if tol >= 1 or tol <= 0:
            raise ValueError(f"'tol' must be between '0' and '1' but got '{tol}'")
        self._tol = tol

    @property
    def l2_penalty(self) -> Union[int, float]:
        return self._l2_penalty

    @l2_penalty.setter
    def l2_penalty(self, l2_penalty: Union[int, float]) -> None:
        self._check_type(
            att_name='l2_penalty',
            att=l2_penalty,
            right_type=(int, float)
        )
        if l2_penalty < 0:
            raise ValueError(f"'l2_penalty' must be strictly positive, got '{l2_penalty}'")
        self._l2_penalty = l2_penalty

    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept: bool) -> None:
        self._check_type(
            att_name='fit_intercept',
            att=fit_intercept,
            right_type=bool
        )
        self._fit_intercept = fit_intercept

    @property
    def random_state(self) -> Optional[int]:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]) -> None:
        self._check_type(
            att_name='random_state',
            att=random_state,
            right_type=(Optional[int])
        )
        if random_state < 0:
            random_state = -random_state
        self._random_state = random_state

    @property
    def solver(self) -> str:
        return self._solver

    @solver.setter
    def solver(self, solver: str) -> None:
        self._check_type(att_name='solver', att=solver, right_type=str)
        valid = ['newton', 'gradient', 'bfgs', 'coordinate', 'batch', 'conjugate']
        if solver not in valid:
            raise ValueError(f"'solver' must be in '{valid}' but got '{solver}'")
        self._solver = solver

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self._check_type(att_name='max_iter', att=max_iter, right_type=int)
        if max_iter < 0:
            raise ValueError(f"'max_iter' must be strictly positive but got '{max_iter}'")
        self._max_iter = max_iter

    @property
    def learning_rate(self) -> Union[int, float]:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: Union[int, float]) -> None:
        self._check_type(
            att_name='learning_rate',
            att=learning_rate,
            right_type=(int, float)
        )
        if learning_rate < 0:
            raise ValueError(
                f"'learning_rate' must be strictly positive but got '{learning_rate}'"
            )
        self._learning_rate = learning_rate

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        self._check_type(att_name='beta', att=beta, right_type=(int, float))
        if beta < 0 or beta > 1:
            raise ValueError(f"'beta' must be between 0 and 1 but got '{beta}'")
        self._beta = beta

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._check_type(
            att_name='batch_size',
            att=batch_size,
            right_type=int
        )
        if batch_size < 0:
            raise ValueError(f"'batch_zie' must be strictly positive!")
        self._batch_size = batch_size

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool) -> None:
        self._check_type(att_name='verbose', att=verbose, right_type=bool)
        self._verbose = verbose

    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility."""
        if self.random_state:
            np.random.seed(self.random_state)

    @staticmethod
    def _positive_sigmoid(x: np.ndarray) -> np.ndarray:
        """Aliviate overflow for positive values"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _negative_sigmoid(x: np.ndarray) -> np.ndarray:
        """Aliviate overflow for negative values."""
        exp = np.exp(x)
        return exp / (1 + exp)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply stable sigmoid link function to an array."""
        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x, dtype=np.float32)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])
        return result

    def _neg_log_likelihood(
            self,
            regressors: np.ndarray,
            target: np.ndarray,
            coefs: np.ndarray
    ) -> Union[int, float]:
        """Compute (stable) negative log-likelihood at current iteration.

        Args:
            regressors: Feature matrix
            target: Response variable vector
            coefs: Vector of weights
        """
        epsilon = 1.e-15
        n = len(target)

        # Get components of the objective function
        preds = self._sigmoid(x=regressors @ coefs)
        complement = 1 - preds
        preds[preds == 0] = epsilon
        complement[complement == 0] = epsilon
        reg = (1 / (2*n)) * self._l2_penalty * np.linalg.norm(x=coefs, ord=2)
        positive_class = np.dot(a=target, b=np.log(preds))
        negative_class = np.dot(a=(1 - target), b=np.log(complement))
        neg_log_likelihood = - (1 / n) * np.sum(positive_class + negative_class) + reg
        return neg_log_likelihood

    def _get_intercept(self, regressors: np.ndarray) -> Optional[int]:
        """Identify if there is a constant in the features and change its position.

        Args:
            regressors: Feature matrix
        """
        p = regressors.shape[1]
        for covariate in range(p):
            unique_vals = set(regressors[:, covariate])
            if len(unique_vals) == 1 and 1 in unique_vals:
                self.has_intercept = True
                return covariate
        return None

    def _get_gradient(
            self,
            regressors: np.ndarray,
            target: np.ndarray
    ) -> np.ndarray:
        """Compute logistic regression gradient.

        Args:
            regressors: Feature matrix
            target: Response variable vector
        """
        # Get predictions
        preds = self._sigmoid(regressors @ self.coef_)

        # Accomodate Stochastic Gradient Descent
        if not isinstance(target, (np.ndarray, list)):
            gradient = regressors * (preds - target)
        else:
            gradient = regressors.T @ (preds - target)

        # Do not regularize intercept (if any)
        if self.has_intercept:
            penalty = self._l2_penalty * self.coef_[1:]
            gradient[1:] += penalty
        else:
            penalty = self._l2_penalty * self.coef_
            gradient += penalty
        self.grad_size = np.linalg.norm(x=gradient)
        return gradient

    def _get_hessian(
            self,
            regressors: np.ndarray
    ) -> np.ndarray:
        """Compute logistic regression Hessian matrix.

        Args:
            regressors: Feature matrix
        """
        preds = self._sigmoid(x=regressors @ self.coef_)
        variance = np.diag(preds * (1 - preds))
        fisher_matrix = regressors.T @ variance @ regressors
        if self._l2_penalty:
            identity = np.eye(N=len(self.coef_), dtype=np.float32)
            identity[0, 0] = 0.0 if self.has_intercept else 1.0
            fisher_matrix += self._l2_penalty * identity
        return fisher_matrix

    def gradient_descent(self, regressors: np.ndarray, target: np.ndarray) -> None:
        """Update coefficients implementing Gradient Descent algorithm.

        Args:
            regressors: Feature matrix
            target: Response variable vector
        """
        # Update coefficients
        gradient = self._get_gradient(regressors=regressors, target=target)
        self.coef_ -= self._learning_rate * gradient

    def newton_rapshon(self, regressors: np.ndarray, target: np.ndarray) -> None:
        """Update coefficients implementing Newton-Raphson algorithm.

        Args:
            regressors: Feature matrix
            target: Response variable vector
        """
        # Compute gradient
        gradient = self._get_gradient(regressors=regressors, target=target)

        # Get Hessian
        hessian = self._get_hessian(regressors=regressors)

        # Update coefficients
        self.coef_ -= np.linalg.inv(hessian) @ gradient

    def bfgs(
            self,
            past_coefs: np.ndarray,
            past_gradient: np.ndarray,
            regressors: np.ndarray,
            target: np.ndarray
    ) -> None:
        """Update coefficients implementing BFGS algorithm.

        Args:
            past_coefs: Coefficient vector from the previous iteration
            past_gradient: Gradient from the previous iteration
            regressors: Regressors matrix
            target: Vector with response variable
        """
        p = len(self.coef_)

        # Coefficient difference
        diff_coef = self.coef_ - past_coefs

        # Gradient difference
        current_gradient = self._get_gradient(regressors=regressors, target=target)
        diff_grad = current_gradient - past_gradient

        # Update inverse hessian approximation
        if np.dot(diff_grad, diff_coef) > 0:
            rho = 1.0 / (diff_grad.T @ diff_coef)
            lhs = np.eye(N=p) - rho * np.outer(diff_coef, diff_grad)
            rhs = np.eye(N=p) - rho * np.outer(diff_grad, diff_coef)
            addition = rho * np.outer(diff_coef, diff_coef)
            self.inv_hessian = lhs @ self.inv_hessian @ rhs + addition

    def coordinate(self, regressors: np.ndarray, target: np.ndarray) -> None:
        """Update coefficients implementing Coordinate Gradient method.
        Update either a random coefficient per iteration or the coefficient
        associated with the variable with highest absolute value in the gradient.

        Args:
            regressors: Feature matrix
            target: Response variable vector
        """
        # Get the gradient
        gradient = self._get_gradient(regressors=regressors, target=target)

        # Random index or greatest absolute value in gradient
        choice = np.random.binomial(n=1, p=0.7, size=1)[0]
        if not choice:
            max_value = np.argmax(a=abs(gradient))
            self.coef_[max_value] -= self._learning_rate * gradient[max_value]
        else:
            index = np.random.randint(low=0, high=len(gradient) - 1, size=1)[0]
            self.coef_[index] -= self._learning_rate * gradient[index]

    def batch_gradient(
            self,
            regressors: np.ndarray,
            target: np.ndarray,
            ma_gradient: Union[int, float]
    ) -> np.ndarray:
        """Update coefficients implementing Mini-Batch Gradient Descent.

        Args:
            regressors: Feature matrix
            target: Response variable vector
            ma_gradient: Moving Average of the gradient

        Returns:
            Updated coefficients
        """
        k = 0
        # Run algorithm for the necessary epochs
        while k + self._batch_size < len(target):
            # Set up regressors and target
            x_batch = regressors[k:k + self._batch_size]
            y_batch = target[k:k + self._batch_size]

            # Update coefficients
            gradient = self._get_gradient(regressors=x_batch, target=y_batch)
            momentum = self._beta * ma_gradient + (1 - self._beta) * gradient
            self.coef_ -= self._learning_rate * (1 - self._beta**self.n_iter) * momentum
            k += self._batch_size

        # Iterate over the remaining observations
        n = len(target) - 1
        x_batch = regressors[k: n] if k < n else regressors[-1]
        y_batch = target[k: n] if k < n else target[-1]

        # Final update of coefficients
        gradient = self._get_gradient(regressors=x_batch, target=y_batch)
        momentum = self._beta * ma_gradient + (1 - self._beta) * gradient
        self.coef_ -= self._learning_rate * (1 - self._beta**self.n_iter) * momentum
        return self.coef_

    def fit(
            self,
            regressors: Union[np.ndarray, pd.DataFrame],
            target: Union[np.ndarray, pd.Series, List[int]]
    ) -> None:
        """Estimate logistic regression model with a specific solver.

        Args:
            regressors: Feature matrix
            target: Response variable vector
        """
        # Convert to numpy for higher efficiency
        if isinstance(regressors, pd.DataFrame):
            regressors = regressors.to_numpy()
        if isinstance(target, pd.Series):
            target = target.to_numpy()
        elif isinstance(target, list):
            target = np.asarray(a=target, dtype=np.int32)

        # Include intercept
        n = len(target)
        if self._fit_intercept:
            regressors = np.c_[np.ones(shape=(n,)), regressors]
            self.has_intercept = True
        else:
            # Verify there is no intercept otherwise place it first
            index = self._get_intercept(regressors=regressors)
            if isinstance(index, int):
                intercept = regressors[:, index]
                regressors = np.c_[intercept, np.delete(arr=regressors, obj=index, axis=1)]

        # Initialize coefficients
        p = regressors.shape[1]
        log_likelihood = 0
        self.coef_ = np.random.normal(size=p, scale=0.001)

        # Initialize momentum
        ma_gradient = 0

        # Initialize step and beta
        step = 0
        beta = 0

        # Initialize inverse hessian approximation
        if self._solver == 'bfgs':
            self.inv_hessian = np.eye(N=regressors.shape[1])

        # Fit algorithm to data
        diff_grad = float('inf')
        while self.n_iter < self._max_iter and self.grad_size > self._tol < diff_grad:
            self.n_iter += 1

            # Get gradient
            past_gradient = self._get_gradient(regressors=regressors, target=target)

            # Gradient Descent
            if self._solver == 'gradient':
                self.gradient_descent(regressors=regressors, target=target)

            # Newton-Raphson
            elif self._solver == 'newton':
                self.newton_rapshon(regressors=regressors, target=target)

            # BFGS (Quasi-Newton)
            elif self._solver == 'bfgs':
                past_coef = self.coef_.copy()
                self.coef_ -= self._learning_rate * self.inv_hessian @ past_gradient
                self.bfgs(
                    past_coefs=past_coef,
                    past_gradient=past_gradient,
                    regressors=regressors,
                    target=target
                )

            # Coordinate Descent
            elif self._solver == 'coordinate':
                self.coordinate(regressors=regressors, target=target)

            # Mini Batch Gradient Descent (with Momentum)
            elif self._solver == 'batch':
                if self._batch_size > len(target):
                    aux = f"Maximum: '{len(target)}', got '{self._batch_size}'"
                    raise ValueError(f"'batch_size' cannot exceed the training instances! " + aux)

                # Update coefficients
                self.batch_gradient(
                    regressors=regressors,
                    target=target,
                    ma_gradient=ma_gradient
                )
                gradient = self._get_gradient(regressors=regressors, target=target)
                ma_gradient = self._beta * ma_gradient + (1 - self._beta) * gradient

            elif self._solver == 'conjugate':
                step = - past_gradient + beta * step
                self.coef_ += self._learning_rate * step
                grad = - self._get_gradient(regressors=regressors, target=target)
                beta = np.dot(grad, grad) / np.dot(-past_gradient, -past_gradient)

            # Check changes in gradient
            current_gradient = self._get_gradient(regressors=regressors, target=target)
            diff_grad = np.linalg.norm(current_gradient - past_gradient)
            log_likelihood = self._neg_log_likelihood(
                regressors=regressors,
                target=target,
                coefs=self.coef_
            )

            # Convergence information
            if self._verbose and not self.n_iter % 10:
                print(f'Iteration {self.n_iter}')
                print(f'Negative log-likelihood: {round(log_likelihood, 4)}')
                print(f'Gradient size: {round(self.grad_size, 4)}')
                print(f'Gradient difference: {round(diff_grad, 4)}')
                print('\n')

        # Set up coefficient attributes
        if self.has_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]

        if self._verbose and self.n_iter == self._max_iter:
            print('Convergence not attained, consider increasing the number of iterations')
            print(f'Negative log-likelihood: {round(log_likelihood, 4)}')
            print(f'Gradient size: {round(self.grad_size, 4)}')
            print(f'Gradient difference: {round(diff_grad, 4)}')
            print('\n')
