import numpy as np
from typing import Optional, Dict
from regression.polynomial_base_builder import PolynomialBasisBuilder
from regression.regression_helpers import check_degenerate_case

from regression.regression_helpers import KEY_PREDICTED, KEY_R2, KEY_RESIDUALS, check_degenerate_case, EPS


class MinVarPolynomialRegression:
    """
    Minimum-variance polynomial regression for hedge delta estimation.

    This class fits a polynomial function δ(x) to minimize the variance of
    (cashflow - δ(x) * beta) across simulation scenarios.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Forward price features for each scenario.
    y : np.ndarray, shape (n_samples,)
        Target cashflows from the asset.
    beta : np.ndarray, shape (n_samples,)
        Scaling factor (e.g., delivery-month spot sums).
    degree : int, default=3
        Degree of the polynomial to fit.
    poly_type : {'standard','legendre','chebyshev'}, default='standard'
        Polynomial basis used by PolynomialBasisBuilder.
    n_components : int, optional
        If given, PCA components to reduce x before polynomial expansion.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        degree: int = 3,
        poly_type: str = PolynomialBasisBuilder.POLY_STANDARD,
        n_components: Optional[int] = 4,
    ) -> None:
        """
        Initialize the minimum-variance polynomial regression.

        Stores input data and sets up the polynomial basis builder.

        Attributes
        ----------
        self.x : np.ndarray
            Forward price features, shape (n_samples, n_features).
        self.y : np.ndarray
            Target cashflows, shape (n_samples,).
        self.beta : np.ndarray
            Scaling factor for hedge, shape (n_samples,).
        self.degree : int
            Degree of the polynomial.
        self.poly_type : str
            Polynomial basis type.
        self.n_components : Optional[int]
            Number of PCA components to retain (if applicable).
        self._builder : PolynomialBasisBuilder
            Builder for polynomial design matrices.
        self.coef : Optional[np.ndarray]
            Fitted polynomial coefficients (initialized as None).
        """
        self.x: np.ndarray = np.atleast_2d(x)
        self.y: np.ndarray = np.asarray(y).reshape(-1)
        self.beta: np.ndarray = np.asarray(beta).reshape(-1)

        self.degree: int = degree
        self.poly_type: str = poly_type
        self.n_components: Optional[int] = n_components

        # Polynomial design matrix builder
        self._builder = PolynomialBasisBuilder(degree=degree, poly_type=poly_type)

    def regress(self) -> Dict[str, np.ndarray]:
        """
        Perform minimum-variance regression using the stored x, y, and beta.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with:
            - predicted        : δ(x) * beta  (hedge value)
            - r2               : R² score of hedge value vs y
            - residuals        : y − δ(x)*beta
            - condition_number : Condition number of design matrix
        """
        # -------------------------
        # Check for degenerate case: all features almost constant
        # -------------------------
        check_degenerate_case(self.x, self.y)

        # Build polynomial design matrix using the reusable builder
        x_poly = self._builder.build(self.x)

        # Each column scaled by beta for hedge cashflow
        A = x_poly * self.beta[:, None]

        # Ordinary least squares solve
        ata = A.T @ A
        aty = A.T @ self.y
        self.coef = np.linalg.lstsq(ata, aty, rcond=None)[0]

        # Predictions and diagnostics
        y_pred = A @ self.coef
        residuals = self.y - y_pred

        # Total sum of squares
        ss_tot = np.sum((self.y - self.y.mean()) ** 2)

        # Compute R², override if target is constant
        if ss_tot < EPS:  # target is (nearly) constant
            r2 = 1.0
        else:
            r2 = 1 - np.sum(residuals ** 2) / ss_tot

        return {
            KEY_PREDICTED: y_pred,
            KEY_R2: r2,
            KEY_RESIDUALS: residuals
        }
