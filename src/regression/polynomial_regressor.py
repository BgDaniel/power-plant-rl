import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, ClassVar, List


from regression.polynomial_base_builder import PolynomialBasisBuilder
from regression.regression_helpers import KEY_PREDICTED, KEY_R2, KEY_RESIDUALS, check_degenerate_case, EPS


class PolynomialRegression:
    """
    Class for performing polynomial regression with different polynomial bases.

    Attributes
    ----------
    POLY_STANDARD : ClassVar[str]
        Standard polynomial basis.
    POLY_LEGENDRE : ClassVar[str]
        Legendre polynomial basis.
    POLY_CHEBYSHEV : ClassVar[str]
        Chebyshev polynomial basis.
    SUPPORTED_POLYNOMIALS : ClassVar[List[str]]
        List of supported polynomial types.
    x : np.ndarray
        Input feature vector(s), shape (n_samples, n_features).
    y : np.ndarray
        Target vector, shape (n_samples,).
    degree : int
        Degree of the polynomial to be used.
    poly_type : str
        Type of polynomial basis to use.
    model : Optional[LinearRegression]
        The fitted scikit-learn LinearRegression model.
    x_poly : Optional[np.ndarray]
        The expanded feature matrix after applying the polynomial transformation.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        poly_type: str = PolynomialBasisBuilder.POLY_STANDARD,
        n_components: Optional[int] = 4,  # NEW: PCA components
    ) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y

        self.degree: int = degree
        self.poly_type: str = poly_type
        self.n_components: Optional[int] = n_components

        self._builder = PolynomialBasisBuilder(degree, poly_type)

    def regress(self) -> Dict[str, np.ndarray]:
        """
        Perform polynomial regression using the specified polynomial type.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - predicted: Predicted values from the regression.
            - r2: R² score.
            - residuals: Residuals of the regression.
            - condition_number: Condition number of the design matrix.
        """
        # -------------------------
        # Check for degenerate case: all features almost constant
        # -------------------------
        check_degenerate_case(self.x, self.y)

        # -------------------------
        # Normal polynomial regression
        # -------------------------
        self.x_poly = self._builder.build(self.x)

        # Fit linear regression
        self.model = LinearRegression()
        self.model.fit(self.x_poly, self.y)
        y_pred = self.model.predict(self.x_poly)
        residuals = self.y - y_pred

        # Compute R², override if target is constant
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = self.model.score(self.x_poly, self.y)

        if ss_tot < EPS:  # target is constant
            r2 = 1.0

        return {
            KEY_PREDICTED: y_pred,
            KEY_R2: r2,
            KEY_RESIDUALS: residuals
        }
