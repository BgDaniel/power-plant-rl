import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, ClassVar, List

# -----------------------
# Dictionary key constants
# -----------------------
KEY_PREDICTED = "predicted"
KEY_R2 = "r2"
KEY_RESIDUALS = "residuals"
KEY_CONDITION_NUMBER = "condition_number"

EPS = 1e-5


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

    POLY_STANDARD: ClassVar[str] = "standard"
    POLY_LEGENDRE: ClassVar[str] = "legendre"
    POLY_CHEBYSHEV: ClassVar[str] = "chebyshev"
    SUPPORTED_POLYNOMIALS: ClassVar[List[str]] = [
        POLY_STANDARD,
        POLY_LEGENDRE,
        POLY_CHEBYSHEV,
    ]

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        poly_type: str = POLY_STANDARD,
    ) -> None:
        """
        Initialize the polynomial regression.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector(s), shape (n_samples, n_features).
        y : np.ndarray
            Target vector, shape (n_samples,).
        degree : int
            Degree of polynomial to fit.
        poly_type : str
            Type of polynomial basis (standard, legendre, chebyshev).
        """
        if poly_type not in self.SUPPORTED_POLYNOMIALS:
            raise ValueError(f"Unsupported polynomial type: {poly_type}")

        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.degree: int = degree
        self.poly_type: str = poly_type
        self.model: Optional[LinearRegression] = None
        self.x_poly: Optional[np.ndarray] = None

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
        # Create polynomial features
        # -------------------------
        # Check for degenerate case: all features almost constant
        # -------------------------
        x_range = self.x.max(axis=0) - self.x.min(axis=0)
        if np.all(x_range < EPS):
            # All features are (almost) constant -> predict mean of y
            y_pred = np.full_like(self.y, np.mean(self.y))
            residuals = self.y - y_pred
            return {
                KEY_PREDICTED: y_pred,
                KEY_R2: 0.0,
                KEY_RESIDUALS: residuals,
                KEY_CONDITION_NUMBER: np.nan,
            }

        # -------------------------
        # Normal polynomial regression
        # -------------------------
        if self.poly_type == self.POLY_STANDARD:
            poly = PolynomialFeatures(degree=self.degree)
            self.x_poly = poly.fit_transform(self.x)
        elif self.poly_type == self.POLY_LEGENDRE:
            x_scaled = (
                2
                * (self.x - self.x.min(axis=0))
                / (self.x.max(axis=0) - self.x.min(axis=0))
                - 1
            )
            self.x_poly = np.polynomial.legendre.legvander2d(
                x_scaled[:, 0], x_scaled[:, 1], [self.degree, self.degree]
            )
        elif self.poly_type == self.POLY_CHEBYSHEV:
            x_scaled = (
                2
                * (self.x - self.x.min(axis=0))
                / (self.x.max(axis=0) - self.x.min(axis=0))
                - 1
            )
            self.x_poly = np.polynomial.chebyshev.chebvander2d(
                x_scaled[:, 0], x_scaled[:, 1], [self.degree, self.degree]
            )

        # Fit linear regression
        self.model = LinearRegression()
        self.model.fit(self.x_poly, self.y)
        y_pred = self.model.predict(self.x_poly)
        residuals = self.y - y_pred
        cond_number = np.linalg.cond(self.x_poly)

        return {
            KEY_PREDICTED: y_pred,
            KEY_R2: self.model.score(self.x_poly, self.y),
            KEY_RESIDUALS: residuals,
            KEY_CONDITION_NUMBER: cond_number,
        }

    def plot(self, results: Dict[str, np.ndarray]) -> None:
        """
        Plot the actual vs predicted values including R² and condition number.

        Parameters
        ----------
        results : Dict[str, np.ndarray]
            Dictionary returned from `regress()`.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(self.y)), self.y, label="Actual", color="blue", alpha=0.7)
        plt.plot(
            range(len(self.y)), results[KEY_PREDICTED], label="Predicted", color="red"
        )
        plt.title(
            f"Polynomial Regression ({self.poly_type}, degree={self.degree}, "
            f"R²={results[KEY_R2]:.4f}, Cond={results[KEY_CONDITION_NUMBER]:.2e})"
        )
        plt.xlabel("Sample index")
        plt.ylabel("Target")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example with complex sinusoidal target
    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = np.sin(x).ravel() + 0.5 * np.sin(3 * x).ravel()
    y_noisy = y_true + 0.2 * np.random.randn(n_samples)

    # Stack X for 2D input example
    x_stacked = np.hstack([x, x**2])

    # Fit polynomial regression with different polynomial types
    for poly_type in PolynomialRegression.SUPPORTED_POLYNOMIALS:
        print(f"\n--- Polynomial type: {poly_type} ---")
        poly_reg = PolynomialRegression(
            x=x_stacked, y=y_noisy, degree=5, poly_type=poly_type
        )
        results = poly_reg.regress()
        print(f"R² score: {results[KEY_R2]:.4f}")
        print(f"Condition number of design matrix: {results[KEY_CONDITION_NUMBER]:.2e}")
        poly_reg.plot(results)
