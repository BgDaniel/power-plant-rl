import numpy as np
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from typing import ClassVar, List, Literal, Optional


class PolynomialBasisBuilder:
    """
    Utility class to generate polynomial design matrices for different bases,
    with optional PCA pre-processing to reduce dimensionality.

    Supports
    --------
    - 'standard'  : Ordinary multivariate power polynomials (scikit-learn).
    - 'legendre'  : Multivariate Legendre orthogonal polynomials on [-1, 1].
    - 'chebyshev' : Multivariate Chebyshev orthogonal polynomials on [-1, 1].

    Parameters
    ----------
    degree : int
        Maximum degree of polynomial expansion.
    poly_type : {'standard', 'legendre', 'chebyshev'}, default='standard'
        Type of polynomial basis to use.
    n_components : int, optional
        Number of most relevant features to keep via PCA.
        If None, all original features are used.
    """

    POLY_STANDARD: ClassVar[str] = "standard"
    POLY_LEGENDRE: ClassVar[str] = "legendre"
    POLY_CHEBYSHEV: ClassVar[str] = "chebyshev"
    SUPPORTED: ClassVar[List[str]] = [POLY_STANDARD, POLY_LEGENDRE, POLY_CHEBYSHEV]

    def __init__(
        self,
        degree: int,
        poly_type: Literal["standard", "legendre", "chebyshev"] = "standard",
        n_components: Optional[int] = 6,  # <= NEW: PCA components
    ) -> None:
        """
        Initialize the polynomial basis builder.

        Parameters
        ----------
        degree : int
            Maximum degree of polynomial expansion.
        poly_type : {'standard', 'legendre', 'chebyshev'}, default='standard'
            Type of polynomial basis to use.
        n_components : int, optional
            Number of most relevant features to keep via PCA. If None, all features are used.
        """
        if poly_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported polynomial type: {poly_type}")
        self.degree = degree
        self.poly_type = poly_type
        self.n_components = n_components

    # -------------------------
    # Core utilities
    # -------------------------
    @staticmethod
    def _scale_to_unit(x: np.ndarray) -> np.ndarray:
        """
        Scale features column-wise to [-1, 1] for orthogonal polynomials.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Scaled feature matrix with values in [-1, 1].
        """
        """Scale features column-wise to [-1, 1] for orthogonal polynomials."""
        return 2 * (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) - 1

    @staticmethod
    def _multiindex(n_features: int, degree: int):
        """
        Yield all exponent tuples with total degree <= degree.

        Parameters
        ----------
        n_features : int
            Number of input features.
        degree : int
            Maximum polynomial degree.

        Yields
        ------
        Tuple[int, ...]
            Exponent tuple for each feature.
        """
        for powers in product(range(degree + 1), repeat=n_features):
            if sum(powers) <= degree:
                yield powers

    def _select_relevant_features(self, x: np.ndarray) -> np.ndarray:
        """
        Select the most relevant original features based on PCA.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Original feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Subset of x containing the most relevant original features.
        """
        n_samples, n_features = x.shape
        if self.n_components >= n_features:
            return x  # no selection needed

        # Fit PCA
        pca = PCA(n_components=n_features)
        pca.fit(x)

        # Compute feature importance as sum of squared loadings across top n_components
        feature_importance = np.sum(pca.components_[:self.n_components] ** 2, axis=0)

        # Select indices of top n_components most important features
        top_indices = np.argsort(feature_importance)[-self.n_components:]

        # Return only the most relevant features
        return x[:, top_indices]

    # -------------------------
    # Main builder
    # -------------------------
    def build(self, x: np.ndarray) -> np.ndarray:
        """
        Build the polynomial design matrix for input x.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Design matrix with polynomial basis.
        """
        # ----- Optional PCA -----
        if self.n_components is not None:
            x = self._select_relevant_features(x)

        n_samples, n_features = x.shape

        # ----- Polynomial expansion -----
        if self.poly_type == self.POLY_STANDARD:
            return PolynomialFeatures(degree=self.degree).fit_transform(x)

        # Orthogonal (Legendre/Chebyshev)
        x_scaled = self._scale_to_unit(x)
        basis_cols = []
        for powers in self._multiindex(n_features, self.degree):
            col = np.ones(n_samples)
            for j, p in enumerate(powers):
                if self.poly_type == self.POLY_LEGENDRE:
                    col *= np.polynomial.legendre.Legendre.basis(p)(x_scaled[:, j])
                else:
                    col *= np.polynomial.chebyshev.Chebyshev.basis(p)(x_scaled[:, j])
            basis_cols.append(col)
        return np.column_stack(basis_cols)
