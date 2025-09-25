from typing import Dict
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
from itertools import combinations
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander

from constants import SIMULATION_PATH, ASSET, POWER, COAL, SPREAD, KEY_DELTA_POSITION
from delta_position.delta_position import DeltaPosition
from constants import KEY_DELTA_POSITION, KEY_PREDICTED, KEY_R2, KEY_RESIDUALS


class MinVarDelta(DeltaPosition):
    """
    Minimum-variance polynomial regression for hedge delta estimation.

    This class extends :class:`DeltaPosition` to compute hedge deltas
    using polynomial regression with optional standard, Legendre, or
    Chebyshev basis functions.
    """

    POLY_STANDARD = "standard"
    POLY_LEGENDRE = "legendre"
    POLY_CHEBYSHEV = "chebyshev"

    def __init__(
        self,
        fwds: xr.DataArray,
        y: np.ndarray,
        beta: xr.DataArray,
        efficiency: float,
        degree: int = 3,
        poly_type: str = POLY_LEGENDRE,
        init_c: float = 1.0,
    ) -> None:
        """
        Initialize the MinVarDelta regression model.

        Args:
            fwds (xr.DataArray): Forward prices of underlying assets.
            y (np.ndarray): Target variable array of shape (n_samples,).
            beta (xr.DataArray): Beta exposures for power and coal.
            efficiency (float): Efficiency factor used in spread calculation.
            degree (int, optional): Degree of the polynomial expansion.
                Defaults to 3.
            poly_type (str, optional): Type of polynomial basis to use.
                One of {"standard", "legendre", "chebyshev"}.
                Defaults to "standard".
            init_c (float, optional): Initial constant coefficient.
                Defaults to 1.0.
        """
        super().__init__(fwds, y, beta)

        x_spread = self.x_fwd_power - efficiency * self.x_fwd_coal
        self.x: Dict[str, np.ndarray] = {
            POWER: self.x_fwd_power,
            COAL: self.x_fwd_coal,
            SPREAD: x_spread,
        }

        self.degree: int = degree
        self.poly_type: str = poly_type
        self.init_c: float = init_c

    def _scale_features(self) -> None:
        """
        Scale input features to the range [-1, 1].

        Returns:
            None
        """
        def scale(x: np.ndarray) -> np.ndarray:
            return 2 * (x - x.min()) / (x.max() - x.min()) - 1.0

        self.x_scaled: Dict[str, np.ndarray] = {
            feature: scale(x) for feature, x in self.x.items()
        }

    def _poly_single(self, feature: str, values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate polynomial features for a single input feature.

        Args:
            feature (str): Feature name.
            values (np.ndarray): Scaled feature values.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping feature names
            to their polynomial expansions.
        """
        features: Dict[str, np.ndarray] = {}
        if self.poly_type == self.POLY_STANDARD:
            for d in range(1, self.degree + 1):
                features[f"{feature}^{d}"] = values ** d
        elif self.poly_type == self.POLY_LEGENDRE:
            full = legvander(values, self.degree)
            for d in range(1, self.degree + 1):
                features[f"{feature}_Leg^{d}"] = full[:, d]
        elif self.poly_type == self.POLY_CHEBYSHEV:
            full = chebvander(values, self.degree)
            for d in range(1, self.degree + 1):
                features[f"{feature}_Che^{d}"] = full[:, d]
        else:
            raise ValueError(f"Unknown poly_type: {self.poly_type}")
        return features

    def _create_polynomial_features(self) -> None:
        """
        Create polynomial features for all input variables.

        Returns:
            None
        """
        self.features_poly: Dict[str, np.ndarray] = {}
        for feature_name, x_scaled in self.x_scaled.items():
            self.features_poly.update(self._poly_single(feature_name, x_scaled))

    def _build_design_matrix(self) -> np.ndarray:
        """
        Build the full design matrix including constant, polynomial,
        and pairwise interaction terms.

        Returns:
            np.ndarray: Design matrix of shape (n_samples, n_features).
        """
        design_blocks = [np.ones((self.n_samples, 1))]
        self.feature_names = ["CONST"]

        # Individual features
        single_features = list(self.features_poly.items())
        design_blocks.append(np.column_stack([col for _, col in single_features]))
        self.feature_names.extend([name for name, _ in single_features])

        # Pairwise interactions
        pair_cols, pair_names = [], []
        keys = [name for name, _ in single_features]
        for i, j in combinations(range(len(keys)), 2):
            name_i, name_j = keys[i], keys[j]
            pair_cols.append(self.features_poly[name_i] * self.features_poly[name_j])
            pair_names.append(f"{name_i}*{name_j}")
        if pair_cols:
            design_blocks.append(np.column_stack(pair_cols))
            self.feature_names.extend(pair_names)

        return np.column_stack(design_blocks)

    def compute(self) -> Dict[str, xr.DataArray]:
        """
        Fit the polynomial regression model and compute hedge deltas.

        Returns:
            Dict[str, xr.DataArray]: A dictionary with keys:
                - KEY_PREDICTED: Predicted deltas as an xarray.DataArray.
                - KEY_RESIDUALS: Residuals of the regression.
                - KEY_R2: R² score of the fitted model.
        """
        self._scale_features()
        self._create_polynomial_features()
        design_matrix = self._build_design_matrix()

        beta_power_diag = np.diag(self.beta_power)
        beta_coal_diag = np.diag(self.beta_coal)

        design_matrix_power = beta_power_diag @ design_matrix
        design_matrix_coal = beta_coal_diag @ design_matrix
        design_matrix_combined = np.hstack([design_matrix_power, design_matrix_coal])

        coef_opt = np.linalg.pinv(design_matrix_combined) @ self.y
        n_feat = design_matrix.shape[1]
        coef_power = coef_opt[:n_feat]
        coef_coal = coef_opt[n_feat:]

        y_pred_power = design_matrix @ coef_power
        y_pred_coal = design_matrix @ coef_coal
        y_pred = self.beta_power * y_pred_power + self.beta_coal * y_pred_coal

        residuals: np.ndarray = self.y - y_pred
        r2: float = r2_score(self.y, y_pred)

        delta_positions: xr.DataArray = xr.DataArray(
            np.stack([y_pred_power, y_pred_coal], axis=1),
            dims=[SIMULATION_PATH, ASSET],
            coords={ASSET: [POWER, COAL], SIMULATION_PATH: np.arange(self.n_samples)},
            name=KEY_DELTA_POSITION,
        )

        return {KEY_DELTA_POSITION: delta_positions,KEY_PREDICTED:  y_pred, KEY_RESIDUALS: residuals, KEY_R2: r2}
