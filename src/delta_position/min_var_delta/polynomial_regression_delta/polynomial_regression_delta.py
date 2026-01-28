from typing import Dict, Tuple, Optional
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
from itertools import combinations
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander

from delta_position.delta_position_result import DeltaPositionResult
from regression.regression_helpers import EPS

from constants import SIMULATION_PATH, ASSET, POWER, COAL, SPREAD
from delta_position.delta_base import DeltaBase
from constants import KEY_DELTA_POSITION, KEY_PREDICTED, KEY_R2, KEY_RESIDUALS


class PolynomialRegressionDelta(DeltaBase):
    """
    Minimum-variance polynomial_regression_delta regression for hedge polynomial_regression_delta estimation.

    This class extends :class:`DeltaBase` to compute hedge deltas
    using polynomial_regression_delta regression with optional standard, Legendre, or
    Chebyshev basis functions.
    """

    POLY_STANDARD = "standard"
    POLY_LEGENDRE = "legendre"
    POLY_CHEBYSHEV = "chebyshev"

    def __init__(
        self,
        degree: int = 3,
        poly_type: str = POLY_LEGENDRE,
        init_c: float = 1.0,
    ) -> None:
        """
        Initialize the MinVarDelta regression model.

        Args:
            degree (int, optional): Degree of the polynomial_regression_delta expansion.
                Defaults to 3.
            poly_type (str, optional): Type of polynomial_regression_delta basis to use.
                One of {"standard", "legendre", "chebyshev"}.
                Defaults to "standard".
            init_c (float, optional): Initial constant coefficient.
                Defaults to 1.0.
        """
        self.degree: int = degree
        self.poly_type: str = poly_type
        self.init_c: float = init_c

    @staticmethod
    def _scale(x: np.ndarray) -> np.ndarray:
        """
        Scale a vector to the interval [-1, 1].

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Scaled array.
        """
        return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0

    def _poly_single(
        self,
        feature: str,
        values: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate polynomial features for a single input variable.

        Parameters
        ----------
        feature : str
            Feature name.
        values : np.ndarray
            Scaled feature values.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from feature names to polynomial feature arrays.
        """
        features: Dict[str, np.ndarray] = {}

        if self.poly_type == self.POLY_STANDARD:
            for d in range(1, self.degree + 1):
                features[f"{feature}^{d}"] = values**d

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

    def _build_design_matrix(
        self,
        x_scaled: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Construct the full regression design matrix.

        Includes:
        - constant term
        - polynomial terms
        - pairwise interaction terms

        Parameters
        ----------
        x_scaled : Dict[str, np.ndarray]
            Scaled input features.

        Returns
        -------
        np.ndarray
            Design matrix of shape (n_samples, n_features).
        """
        features_poly: Dict[str, np.ndarray] = {}
        for name, values in x_scaled.items():
            features_poly.update(self._poly_single(name, values))

        design_blocks = [np.ones((len(next(iter(x_scaled.values()))), 1))]
        single_items = list(features_poly.items())

        design_blocks.append(
            np.column_stack([col for _, col in single_items])
        )

        # Pairwise interactions
        pair_cols = []
        keys = [name for name, _ in single_items]
        for i, j in combinations(range(len(keys)), 2):
            pair_cols.append(
                features_poly[keys[i]] * features_poly[keys[j]]
            )

        if pair_cols:
            design_blocks.append(np.column_stack(pair_cols))

        return np.column_stack(design_blocks)

    def _check_degenerate_case(
        self,
        x: Dict[str, np.ndarray],
        y: np.ndarray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
    ) -> Optional[DeltaPositionResult]:
        """
        Handle the degenerate case where all features are nearly constant.

        Parameters
        ----------
        x : Dict[str, np.ndarray]
            Raw input features.
        y : np.ndarray
            Target variable.
        beta_power : np.ndarray
            Power beta values.
        beta_coal : np.ndarray
            Coal beta values.

        Returns
        -------
        Optional[DeltaPositionResult]
            Degenerate solution if applicable, otherwise None.
        """
        for arr in x.values():
            if np.ptp(arr) >= EPS:
                return None

        beta_mat = np.column_stack([beta_power, beta_coal])

        try:
            delta_vals, *_ = np.linalg.lstsq(beta_mat, y, rcond=None)
        except np.linalg.LinAlgError:
            delta_vals = np.zeros(2)

        y_pred = beta_mat @ delta_vals
        residuals = y - y_pred

        delta_positions = xr.DataArray(
            np.tile(delta_vals, (len(y), 1)),
            dims=[SIMULATION_PATH, ASSET],
            coords={
                ASSET: [POWER, COAL],
                SIMULATION_PATH: np.arange(len(y)),
            },
            name=KEY_DELTA_POSITION,
        )

        return DeltaPositionResult(
            delta_positions=delta_positions,
            predicted=y_pred,
            residuals=residuals,
            r_squared=float("nan"),
        )

    def delta(
        self,
        fwds: xr.DataArray,
        y: np.ndarray,
        beta: xr.DataArray,
        efficiency: float,
    ) -> DeltaPositionResult:
        """
        Compute hedge deltas using polynomial regression.

        Parameters
        ----------
        fwds : xr.DataArray
            Forward prices with dimension ASSET.
        y : np.ndarray
            Target variable of shape (n_samples,).
        beta : xr.DataArray
            Asset betas with dimension ASSET.
        efficiency : float
            Efficiency factor used to compute the spread.

        Returns
        -------
        DeltaPositionResult
            Regression result including:
            - delta positions
            - predicted values
            - residuals
            - R² score
        """
        # -----------------------------
        # Step 0: Extract raw input arrays from xarray
        # -----------------------------
        x_fwd_power = fwds.sel({ASSET: POWER}).values  # power forward curve
        x_fwd_coal = fwds.sel({ASSET: COAL}).values  # coal forward curve

        beta_power = beta.sel({ASSET: POWER}).values  # power beta
        beta_coal = beta.sel({ASSET: COAL}).values  # coal beta

        # -----------------------------
        # Step 1: Compute spread and build feature dictionary
        # -----------------------------
        x = {
            POWER: x_fwd_power,
            COAL: x_fwd_coal,
            SPREAD: x_fwd_power - efficiency * x_fwd_coal,  # efficiency-adjusted spread
        }

        # -----------------------------
        # Step 2: Handle degenerate case
        # If all features are nearly constant, return a simple linear solution
        # -----------------------------
        degenerate = self._check_degenerate_case(
            x, y, beta_power, beta_coal
        )
        if degenerate is not None:
            return degenerate

        # -----------------------------
        # Step 3: Scale features to [-1, 1] and create polynomial features
        # -----------------------------
        x_scaled = {k: self._scale(v) for k, v in x.items()}
        design_matrix = self._build_design_matrix(x_scaled)  # polynomial + interactions

        # -----------------------------
        # Step 4: Apply beta weights using diagonal matrices
        # -----------------------------
        dm_power = np.diag(beta_power) @ design_matrix
        dm_coal = np.diag(beta_coal) @ design_matrix

        # -----------------------------
        # Step 5: Combine matrices for extended regression
        # -----------------------------
        dm_combined = np.hstack([dm_power, dm_coal])

        # -----------------------------
        # Step 6: Solve for optimal regression coefficients using pseudo-inverse
        # -----------------------------
        coef = np.linalg.pinv(dm_combined) @ y
        n_feat = design_matrix.shape[1]

        coef_power = coef[:n_feat]  # coefficients for power
        coef_coal = coef[n_feat:]  # coefficients for coal

        # -----------------------------
        # Step 7: Compute predicted values for each asset and combined
        # -----------------------------
        y_pred_power = design_matrix @ coef_power
        y_pred_coal = design_matrix @ coef_coal
        y_pred = beta_power * y_pred_power + beta_coal * y_pred_coal  # weighted sum

        # -----------------------------
        # Step 8: Compute residuals and R² score
        # -----------------------------
        residuals = y - y_pred
        r2 = r2_score(y, y_pred)

        # -----------------------------
        # Step 9: Construct delta positions as an xarray DataArray
        # -----------------------------
        delta_positions = xr.DataArray(
            np.stack([y_pred_power, y_pred_coal], axis=1),
            dims=[SIMULATION_PATH, ASSET],
            coords={
                ASSET: [POWER, COAL],
                SIMULATION_PATH: np.arange(len(y)),
            },
            name=KEY_DELTA_POSITION,
        )

        # -----------------------------
        # Step 10: Return results in a DeltaPositionResult object
        # -----------------------------
        return DeltaPositionResult(
            delta_position=delta_positions,
            predicted=y_pred,
            residuals=residuals,
            r2=r2,
        )