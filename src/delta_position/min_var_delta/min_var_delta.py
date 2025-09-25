from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
from itertools import combinations
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm

from constants import SIMULATION_PATH, ASSET, POWER, COAL, SPREAD
from delta_position.delta_position import DeltaPosition

from constants import DELTA_POSITION, KEY_PREDICTED, KEY_R2, KEY_RESIDUALS



class MinVarDelta(DeltaPosition):
    """
    Minimum-variance polynomial regression for hedge delta estimation.
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
        poly_type: str = POLY_STANDARD,
        init_c: float = 1.0,
    ) -> None:

        super().__init__(fwds, y, beta)


        x_spread= self.x_fwd_power - efficiency*self.x_fwd_coal

        self.x = {POWER: self.x_fwd_power, COAL: self.x_fwd_coal, SPREAD: x_spread}

        self.degree = degree
        self.poly_type = poly_type
        self.init_c = init_c



    # ------------------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------------------
    def _scale_features(self) -> None:
        def scale(x):
            return 2 * (x - x.min()) / (x.max() - x.min()) - 1.0
        self.x_scaled = {feature: scale(x) for feature, x in self.x.items()}

    def _poly_single(self, feature: str, values: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
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
        self.features_poly = {}
        for feature_name, x_scaled in self.x_scaled.items():
            self.features_poly.update(self._poly_single(feature_name, x_scaled))

    def _build_design_matrix(self) -> np.ndarray:
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

    def _plot_fitted_surfaces(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot fitted surfaces, residual heatmaps, histograms, and binned regression plots.
        """
        feature_combos = [
            (self.x["POWER"], self.x["COAL"], "POWER", "COAL"),
            (self.x["POWER"], self.x["SPREAD"], "POWER", "SPREAD"),
            (self.x["COAL"], self.x["SPREAD"], "COAL", "SPREAD")
        ]

        n_bins = 10  # 10% quantiles

        for x1_vals, x2_vals, xlabel, ylabel in feature_combos:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

            # -----------------------
            # Row 1 Left: 3D scatter
            # -----------------------
            ax3d = fig.add_subplot(2, 3, 1, projection='3d')
            # Plot true y as circles first
            ax3d.scatter(x1_vals, x2_vals, y_true, c='r', s=20, alpha=0.9, marker='o', label="True y")
            # Plot predicted y as crosses on top
            ax3d.scatter(x1_vals, x2_vals, y_pred, c='b', s=20, alpha=0.6, marker='x', label="Predicted y")
            ax3d.set_xlabel(xlabel)
            ax3d.set_ylabel(ylabel)
            ax3d.set_zlabel("y")
            ax3d.set_title(f"Predicted vs True y over {xlabel} and {ylabel}")
            ax3d.legend()

            # -----------------------
            # Row 1 Middle: Residual heatmap
            # -----------------------
            xi = np.linspace(x1_vals.min(), x1_vals.max(), 50)
            yi = np.linspace(x2_vals.min(), x2_vals.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            residuals_abs = np.abs(y_true - y_pred)
            residual_grid = griddata((x1_vals, x2_vals), residuals_abs, (Xi, Yi), method='linear')

            im = axes[0, 1].imshow(residual_grid, extent=(x1_vals.min(), x1_vals.max(),
                                                          x2_vals.min(), x2_vals.max()),
                                   origin='lower', aspect='auto', cmap=cm.viridis)
            axes[0, 1].set_xlabel(xlabel)
            axes[0, 1].set_ylabel(ylabel)
            axes[0, 1].set_title("Residuals |y_true - y_pred|")
            fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

            # -----------------------
            # Row 1 Right: overlapping histograms of y_true and y_pred
            # -----------------------
            axes[0, 2].hist(y_true, bins=30, alpha=0.5, label='y_true', color='r')
            axes[0, 2].hist(y_pred, bins=30, alpha=0.5, label='y_pred', color='b')
            axes[0, 2].set_title("Distribution: y_true vs y_pred")
            axes[0, 2].set_xlabel("y")
            axes[0, 2].set_ylabel("Frequency")
            axes[0, 2].legend()

            print(f"y_true min: {y_true.min():.1f}, max: {y_true.max():.1f}")
            print(f"y_pred min: {y_pred.min():.1f}, max: {y_pred.max():.1f}")

            # -----------------------
            # Row 2 Left: Binned along X1 but plotted against X2
            # -----------------------
            n_bins = 100
            bins_x1 = np.quantile(x1_vals, np.linspace(0, 1, n_bins + 1))
            colors = plt.cm.tab10(np.linspace(0, 1, n_bins))

            for i in [0, 25, 50, 75, 99]:
                mask = (x1_vals >= bins_x1[i]) & (x1_vals < bins_x1[i + 1])
                if np.any(mask):
                    base_color = colors[i]
                    lighter_color = list(base_color[:3]) + [0.6]
                    axes[1, 0].scatter(x2_vals[mask], y_true[mask], color=base_color, s=20, alpha=0.8)
                    sorted_idx = np.argsort(x2_vals[mask])
                    # Optional: plot predicted lines if desired
                    # axes[1, 0].plot(x2_vals[mask][sorted_idx], y_pred[mask][sorted_idx],
                    #                color=lighter_color, lw=2)
            axes[1, 0].set_xlabel(ylabel)
            axes[1, 0].set_ylabel("y")
            axes[1, 0].set_title(f"Binned along {xlabel}, plotted vs {ylabel}")

            # -----------------------
            # Row 2 Middle: Binned along X2 but plotted against X1
            # -----------------------
            bins_x2 = np.quantile(x2_vals, np.linspace(0, 1, n_bins + 1))
            for i in [0, 25, 50, 75, 99]:
                mask = (x2_vals >= bins_x2[i]) & (x2_vals < bins_x2[i + 1])
                if np.any(mask):
                    base_color = colors[i]
                    lighter_color = list(base_color[:3]) + [0.6]
                    axes[1, 1].scatter(x1_vals[mask], y_true[mask], color=base_color, s=20, alpha=0.8)
                    sorted_idx = np.argsort(x1_vals[mask])
                    # Optional: plot predicted lines if desired
                    # axes[1, 1].plot(x1_vals[mask][sorted_idx], y_pred[mask][sorted_idx],
                    #                color=lighter_color, lw=2)
            axes[1, 1].set_xlabel(xlabel)
            axes[1, 1].set_ylabel("y")
            axes[1, 1].set_title(f"Binned along {ylabel}, plotted vs {xlabel}")

            # -----------------------
            # Row 2 Right: histogram of residuals
            # -----------------------
            residuals = y_true - y_pred
            axes[1, 2].hist(residuals, bins=30, color='purple', alpha=0.7)
            axes[1, 2].set_title("Residual distribution")
            axes[1, 2].set_xlabel("y_true - y_pred")
            axes[1, 2].set_ylabel("Frequency")

            plt.show(block=True)

    # ------------------------------------------------------------------
    # Core regression logic
    # ------------------------------------------------------------------
    def compute(self) -> xr.DataArray:
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

        residuals = self.y - y_pred
        r2 = r2_score(self.y, y_pred)

        y_pred_combined = np.stack([y_pred_power, y_pred_coal], axis=1)
        y_pred_xr = xr.DataArray(
            y_pred_combined,
            dims=[SIMULATION_PATH, ASSET],
            coords={ASSET: [POWER, COAL], SIMULATION_PATH: np.arange(self.n_samples)},
            name=DELTA_POSITION
        )

        return {KEY_PREDICTED: y_pred_xr, KEY_RESIDUALS: residuals, KEY_R2: r2}


# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 500
    fwd_power0 = 70.0
    fwd_coal0 = 95.0

    x_fwd_power = fwd_power0 * np.random.rand(n_samples)
    x_fwd_coal = fwd_coal0 * np.random.rand(n_samples)
    x_spread = x_fwd_power - 0.9 * x_fwd_coal

    y = 0.5 * x_fwd_power ** 2 + np.sin(x_fwd_coal) + 0.3 * x_fwd_power * x_fwd_coal + np.random.normal(0, 1, n_samples)

    # ---------------------------
    # Construct xarray DataArrays
    # ---------------------------
    fwds = xr.DataArray(
        np.stack([x_fwd_power, x_fwd_coal], axis=1),
        dims=[SIMULATION_PATH, ASSET],
        coords={SIMULATION_PATH: np.arange(n_samples), ASSET: [POWER, COAL]},
    )

    beta = xr.DataArray(
        np.ones((n_samples, 2)),
        dims=[SIMULATION_PATH, ASSET],  # Only POWER and COAL
        coords={SIMULATION_PATH: np.arange(n_samples), ASSET: [POWER, COAL]},
    )

    # ---------------------------
    # Initialize MinVarDelta
    # ---------------------------
    min_var_delta = MinVarDelta(
        fwds=fwds,
        y=y,
        beta=beta,
        efficiency=0.9,
        degree=2,
        poly_type=MinVarDelta.POLY_STANDARD,
        init_c=1.0
    )

    # ---------------------------
    # Compute delta positions
    # ---------------------------
    result = min_var_delta.compute()
    print("R² score:", result[KEY_R2])
    print("Predicted shape:", result[KEY_PREDICTED].shape)