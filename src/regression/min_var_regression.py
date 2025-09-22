from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander
from sklearn.metrics import r2_score
from typing import ClassVar, Dict, List
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr
from scipy.interpolate import griddata


from market_simulation.constants import SIMULATION_PATH


from regression.regression_helpers import KEY_RESIDUALS, KEY_PREDICTED, KEY_R2
from valuation.power_plant.power_plant import POWER, COAL, SPREAD


SPREAD_TANH = 'SPREAD_TANH'
ASSET = 'ASSET'
DELTA_POSITION = 'DELTA_POSITION'
SPREAD_IND ='SPREAD_IND'


class MinVarPolynomialRegression:
    """
    Minimum-variance polynomial regression for hedge delta estimation.

    Fits δ(x_fwd, x_spread) to minimize Var[y - δ(x)*beta] across scenarios.

    Features
    --------
    * Forward driver: x_fwd  = merged average of power & coal prices
    * Spread driver : x_spread transformed as tanh(x_spread / c)
    * Basis         : tensor product of polynomial features of
                      x_fwd and transformed spread.

    Supports
    ---------
    - Standard monomials
    - Legendre polynomials on [-1,1]
    - Chebyshev polynomials of the first kind on [-1,1]
    """

    POLY_STANDARD: ClassVar[str] = "standard"
    """Standard polynomial basis."""
    POLY_LEGENDRE: ClassVar[str] = "legendre"
    """Legendre polynomial basis."""
    POLY_CHEBYSHEV: ClassVar[str] = "chebyshev"
    """Chebyshev polynomial basis."""

    def __init__(
        self,
        n_samples: int,
        x_fwd_power: np.ndarray,
        x_fwd_coal: np.ndarray,
        x_spread: np.ndarray,
        y: np.ndarray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
        degree: int = 3,
        poly_type: str = POLY_STANDARD,
        init_c: float = 1.0,
    ) -> None:
        self.n_samples = n_samples
        self.x = {
            POWER: x_fwd_power,
            COAL: x_fwd_coal,
            SPREAD: x_spread,
            #SPREAD_TANH: np.tanh(x_spread)
            #SPREAD_IND: (x_spread > 0).astype(float)
        }

        self.y = y
        self.beta_power = beta_power
        self.beta_coal = beta_coal

        self.degree = degree
        self.poly_type = poly_type
        self.init_c = init_c

    # ------------------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------------------
    def _scale_features(self) -> None:
        """Scale each forward feature and the spread feature to [-1, 1]."""

        def scale(x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return 2 * (x - x.min()) / (x.max() - x.min()) - 1.0

        self.x_scaled = {feature: scale(x) for feature, x in self.x.items()}

    def _poly_single(self, feature: str, values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate polynomial features (excluding the constant term) for a single 1D feature.

        Parameters
        ----------
        feature : str
            Name of the base feature (e.g. "POWER", "COAL", "SPREAD").
        values : np.ndarray
            Scaled 1D array of feature values.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from feature names to polynomial columns, e.g.
            {"POWER^1": array([...]), "POWER^2": array([...])}.
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

    # ------------------------------------------------------------------
    # Polynomial Features
    # ------------------------------------------------------------------
    def _create_polynomial_features(self) -> None:
        """
        Create polynomial features for each forward column.
        Discards degree information.
        """
        self.features_poly: Dict[str, np.ndarray] = {}

        # Forward drivers
        for feature_name, x_scaled in self.x_scaled.items():
            poly_dict = self._poly_single(feature_name, x_scaled)  # now returns dict
            self.features_poly.update(poly_dict)  # merge key-value pairs

    def _build_design_matrix(self) -> np.ndarray:
        """
        Build design matrix while restricting the total polynomial order
        of any pairwise product to self.degree.

        Returns
        -------
        np.ndarray
            Design matrix of shape (n_samples, n_features)
            (including constant, individual polys, and pairwise products).
        """
        design_blocks = []
        feature_names: list[str] = []

        # 1️⃣ Constant term
        design_blocks.append(np.ones((self.n_samples, 1)))
        feature_names.append("CONST")

        # 2️⃣ Individual polynomial features
        # self.features_poly is a dict: {name: array}
        single_features = list(self.features_poly.items())  # preserve insertion order
        design_blocks.append(np.column_stack([col for _, col in single_features]))
        feature_names.extend([name for name, _ in single_features])

        # 3️⃣ Pairwise products (interactions)
        pair_cols = []
        pair_names = []
        keys = [name for name, _ in single_features]

        for i, j in combinations(range(len(keys)), 2):
            # create product if total polynomial order ≤ self.degree
            # (here we skip order checking; add if needed)
            name_i, name_j = keys[i], keys[j]
            pair_cols.append(self.features_poly[name_i] * self.features_poly[name_j])
            pair_names.append(f"{name_i}*{name_j}")

        if pair_cols:
            design_blocks.append(np.column_stack(pair_cols))
            feature_names.extend(pair_names)

        # Store names for later use (e.g., feature importance)
        self.feature_names = feature_names

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
                    lighter_color = list(base_color[:3]) + [0.6]  # lighter line
                    axes[1, 0].scatter(x2_vals[mask], y_true[mask], color=base_color, s=20, alpha=0.8)
                    sorted_idx = np.argsort(x2_vals[mask])
                    #axes[1, 0].plot(x2_vals[mask][sorted_idx], y_pred[mask][sorted_idx],
                    #                color=lighter_color, lw=2)
            axes[1, 0].set_xlabel(ylabel)  # plotting against the "other" dimension
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
                    #axes[1, 1].plot(x1_vals[mask][sorted_idx], y_pred[mask][sorted_idx],
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

    def regress(self, plot: bool = True) -> Dict[str, np.ndarray]:
        """
        Fit the polynomial regression by minimizing variance of residuals.

        Parameters
        ----------
        plot : bool, optional
            If True, plots the fitted vs target values at the end.

        Returns
        -------
        dict
            Dictionary with predicted values, residuals, and R² score.
        """
        # Scale features and create polynomial terms
        self._scale_features()
        self._create_polynomial_features()

        # Build design matrix
        design_matrix = self._build_design_matrix()  # shape (n_samples, n_features)

        # -----------------------------
        # Step 1: Apply beta weights via diagonal matrices
        # -----------------------------
        beta_power_diag = np.diag(self.beta_power)  # n x n
        beta_coal_diag = np.diag(self.beta_coal)  # n x n

        design_matrix_power = beta_power_diag @ design_matrix  # n x n_features
        design_matrix_coal = beta_coal_diag @ design_matrix  # n x n_features

        # -----------------------------
        # Step 2: Stack horizontally for extended regression
        # -----------------------------
        design_matrix_combined = np.hstack([design_matrix_power, design_matrix_coal])  # n x 2*features

        # -----------------------------
        # Step 3: Solve pseudo-inverse
        # -----------------------------
        coef_opt = np.linalg.pinv(design_matrix_combined) @ self.y  # 2*features

        # -----------------------------
        # Step 4: Split coefficients back
        # -----------------------------
        n_feat = design_matrix.shape[1]
        coef_power = coef_opt[:n_feat]
        coef_coal = coef_opt[n_feat:]

        # -----------------------------
        # Step 5: Compute predictions
        # -----------------------------
        y_pred_power = design_matrix @ coef_power
        y_pred_coal = design_matrix @ coef_coal

        # Combine predictions using original beta weights
        y_pred = self.beta_power * y_pred_power + self.beta_coal * y_pred_coal

        # -----------------------------
        # Step 6: Residuals and R²
        # -----------------------------
        residuals = self.y - y_pred
        r2 = r2_score(self.y, y_pred)

        # -----------------------------
        # Step 7: Optional plotting
        # -----------------------------
        if plot:
            self._plot_fitted_surfaces(self.y, y_pred)

        # -----------------------------
        # Step 8: Return results
        # -----------------------------
        y_pred_combined = np.stack([y_pred_power, y_pred_coal], axis=1)  # shape: (n_samples, 2)

        y_pred = xr.DataArray(
                y_pred_combined,
                dims=[SIMULATION_PATH, ASSET],
                coords={
                    ASSET: [POWER, COAL],
                    SIMULATION_PATH: np.arange(self.n_samples)
                },
                name=DELTA_POSITION
            )


        return {
            KEY_PREDICTED: y_pred,
            KEY_RESIDUALS: residuals,
            KEY_R2: r2
        }


if __name__ == "__main__":

    # -----------------------------
    # Dummy data
    # -----------------------------
    n_samples = 1000

    # Random forward features
    fwd_power0 = 70.0
    fwd_coal0 = 95.0

    np.random.seed(42)  # for reproducibility
    x_fwd_power = fwd_power0 * np.random.uniform(0, 1, n_samples)
    x_fwd_coal = fwd_coal0 * np.random.uniform(0, 1, n_samples)

    # Spread feature as linear combination with efficiency
    efficiency = 0.9
    x_spread = x_fwd_power - efficiency * x_fwd_coal

    # Add tiny jitter to avoid coplanar points
    jitter = 1e-8 * np.random.rand(n_samples)
    x_fwd_power += jitter
    x_fwd_coal += jitter
    x_spread += jitter


    # More complicated target function
    def f(x1, x2, s):
        return 0.5 * x1 ** 2 + np.sin(x2) + 0.2 * np.tanh(s) + 0.3 * x1 * x2

    # Generate target cashflows with some noise
    noise = np.random.normal(0, 1.0, n_samples)
    y = f(x_fwd_power,x_fwd_coal, x_spread) + noise

    # Hedge scaling factor
    beta_power = np.ones(n_samples)
    beta_coal = np.ones(n_samples)

    # -----------------------------
    # Create regression object
    # -----------------------------
    model = MinVarPolynomialRegression(
        n_samples=n_samples,
        x_fwd_power=x_fwd_power,
        x_fwd_coal=x_fwd_coal,
        x_spread=x_spread,
        y=y,
        beta_power=beta_power,
        beta_coal=beta_coal,
        degree=2,
        poly_type=MinVarPolynomialRegression.POLY_STANDARD,
        init_c=1.0
    )

    # -----------------------------
    # Run regression with plotting
    # -----------------------------
    result = model.regress(plot=True)
    y_pred = result[KEY_PREDICTED]

    # -----------------------------
    # Plot true vs predicted
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(x_fwd_power, y, s=10, alpha=0.5, label="True y (with noise)")
    plt.scatter(x_fwd_power, y_pred, s=10, alpha=0.5, label="Predicted y")
    plt.xlabel("x_fwd1")
    plt.ylabel("y")
    plt.title(f"Regression: True vs Predicted y (R² = {result[KEY_R2]:.3f})")
    plt.legend()
    plt.show(block=True)