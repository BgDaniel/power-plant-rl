import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import xarray as xr

from constants import ASSET, POWER, COAL, SIMULATION_PATH, KEY_PREDICTED, SPREAD
from delta_position.min_var_delta.min_var_delta import MinVarDelta


class OpsPlotMinVarDelta:
    """
    Wrapper around a MinVarDelta instance that provides
    dedicated plotting methods:

    1) 3D surfaces and residual heatmaps
    2) Binned scatter plots and residual histograms
    """
    def __init__(self, model: MinVarDelta) -> None:
        """
        Initialize the plotting wrapper.

        Args:
            model (MinVarDelta): A trained MinVarDelta model
                containing feature and prediction data.
        """
        self.model: MinVarDelta = model

    def plot(self, y_true: np.ndarray, y_pred: xr.DataArray) -> None:
        """
        Master plotting method.

        Calls the surface plotting and binned/residual plotting methods.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (xr.DataArray): Predicted target values as an xarray.DataArray.

        Returns:
            None
        """
        self._plot_surfaces(y_true, y_pred)
        self._plot_binned_and_residuals(y_true, y_pred)

    def _plot_surfaces(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
         Plot 3D scatter of predicted vs true values, residual heatmaps,
         and histograms of true vs predicted distributions.

         Args:
             y_true (np.ndarray): Array of true target values.
             y_pred (np.ndarray): Array of predicted target values.

         Returns:
             None
         """
        feature_combos = [
            (self.model.x[POWER], self.model.x[COAL], POWER, COAL),
            (self.model.x[POWER], self.model.x[SPREAD], POWER, SPREAD),
            (self.model.x[COAL], self.model.x[SPREAD], COAL, SPREAD),
        ]

        for x1_vals, x2_vals, xlabel, ylabel in feature_combos:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

            # --- 3D scatter ---
            ax3d = fig.add_subplot(1, 3, 1, projection="3d")
            ax3d.scatter(x1_vals, x2_vals, y_true, c="r", s=20, alpha=0.9, marker="o", label="True y")
            ax3d.scatter(x1_vals, x2_vals, y_pred, c="b", s=20, alpha=0.6, marker="x", label="Pred y")
            ax3d.set_xlabel(xlabel)
            ax3d.set_ylabel(ylabel)
            ax3d.set_zlabel("y")
            ax3d.set_title(f"Predicted vs True y over {xlabel} & {ylabel}")
            ax3d.legend()

            # --- Residual heatmap ---
            xi = np.linspace(x1_vals.min(), x1_vals.max(), 50)
            yi = np.linspace(x2_vals.min(), x2_vals.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            residuals_abs = np.abs(y_true - y_pred)
            residual_grid = griddata((x1_vals, x2_vals), residuals_abs, (Xi, Yi), method="linear")

            im = axes[1].imshow(
                residual_grid,
                extent=(x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()),
                origin="lower",
                aspect="auto",
                cmap=cm.viridis,
            )
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)
            axes[1].set_title("|Residuals|")
            fig.colorbar(im, ax=axes[1], shrink=0.8)

            # --- Histograms of y_true vs y_pred ---
            axes[2].hist(y_true, bins=30, alpha=0.5, label="y_true", color="r")
            axes[2].hist(y_pred, bins=30, alpha=0.5, label="y_pred", color="b")
            axes[2].set_title("Distribution of y")
            axes[2].set_xlabel("y")
            axes[2].set_ylabel("Frequency")
            axes[2].legend()

            print(f"[{xlabel}-{ylabel}] y_true range: {y_true.min():.1f}–{y_true.max():.1f}")
            print(f"[{xlabel}-{ylabel}] y_pred range: {y_pred.min():.1f}–{y_pred.max():.1f}")

            plt.show(block=True)

    def _plot_binned_and_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot binned scatter plots along each feature and residual histograms.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.

        Returns:
            None
        """
        feature_combos = [
            (self.model.x[POWER], self.model.x[COAL], POWER, COAL),
            (self.model.x[POWER], self.model.x[SPREAD], POWER, SPREAD),
            (self.model.x[COAL], self.model.x[SPREAD], COAL, SPREAD),
        ]

        for x1_vals, x2_vals, xlabel, ylabel in feature_combos:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

            # --- Binned along X1 ---
            bins_x1 = np.quantile(x1_vals, np.linspace(0, 1, 101))
            colors = plt.cm.tab10(np.linspace(0, 1, 100))
            for i in [0, 25, 50, 75, 99]:
                mask = (x1_vals >= bins_x1[i]) & (x1_vals < bins_x1[i + 1])
                if np.any(mask):
                    axes[0].scatter(x2_vals[mask], y_true[mask], color=colors[i], s=20, alpha=0.8)
            axes[0].set_xlabel(ylabel)
            axes[0].set_ylabel("y")
            axes[0].set_title(f"Binned along {xlabel} vs {ylabel}")

            # --- Binned along X2 ---
            bins_x2 = np.quantile(x2_vals, np.linspace(0, 1, 101))
            for i in [0, 25, 50, 75, 99]:
                mask = (x2_vals >= bins_x2[i]) & (x2_vals < bins_x2[i + 1])
                if np.any(mask):
                    axes[1].scatter(x1_vals[mask], y_true[mask], color=colors[i], s=20, alpha=0.8)
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel("y")
            axes[1].set_title(f"Binned along {ylabel} vs {xlabel}")

            # --- Residual histogram ---
            residuals = y_true - y_pred
            axes[2].hist(residuals, bins=30, color="purple", alpha=0.7)
            axes[2].set_title("Residual distribution")
            axes[2].set_xlabel("y_true - y_pred")
            axes[2].set_ylabel("Frequency")

            plt.show(block=True)


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    fwd_power0 = 70.0
    fwd_coal0 = 95.0

    # Forward prices (as NumPy arrays)
    x_fwd_power = fwd_power0 * np.random.rand(n_samples)
    x_fwd_coal = fwd_coal0 * np.random.rand(n_samples)
    x_spread = x_fwd_power - 0.9 * x_fwd_coal  # optional extra feature

    # True target with noise
    y_true = (
            0.5 * x_fwd_power ** 2
            + np.sin(x_fwd_coal)
            + 0.3 * x_fwd_power * x_fwd_coal
            + np.random.normal(0, 1, n_samples)
    )

    # Use SIMULATION_PATH constant for the dimension name
    sim_coord = xr.DataArray(np.arange(n_samples), dims=[SIMULATION_PATH], name=SIMULATION_PATH)

    power_fwd = xr.DataArray(
        x_fwd_power,
        dims=[SIMULATION_PATH],
        coords={SIMULATION_PATH: sim_coord},
        name=POWER,
    )

    coal_fwd = xr.DataArray(
        x_fwd_coal,
        dims=[SIMULATION_PATH],
        coords={SIMULATION_PATH: sim_coord},
        name=COAL,
    )

    # Concatenate along ASSET dimension
    fwds = xr.concat(
        [power_fwd, coal_fwd],
        dim=xr.DataArray([POWER, COAL], dims=[ASSET], name=ASSET),
    )

    beta = xr.DataArray(
        np.ones((n_samples, fwds.sizes[ASSET])),
        dims=(SIMULATION_PATH, ASSET),
        coords={SIMULATION_PATH: fwds[SIMULATION_PATH], ASSET: fwds[ASSET]},
    )

    min_var_delta = MinVarDelta(
        fwds=fwds,
        y=y_true,
        beta=beta,
        efficiency=0.95,
    )

    y_pred = min_var_delta.compute()[KEY_PREDICTED]

    OpsPlotMinVarDelta(min_var_delta).plot(y_true=y_true, y_pred=y_pred)
