import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import xarray as xr

from constants import ASSET, POWER, COAL, SIMULATION_PATH, KEY_PREDICTED
from delta_position.min_var_delta.polynomial_delta import PolynomialDelta


class OpsPlotMinVarDelta:
    """
    Wrapper around a MinVarDelta instance that provides
    dedicated plotting methods:

    1) 3D surfaces, residual heatmaps, histograms
    2) Binned scatter plots and residual histograms
    """

    def __init__(self, model: PolynomialDelta) -> None:
        """
        Initialize the plotting wrapper.

        Args:
            model (PolynomialDelta): A trained MinVarDelta model
                containing feature and prediction data.
        """
        self.model: PolynomialDelta = model

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

        Arranged in a 2×2 grid. The bottom row shares the same x-axis
        between the two subplots.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.

        Returns:
            None
        """
        # Only POWER vs COAL is plotted
        x1_vals, x2_vals, xlabel, ylabel = (
            self.model.x[POWER],
            self.model.x[COAL],
            POWER,
            COAL,
        )

        # 2x2 grid – share x between the bottom row
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 8), constrained_layout=True, sharex="col"
        )
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")

        # --- 3D scatter ---
        ax3d.scatter(
            x1_vals,
            x2_vals,
            y_true,
            c="r",
            s=20,
            alpha=0.9,
            marker="o",
            label="True",
        )
        ax3d.scatter(
            x1_vals,
            x2_vals,
            y_pred,
            c="b",
            s=20,
            alpha=0.6,
            marker="x",
            label="Pred",
        )
        ax3d.set_xlabel(xlabel)
        ax3d.set_ylabel(ylabel)
        ax3d.set_title(f"Predicted vs True")
        ax3d.legend()

        # --- Residual heatmap ---
        xi = np.linspace(x1_vals.min(), x1_vals.max(), 50)
        yi = np.linspace(x2_vals.min(), x2_vals.max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        residuals_abs = np.abs(y_true - y_pred)
        residual_grid = griddata(
            (x1_vals, x2_vals), residuals_abs, (Xi, Yi), method="linear"
        )

        im = axes[0, 1].imshow(
            residual_grid,
            extent=(x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()),
            origin="lower",
            aspect="auto",
            cmap=cm.viridis,
        )
        axes[0, 1].set_xlabel(xlabel)
        axes[0, 1].set_ylabel(ylabel)
        axes[0, 1].set_title("Residuals")
        fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

        # --- Histogram of y_true vs y_pred ---
        hist_ax = axes[1, 0]
        hist_ax.hist(y_true, bins=30, alpha=0.5, label="True", color="r")
        hist_ax.hist(y_pred, bins=30, alpha=0.5, label="Pred", color="b")
        hist_ax.set_title("Distribution")

        hist_ax.legend()

        # --- Residual histogram (shares x with hist_ax) ---
        residual_ax = axes[1, 1]
        residuals = y_true - y_pred
        residual_ax.hist(residuals, bins=30, color="purple", alpha=0.7)
        residual_ax.set_title("Residual distribution")
        residual_ax.set_xlabel(hist_ax.get_xlabel())  # share same x label

        plt.show(block=True)

    def _plot_binned_and_residuals(
            self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Plot binned scatter plots for predicted values and line plots for true values in
        different shades of red per bin.

        Arranged in a 1×2 grid.
        Predicted values: scatter with consistent color (blue) and varying alpha.
        True values: line plots with shades of red, no markers.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.

        Returns:
            None
        """
        # Only POWER vs COAL is plotted
        x1_vals, x2_vals, xlabel, ylabel = (
            self.model.x[POWER],
            self.model.x[COAL],
            POWER,
            COAL,
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        base_color_pred = "tab:blue"
        alphas_pred = [0.9, 0.7, 0.5, 0.3, 0.15]  # for scatter transparency
        reds_true = ["#ff9999", "#ff6666", "#ff3333", "#ff0000", "#990000"]  # darker shades

        # --- Binned along X1 ---
        bins_x1 = np.quantile(x1_vals, np.linspace(0, 1, 101))
        for i, alpha, red in zip([0, 25, 50, 75, 99], alphas_pred, reds_true):
            mask = (x1_vals >= bins_x1[i]) & (x1_vals < bins_x1[i + 1])
            if np.any(mask):
                # Predictions as scatter
                axes[0].scatter(
                    x2_vals[mask],
                    y_pred[mask],
                    color=base_color_pred,
                    s=30,
                    alpha=alpha,
                    marker="x",
                    label="Pred y" if i == 0 else "",
                )
                # True values as line without markers
                sorted_idx = np.argsort(x2_vals[mask])
                axes[0].plot(
                    x2_vals[mask][sorted_idx],
                    y_true[mask][sorted_idx],
                    color=red,
                    lw=1.5,
                    label="True y" if i == 0 else "",
                )

        axes[0].set_xlabel(ylabel)
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Binned along {xlabel} vs {ylabel}")
        axes[0].legend()

        # --- Binned along X2 ---
        bins_x2 = np.quantile(x2_vals, np.linspace(0, 1, 101))
        for i, alpha, red in zip([0, 25, 50, 75, 99], alphas_pred, reds_true):
            mask = (x2_vals >= bins_x2[i]) & (x2_vals < bins_x2[i + 1])
            if np.any(mask):
                # Predictions as scatter
                axes[1].scatter(
                    x1_vals[mask],
                    y_pred[mask],
                    color=base_color_pred,
                    s=30,
                    alpha=alpha,
                    marker="x",
                    label="Pred y" if i == 0 else "",
                )
                # True values as line without markers
                sorted_idx = np.argsort(x1_vals[mask])
                axes[1].plot(
                    x1_vals[mask][sorted_idx],
                    y_true[mask][sorted_idx],
                    color=red,
                    lw=1.5,
                    label="True y" if i == 0 else "",
                )

        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("y")
        axes[1].set_title(f"Binned along {ylabel} vs {xlabel}")
        axes[1].legend()

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
        0.5 * x_fwd_power**2
        + np.sin(x_fwd_coal)
        + 0.3 * x_fwd_power * x_fwd_coal
        + np.random.normal(0, 1, n_samples)
    )

    # Use SIMULATION_PATH constant for the dimension name
    sim_coord = xr.DataArray(
        np.arange(n_samples), dims=[SIMULATION_PATH], name=SIMULATION_PATH
    )

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

    min_var_delta = PolynomialDelta(
        fwds=fwds,
        y=y_true,
        beta=beta,
        efficiency=0.95,
    )

    y_pred = min_var_delta.delta()[KEY_PREDICTED]

    OpsPlotMinVarDelta(min_var_delta).plot(y_true=y_true, y_pred=y_pred)
