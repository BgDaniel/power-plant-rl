import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from hedging.min_var_hedge import MinVarHedge

from constants import POWER, COAL, ASSET, DELIVERY_START



class OpsPlotMinVarHedge:
    """
    Class for plotting regression R² results from a MinVarHedge simulation.

    Attributes
    ----------
    hedge : MinVarHedge
        The MinVarHedge instance containing regression results and R² scores.
    """

    def __init__(self, hedge: MinVarHedge) -> None:
        """
        Initialize the plotting utility.

        Parameters
        ----------
        hedge : MinVarHedge
            MinVarHedge instance with completed simulation and R² scores.
        """
        self.hedge = hedge

    def plot_r2(self) -> None:
        """
        Plot R² values over simulation days for both POWER and COAL
        in a two-row plot, with each delivery start date shown as a separate line.
        """
        assets = [POWER, COAL]
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for ax, asset in zip(axes, assets):
            for delivery_start in self.hedge._delivery_start_dates:
                # Use .sel to fetch R² values per delivery start and asset
                r2_values = self.hedge._r2_scores.sel(
                    {DELIVERY_START: delivery_start, ASSET: asset}
                ).values
                # Ensure delivery_start is a pandas Timestamp for formatting
                delivery_ts = pd.Timestamp(delivery_start)
                ax.plot(
                    self.hedge._simulation_days,
                    r2_values,
                    label=f"Delivery {delivery_ts.strftime('%Y-%m')}"
                )

            ax.set_title(f"{asset}")  # Subplot title: asset
            ax.set_ylabel("R² Value")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        axes[1].set_xlabel("Simulation Day")
        fig.suptitle("R² Scores", fontsize=16)  # Super title
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.show(block=True)

    def plot_hedge_effectiveness(self, cashflows_from_asset: pd.DataFrame) -> None:
        """
        Two-panel plot showing:
        - Histogram of total cashflows and residuals.
        - Row-wise variance time series with secondary axis for R².

        Parameters
        ----------
        cashflows_from_asset : pd.DataFrame
            Optimal cashflows to compare against. Must have the same shape as self.hedge.cashflows
        """
        cashflows_from_hedge = self.hedge.cashflows

        if cashflows_from_hedge.shape != cashflows_from_asset.shape:
            raise ValueError("MinVar and optimal cashflows must have the same shape.")

        # Compute residuals
        residuals = cashflows_from_hedge - cashflows_from_asset

        # Row-wise variance
        var_hedge = cashflows_from_hedge.var(axis=1)
        var_asset = cashflows_from_asset.var(axis=1)
        var_residuals = residuals.var(axis=1)

        # Row-wise R²
        r2_rows = np.array([
            r2_score(cashflows_from_asset.iloc[i, :], cashflows_from_hedge.iloc[i, :])
            for i in range(residuals.shape[0])
        ])

        # --------------------------
        # Upper plot: histogram
        # --------------------------
        total_cashflows_hedge = cashflows_from_hedge.sum(axis=1)
        total_residuals = residuals.sum(axis=1)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

        axes[0].hist(total_cashflows_hedge, bins=30, alpha=0.6, label="Total Hedge Cashflows", color='blue')
        axes[0].hist(total_residuals, bins=30, alpha=0.6, label="Total Residuals", color='red')
        axes[0].set_title("Histogram: Total Cashflows vs Residuals")
        axes[0].set_xlabel("Sum of Cashflows")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True)

        # --------------------------
        # Lower plot: time series of row-wise variance
        # --------------------------
        ax1 = axes[1]
        ax2 = ax1.twinx()  # secondary y-axis for R²

        ax1.plot(var_hedge, label="Variance: Hedge Cashflows", lw=2, color='blue')
        ax1.plot(var_asset, label="Variance: Asset Cashflows", lw=2, color='green')
        ax1.plot(var_residuals, label="Variance: Residuals", lw=2, color='red')

        ax2.plot(r2_rows, label="Row-wise R²", lw=2, linestyle='--', color='black')

        ax1.set_xlabel("Simulation Day / Path")
        ax1.set_ylabel("Variance")
        ax2.set_ylabel("Row-wise R²")

        ax1.set_title("Row-wise Variance and R² Over Simulation Days")
        ax1.grid(True)

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.tight_layout()
        plt.show(block=True)