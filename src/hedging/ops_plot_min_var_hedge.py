import matplotlib.pyplot as plt
import pandas as pd
from hedging.min_var_hedge import MinVarHedge, POWER, COAL, ASSET
from market_simulation.constants import DELIVERY_START


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
