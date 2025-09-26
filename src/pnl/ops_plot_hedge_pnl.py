import matplotlib.pyplot as plt
from typing import Optional
from plot_helpers import plot_observables


class OpsPlotPnL:
    """
    Class to handle plotting of Hedge and Asset PnL / cashflows using `plot_observables`.
    """

    def __init__(self, pnl) -> None:
        self.pnl = pnl

    def plot_cash_account(
        self,
        path_index: Optional[int] = None,
        confidence_levels: tuple[float, float] = (0.01, 0.05),
    ) -> None:
        """Plot mean cash account PnL with confidence intervals and optional single path."""
        plot_observables(
            x=self.pnl.simulation_days,
            data=self.pnl.cash_account,
            confidence_levels=confidence_levels,
            path_index=path_index,
            title="Hedge Cash Account PnL",
            ylabel="Cash Account / PnL",
        )

    def plot_cumulative_cashflows(self, path_index: Optional[int] = None) -> None:
        """
        Plot cumulative hedge cashflows, asset cashflows, and their difference in a 3-row subplot.
        """
        hedge_cum = self.pnl.cashflows_from_hedge_cumulative
        asset_cum = self.pnl.cashflows_from_asset_cumulative
        difference = hedge_cum - asset_cum
        days = self.pnl.hedge.hedge_days

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        plot_observables(
            x=days,
            data=hedge_cum,
            path_index=path_index,
            ax=axes[0],
            title="Cumulative Hedge Cashflows",
            ylabel="Hedge Cumulative",
            show=False,
        )

        plot_observables(
            x=days,
            data=asset_cum,
            path_index=path_index,
            ax=axes[1],
            title="Cumulative Asset Cashflows",
            ylabel="Asset Cumulative",
            show=False,
        )

        plot_observables(
            x=days,
            data=difference,
            path_index=path_index,
            ax=axes[2],
            title="Difference (Hedge - Asset)",
            ylabel="Difference",
            show=False,
        )

        plt.tight_layout()
        plt.show()

    def plot_pnl(
        self,
        path_index: Optional[int] = None,
        confidence_levels: tuple[float, float] = (0.01, 0.05),
    ) -> None:
        """
        Convenience method to plot both the cash account PnL and the cumulative cashflows.
        """
        self.plot_cash_account(
            path_index=path_index, confidence_levels=confidence_levels
        )
        self.plot_cumulative_cashflows(path_index=path_index)
