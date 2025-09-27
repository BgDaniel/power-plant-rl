import matplotlib.pyplot as plt
from typing import Optional, Tuple
from plot_helpers import plot_observables


from pnl.hedge_pnl import HedgePnL


class OpsPlotPnL:
    """
    Class to handle plotting of Hedge and Asset PnL / cashflows using `plot_observables`.

    Attributes
    ----------
    pnl : HedgePnL
        The HedgePnL object containing simulation results and cashflows.
    """

    def __init__(self, pnl: HedgePnL) -> None:
        """
        Initialize the plotting utility with a HedgePnL instance.

        Parameters
        ----------
        pnl : HedgePnL
            HedgePnL instance containing simulation results.
        """
        self.pnl = pnl

    def plot_pnl(
        self,
        path_index: Optional[int] = None,
        confidence_levels: tuple[float, float] = (0.01, 0.05),
    ) -> None:
        """
        Plot the overall PnL of the hedging strategy.

        Overall PnL is defined as:

            Asset cumulative cashflows
            - ( Hedge cumulative cashflows
                + Hedge forward positions value
                + Cash account total )

        Parameters
        ----------
        path_index : int, optional
            Index of a single simulation path to overlay. Default is None.
        confidence_levels : tuple of float, default=(0.01, 0.05)
            Percentile levels for confidence intervals (e.g., 1% and 5%).
        """
        # --- Retrieve components ---
        asset_cum = self.pnl.cashflows_from_asset.cumsum()
        hedge_cum = self.pnl.cashflows_from_hedge.cumsum()
        cash_accum = self.pnl.cashflow_cash_account.cumsum()
        fwds_val = self.pnl.value_fwd_positions

        # Align all series on the same index
        pnl_index = asset_cum.index
        hedge_cum = hedge_cum.reindex(pnl_index)
        cash_accum = cash_accum.reindex(pnl_index)
        fwds_val = fwds_val.reindex(pnl_index)

        # --- Compute overall PnL ---
        overall_pnl = asset_cum - (hedge_cum + fwds_val + cash_accum)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_observables(
            x=pnl_index,
            data=overall_pnl,
            confidence_levels=confidence_levels,
            path_index=path_index,
            ax=ax,
            title="PnL Overall (CF Asset - (CF Hedge + Fwd. Pos. + Cash)) Accum.",
            show=False,
        )

        ax.axhline(
            y=self.pnl.asset.value_0,  # y-position of the horizontal line
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Asset Value 0 = {self.pnl.asset.value_0:.1f}",  # no decimals
        )

        ax.legend(
            loc="upper center",  # center the legend horizontally
            bbox_to_anchor=(0.5, -0.15),  # place below the x-axis
            fontsize="small",
            ncol=1,  # single column (adjust if you add more items)
        )

        ax.grid(True)
        plt.tight_layout()
        plt.show(block=True)
