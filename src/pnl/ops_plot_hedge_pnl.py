import matplotlib.pyplot as plt
from typing import Optional, Tuple
from plot_helpers import plot_observables
import pandas as pd

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

    def plot_cash_account(
        self,
        path_index: Optional[int] = None,
        confidence_levels: Tuple[float, float] = (0.01, 0.05),
    ) -> None:
        """
        Plot hedge cash account PnL in two rows:

        • Upper row: Daily hedge cash account PnL with confidence intervals.
        • Lower row: Cumulative hedge cash account PnL with confidence intervals.

        Parameters
        ----------
        path_index : int, optional
            Index of a single simulation path to overlay. Default is None.
        confidence_levels : tuple of float, default=(0.01, 0.05)
            Percentile levels for confidence intervals (e.g., 1% and 5%).
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # --- Upper subplot: daily cash account ---
        plot_observables(
            x=self.pnl.cashflow_cash_account.index,
            data=self.pnl.cashflow_cash_account,
            confidence_levels=confidence_levels,
            path_index=path_index,
            ax=axes[0],
            title="Daily Cashflow Cash Account",
            show=False,
        )

        # --- Lower subplot: cumulative cash account ---
        plot_observables(
            x=self.pnl.cashflow_cash_account.index,
            data=self.pnl.cashflow_cash_account.cumsum(),
            confidence_levels=confidence_levels,
            path_index=path_index,
            ax=axes[1],
            title="Cash Account (cumulative)",
            show=False,
        )

        axes[1].axhline(
            y=self.pnl.asset.value_0,  # y-position of the horizontal line
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Asset Value 0 = {self.pnl.asset.value_0:.1f}"  # no decimals
        )

        axes[1].legend(
            loc="upper center",  # center the legend horizontally
            bbox_to_anchor=(0.5, -0.15),  # place below the x-axis
            fontsize="small",
            ncol=1  # single column (adjust if you add more items)
        )

        plt.tight_layout()
        plt.show(block=True)

    def plot_asset_and_hedge_cashflows(self, path_index: Optional[int] = None) -> None:
        """
        Plot daily and cumulative hedge/asset cashflows and their difference
        in a 3×2 grid of subplots.

        Layout (rows × columns):
            ┌──────────────┬────────────────────┐
            │ Hedge Daily  │ Hedge Cumulative   │
            ├──────────────┼────────────────────┤
            │ Asset Daily  │ Asset Cumulative   │
            ├──────────────┼────────────────────┤
            │ Diff Daily   │ Diff Cumulative    │
            └──────────────┴────────────────────┘

        Parameters
        ----------
        path_index : int, optional
            Index of a single simulation path to overlay. Default is None.
        """
        hedge_daily = self.pnl.cashflows_from_hedge
        hedge_cum = self.pnl.cashflows_from_hedge_cumulative
        asset_daily = self.pnl.cashflows_from_asset
        asset_cum = self.pnl.cashflows_from_asset_cumulative
        diff_daily = hedge_daily - asset_daily
        diff_cum = hedge_cum - asset_cum

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

        # --- Row 1: Hedge ---
        plot_observables(
            x=hedge_daily.index,
            data=hedge_daily,
            path_index=path_index,
            ax=axes[0, 0],
            title="Hedge Cashflows (Daily)",
            show=False,
        )
        plot_observables(
            x=hedge_cum.index,
            data=hedge_cum,
            path_index=path_index,
            ax=axes[0, 1],
            title="Hedge Cashflows (Cumulative)",
            show=False,
        )

        # --- Row 2: Asset ---
        plot_observables(
            x=asset_daily.index,
            data=asset_daily,
            path_index=path_index,
            ax=axes[1, 0],
            title="Asset Cashflows (Daily)",
            show=False,
        )
        plot_observables(
            x=asset_cum.index,
            data=asset_cum,
            path_index=path_index,
            ax=axes[1, 1],
            title="Asset Cashflows (Cumulative)",
            show=False,
        )

        # --- Row 3: Difference ---
        plot_observables(
            x=diff_daily.index,
            data=diff_daily,
            path_index=path_index,
            ax=axes[2, 0],
            title="Difference (Hedge - Asset, Daily)",
            show=False,
        )
        plot_observables(
            x=diff_cum.index,
            data=diff_cum,
            path_index=path_index,
            ax=axes[2, 1],
            title="Difference (Hedge - Asset, Cumulative)",
            show=False,
        )

        plt.tight_layout()
        plt.show(block=True)

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
        self.plot_asset_and_hedge_cashflows(path_index=path_index)
