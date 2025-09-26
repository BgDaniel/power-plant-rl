import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import ASSET, DELIVERY_START, SIMULATION_DAY
from hedging.min_var_hedge import MinVarHedge
from valuation.power_plant.power_plant import PowerPlant


class HedgePnL:
    """
    Class to calculate and analyze the Profit and Loss (PnL) of a hedging strategy.

    Attributes
    ----------
    n_sims : int
        Number of simulation paths.
    simulation_days : pd.DatetimeIndex
        Index of simulation days.
    hedge : MinVarHedge
        The hedging strategy object containing deltas and cashflows.
    cashflows_from_hedge : pd.DataFrame
        Hedge cashflows per simulation path.
    cashflows_from_asset : pd.DataFrame
        Asset cashflows per simulation path.
    cashflows_from_hedge_cumulative : pd.DataFrame
        Cumulative hedge cashflows.
    cashflows_from_asset_cumulative : pd.DataFrame
        Cumulative asset cashflows.
    cash_account : pd.DataFrame
        Daily PnL changes per simulation path.
    cash_account_cumulative : pd.DataFrame
        Cumulative PnL per simulation path.
    fwds : xr.DataArray
        Forward prices used for hedging.
    """

    def __init__(
        self,
        n_sims: int,
        simulation_days: pd.DatetimeIndex,
        asset: PowerPlant,
        hedge: MinVarHedge,
    ) -> None:
        """
        Initialize the HedgePnL class with a hedging strategy.

        Parameters
        ----------
        n_sims : int
            Number of simulation paths.
        simulation_days : pd.DatetimeIndex
            Index of simulation days.
        asset : PowerPlant
            The underlying asset for the hedge.
        hedge : MinVarHedge
            The hedging strategy object.
        """
        self.n_sims: int = n_sims
        self.simulation_days: pd.DatetimeIndex = simulation_days
        self.hedge: MinVarHedge = hedge

        self.cashflows_from_hedge: pd.DataFrame = hedge.cashflows
        self.cashflows_from_asset: pd.DataFrame = asset.cashflows

        self.cashflows_from_hedge_cumulative: pd.DataFrame = hedge.cashflows.cumsum()
        self.cashflows_from_asset_cumulative: pd.DataFrame = asset.cashflows.cumsum()

        self.fwds: xr.DataArray = hedge.fwds

        self.cash_account: pd.DataFrame = pd.DataFrame(
            index=simulation_days,
            columns=range(n_sims),
            data=np.zeros((len(simulation_days), n_sims)),
        )
        self.cash_account_cumulative: pd.DataFrame = self.cash_account.cumsum()

    def calculate_pnl(self) -> None:
        """
        Calculate the PnL of the hedging strategy based on its cashflows.
        Uses a progress bar to track simulation day processing.
        """
        # Buy initial delta positions on the first day
        simulation_start: pd.Timestamp = self.simulation_days[0]
        self._buy_delta_positions(simulation_start)

        # Rebalance delta positions for remaining days
        for simulation_day in tqdm(
            self.simulation_days[1:],
            total=len(self.simulation_days) - 1,
            desc="Calculating PnL",
            unit="day",
        ):
            self._rebalance_delta_positions(simulation_day)

    def _buy_delta_positions(self, simulation_day: pd.Timestamp) -> None:
        """Buy initial delta positions for the hedging strategy."""
        for asset in self.hedge.deltas.coords[ASSET].values:
            for delivery_start_date in (
                self.hedge.deltas.loc[ASSET:asset].coords[DELIVERY_START].values
            ):
                forward_price = self.fwds.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )
                delta_position = self.hedge.deltas.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )

                self.cash_account.loc[simulation_day] -= delta_position * forward_price

    def _rebalance_delta_positions(self, simulation_day: pd.Timestamp) -> None:
        """Rebalance delta positions for a given simulation day."""
        delivery_start_dates = self.hedge.get_front_months_start_dates(simulation_day)

        for asset in self.hedge.deltas.coords[ASSET].values:
            for delivery_start_date in delivery_start_dates:
                forward_prices_today = self.fwds.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )
                delta_positions_today = self.hedge.deltas.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )

                previous_day: pd.Timestamp = simulation_day + pd.Timedelta(days=-1)
                delta_positions_yesterday = self.hedge.deltas.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: previous_day,
                    }
                )
                forward_prices_yesterday = self.fwds.sel(
                    {
                        ASSET: asset,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: previous_day,
                    }
                )

                self.cash_account.loc[simulation_day] = (
                    delta_positions_yesterday * forward_prices_yesterday
                    - delta_positions_today * forward_prices_today
                )

    def plot_pnl(
        self,
        path_index: int | None = None,
        confidence_levels: tuple[float, float] = (1.0, 5.0),
    ) -> None:
        """
        Plot the mean PnL and cumulative PnL of the hedging strategy with configurable confidence bands.

        Parameters
        ----------
        path_index : int | None, optional
            Index of the simulation path to overlay. If None, only aggregate statistics are plotted.
        confidence_levels : tuple[float, float], default=(1.0, 5.0)
            Percentile levels for confidence intervals. Example: (1.0, 5.0) → 1% and 5%.
        """
        lower1, lower5 = confidence_levels
        cash: np.ndarray = self.cash_account.values
        asset_days: pd.DatetimeIndex = self.simulation_days

        mean_cash: np.ndarray = cash.mean(axis=1)
        lower1_vals: np.ndarray = np.percentile(cash, lower1, axis=1)
        upper1_vals: np.ndarray = np.percentile(cash, 100 - lower1, axis=1)
        lower5_vals: np.ndarray = np.percentile(cash, lower5, axis=1)
        upper5_vals: np.ndarray = np.percentile(cash, 100 - lower5, axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot mean
        ax.plot(asset_days, mean_cash, color="green", lw=1.0, label="Mean PnL")

        # Plot confidence intervals
        ax.fill_between(
            asset_days,
            lower5_vals,
            upper5_vals,
            color="gray",
            alpha=0.2,
            label=f"{lower5}–{100 - lower5}% CI",
        )
        ax.fill_between(
            asset_days,
            lower1_vals,
            upper1_vals,
            color="gray",
            alpha=0.4,
            label=f"{lower1}–{100 - lower1}% CI",
        )

        # Plot a single simulation path if specified
        if path_index is not None:
            ax.plot(
                asset_days,
                cash[:, path_index],
                color="blue",
                lw=1.0,
                linestyle="--",
                label=f"Simulation Path {path_index}",
            )

        # Plot the target final value of the power plant
        final_value: float = self.cashflows_from_asset_cumulative.iloc[-1].mean()
        ax.scatter(
            asset_days[-1],
            final_value,
            color="yellow",
            marker="x",
            s=80,
            label="Target Value (Asset End)",
        )

        ax.set_xlabel("Simulation Day")
        ax.set_ylabel("Cash Account / PnL")
        ax.set_title("Hedge PnL Simulation")
        ax.grid(True)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()
