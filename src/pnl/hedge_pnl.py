import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import Optional

from constants import ASSET, DELIVERY_START, SIMULATION_DAY
from hedging.min_var_hedge import MinVarHedge
from valuation.power_plant.power_plant import PowerPlant


class HedgePnL:
    """
    Calculate and analyze the Profit and Loss (PnL) of a hedging strategy.

    This class tracks daily hedge and asset cashflows, accumulates their
    cumulative totals, and computes the cash account PnL from hedge rebalancing.

    Attributes
    ----------
    n_sims : int
        Number of Monte Carlo simulation paths.
    simulation_days : pd.DatetimeIndex
        Ordered sequence of simulation days.
    asset : PowerPlant
        Underlying power plant whose production is being hedged.
    hedge : MinVarHedge
        Hedging strategy containing forward deltas and hedge cashflows.
    cashflows_from_hedge : pd.DataFrame
        Daily hedge cashflows per simulation path.
        Index: simulation_days; Columns: simulation path indices.
    cashflows_from_asset : pd.DataFrame
        Daily asset cashflows per simulation path.
    cashflows_from_hedge_cumulative : pd.DataFrame
        Cumulative hedge cashflows across time.
    cashflows_from_asset_cumulative : pd.DataFrame
        Cumulative asset cashflows across time.
    cashflow_cash_account : pd.DataFrame
        Daily PnL of the hedge cash account per path.
    cash_account_total : pd.DataFrame
        Cumulative hedge cash account per path.
    fwds : xr.DataArray
        Forward price curves indexed by (SIMULATION_DAY, DELIVERY_START, ASSET).
    """

    def __init__(
        self,
        n_sims: int,
        simulation_days: pd.DatetimeIndex,
        asset: PowerPlant,
        hedge: MinVarHedge,
    ) -> None:
        """
        Initialize the HedgePnL instance.

        Parameters
        ----------
        n_sims : int
            Number of Monte Carlo simulation paths.
        simulation_days : pd.DatetimeIndex
            Sequence of trading/simulation days.
        asset : PowerPlant
            Asset providing daily cashflows to be hedged.
        hedge : MinVarHedge
            Hedging strategy with deltas and hedge cashflows.
        """
        self.n_sims: int = n_sims
        self.simulation_days: pd.DatetimeIndex = simulation_days
        self.asset: PowerPlant = asset
        self.hedge: MinVarHedge = hedge

        # Daily and cumulative cashflows
        self.cashflows_from_hedge: pd.DataFrame = hedge.cashflows
        self.cashflows_from_asset: pd.DataFrame = asset.cashflows

        # Forward price curves
        self.fwds: xr.DataArray = hedge.fwds

        # Daily hedge cash account PnL and its cumulative value
        self.cashflow_cash_account: pd.DataFrame = pd.DataFrame(
            index=simulation_days,
            columns=range(n_sims),
            data=np.zeros((len(simulation_days), n_sims)),
        )

        self.value_fwd_positions = pd.DataFrame(
            index=simulation_days,
            columns=range(n_sims),
            data=np.zeros((len(simulation_days), n_sims)),
        )

    # ------------------------------------------------------------------ #
    def calculate_pnl(self) -> None:
        """
        Compute hedge cash account PnL across all simulation days.

        Steps
        -----
        1. Buy initial delta positions at the first simulation day.
        2. Rebalance delta positions on each subsequent simulation day.
        A progress bar tracks simulation progress.
        """
        # Step 1: buy initial hedge positions
        self._initialize_delta_positions()

        # Step 2: rebalance positions over remaining days
        for simulation_day in tqdm(
            self.simulation_days[1:],
            total=len(self.simulation_days) - 1,
            desc="Calculating PnL",
            unit="day",
        ):
            self._rebalance_delta_positions(simulation_day)

    # ------------------------------------------------------------------ #
    def _initialize_delta_positions(self) -> None:
        """
        Establish initial hedge delta positions on the first simulation day.

        Buys the required forward contracts for the front-month delivery
        period and updates the hedge cash account.
        """
        simulation_start: pd.Timestamp = self.simulation_days[0]
        delivery_start_dates = self.hedge.get_front_months_start_dates(simulation_start)

        for asset_name in self.hedge.deltas.coords[ASSET].values:
            for delivery_start_date in delivery_start_dates:
                n_days_in_front_month: int = pd.Timestamp(
                    delivery_start_date
                ).days_in_month

                forward_price = self.fwds.sel(
                    {
                        ASSET: asset_name,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_start,
                    }
                )
                delta_position = self.hedge.deltas.sel(
                    {
                        ASSET: asset_name,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_start,
                    }
                )

                value_fwd_position = (
                    n_days_in_front_month * delta_position * forward_price
                )

                self.cashflow_cash_account.loc[simulation_start] -= value_fwd_position

                self.value_fwd_positions.loc[simulation_start] += value_fwd_position

    # ------------------------------------------------------------------ #
    def _rebalance_delta_positions(self, simulation_day: pd.Timestamp) -> None:
        """
        Rebalance hedge delta positions for a given simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            Current simulation day for which hedge positions are updated.

        Notes
        -----
        The cash account is adjusted by the cost of the incremental
        forward contracts required to move from yesterday's delta to
        today's delta.
        """
        delivery_start_dates = self.hedge.get_front_months_start_dates(simulation_day)

        for asset_name in self.hedge.deltas.coords[ASSET].values:
            for delivery_start_date in delivery_start_dates:
                n_days_in_front_month: int = pd.Timestamp(
                    delivery_start_date
                ).days_in_month

                forward_prices_today = self.fwds.sel(
                    {
                        ASSET: asset_name,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )
                delta_positions_today = self.hedge.deltas.sel(
                    {
                        ASSET: asset_name,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: simulation_day,
                    }
                )

                previous_day: pd.Timestamp = simulation_day + pd.Timedelta(days=-1)
                delta_positions_yesterday = self.hedge.deltas.sel(
                    {
                        ASSET: asset_name,
                        DELIVERY_START: delivery_start_date,
                        SIMULATION_DAY: previous_day,
                    }
                )

                d_delta = delta_positions_today - delta_positions_yesterday

                self.cashflow_cash_account.loc[simulation_day] -= (
                    n_days_in_front_month * d_delta * forward_prices_today
                )

                self.value_fwd_positions.loc[simulation_day] += (
                    delta_positions_today * n_days_in_front_month * forward_prices_today
                )
