import pandas as pd
import xarray as xr
import numpy as np

from constants import ASSET, DELIVERY_START, SIMULATION_DAY
from hedging.min_var_hedge import MinVarHedge
from valuation.power_plant.power_plant import PowerPlant


class HedgePnL:
    """
    Class to calculate and analyze the Profit and Loss (PnL) of a hedging strategy.

    Attributes
    ----------
    hedge : Hedge
        The hedging strategy object containing cashflows and other relevant data.
    pnl : pd.Series
        The calculated PnL series.
    cumulative_pnl : pd.Series
        The cumulative PnL series.
    """

    def __init__(
        self,
        n_sims: int,
        simulation_days: pd.DatetimeIndex,
        asset: PowerPlant,
        hedge: MinVarHedge,
        fwds: xr.DataArray,
    ) -> None:
        """
        Initialize the HedgePnL class with a hedging strategy.

        Parameters
        ----------
        hedge : Hedge
            The hedging strategy object.
        """
        self.n_sims = n_sims
        self.simulation_days = simulation_days

        self.hedge = hedge

        self.cashflows_from_hedge = hedge.cashflows
        self.cashflows_from_asset = asset.cashflows

        self.cashflows_from_hedge_cumulative = hedge.cashflows.cumsum()
        self.cashflows_from_asset_cumulative = asset.cashflows.cumsum()

        self.fwds = fwds

        self.cash_account = pd.DataFrame(index=simulation_days, colums=range(self.n_sims), data= np.zeros((len(simulation_days, self.n_sims))))
        self.cash_account_cumulative = self.cash_account.cumsum()

    def calculate_pnl(self):
        """
        Calculate the PnL of the hedging strategy based on its cashflows.

        Returns
        -------
        pd.Series
            The calculated PnL series.
        """
        simulation_start = self.simulation_days[0]

        # buy initial delta positions
        self._buy_delta_positions(simulation_start)

        for simulation_day in self.simulation_days[1:]:
            # rebalance delta positions
            self._rebalance_delta_positions(simulation_day)

    def _buy_delta_positions(self, simulation_day: pd.Timestamp):

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

                # Update cash account for buying delta positions
                self.cash_account.loc[simulation_day] -= delta_position * forward_price

        pass

    def _rebalance_delta_positions(self, simulation_day: pd.Timestamp):
        for asset in self.hedge.deltas.coords[ASSET].values:
            for delivery_start_date in (
                self.hedge.deltas.loc[ASSET:asset].coords[DELIVERY_START].values
            ):
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
                previous_day = self.simulation_days[
                    self.simulation_days.get_loc(simulation_day) - 1
                ]
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

                # Update cash account for rebalancing delta positions
                self.cash_account.loc[simulation_day] = (
                    delta_positions_yesterday * forward_prices_yesterday
                    - delta_positions_today * forward_prices_today
                )

    def plot_pnl(self):
        """
        Plot the PnL and cumulative PnL of the hedging strategy.
        """
        pass
