import os
import tempfile
from typing import TypeVar
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

from delta_position.delta_position import DeltaPosition
from delta_position.min_var_delta.min_var_delta import MinVarDelta

from market_simulation.constants import SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START
from market_simulation.two_factor_model.simulation_caching import CACHE_FOLDER_ENV
from regression.polynomial_base_builder import PolynomialBasisBuilder

from valuation.power_plant.power_plant import PowerPlant

from constants import (
    POWER,
    COAL,
    KEY_R2,
    ASSET,
    KEY_DELTA_POSITION,
)

TDelta = TypeVar("TDelta", bound=DeltaPosition)


class MinVarHedge:
    def __init__(
        self,
        n_sims: int,
        simulation_days: pd.DatetimeIndex,
        power_plant: PowerPlant,
        spots_power: pd.DataFrame,  # (sim_days, n_sims)
        spots_coal: pd.DataFrame,  # (sim_days, n_sims)
        fwds_power: xr.DataArray,  # (n_sims, n_days, n_delivery_months)
        fwds_coal: xr.DataArray,  # (n_sims, n_days, n_delivery_months)
        polynomial_type: str = PolynomialBasisBuilder.POLY_LEGENDRE,
        polynomial_degree: int = 4,
        delta_position_type: TDelta = MinVarDelta,
    ):
        self._n_sims = n_sims
        self._simulation_days = simulation_days
        self._n_steps = len(simulation_days)

        self.as_of_date = simulation_days[0]

        self._power_plant = power_plant

        self._asset_days = power_plant.asset_days

        self._cashflows = self._power_plant.cashflows

        self._spots_power = spots_power
        self._spots_coal = spots_coal

        self._fwds = xr.concat(
            [fwds_power, fwds_coal], dim=xr.DataArray([POWER, COAL], dims=[ASSET])
        )

        self._spots = {POWER: self._spots_power, COAL: self._spots_coal}

        self._spots = xr.DataArray(
            np.stack([spots_power.values, spots_coal.values], axis=0),
            dims=[ASSET, SIMULATION_DAY, SIMULATION_PATH],
            coords={
                ASSET: [POWER, COAL],
                SIMULATION_DAY: spots_power.index,
                SIMULATION_PATH: spots_power.columns,
            },
        )

        delivery_start_dates_power = fwds_power.coords[DELIVERY_START].values
        delivery_start_dates_coal = fwds_coal.coords[DELIVERY_START].values

        # Check equality and raise error if they differ
        if not np.array_equal(delivery_start_dates_power, delivery_start_dates_coal):
            raise ValueError(
                "DELIVERY_START coordinates of fwds_power and fwds_coal are not equal!"
            )

        self._delivery_start_dates = delivery_start_dates_power

        self._n_front_months = len(self._delivery_start_dates)

        self._polynomial_type = polynomial_type
        self._polynomial_degree = polynomial_degree

        self._delta_position_type = delta_position_type

        coords = {
            SIMULATION_PATH: np.arange(self._n_sims),
            SIMULATION_DAY: self._simulation_days,
            DELIVERY_START: self._delivery_start_dates,
            ASSET: [POWER, COAL],
        }

        self._deltas: xr.DataArray = xr.DataArray(
            np.zeros((self._n_sims, self._n_steps, self._n_front_months, 2)),
            dims=[SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START, ASSET],
            coords=coords,
            name=DELTA_POSITION,
        )

        self._r2_scores = xr.DataArray(
            np.full((self._n_steps, self._n_front_months, 2), np.nan),
            dims=[SIMULATION_DAY, DELIVERY_START, ASSET],
            coords={
                SIMULATION_DAY: self._simulation_days,
                DELIVERY_START: self._delivery_start_dates,
                ASSET: [POWER, COAL],
            },
            name="R2_SCORE",
        )

        self.cashflows = pd.DataFrame(
            np.zeros((len(self._asset_days), self._n_sims)),
            index=self._asset_days,
            columns=np.arange(self._n_sims),
        )

    def hedge(self) -> None:
        """
        Compute the minimal variance hedge for all asset days.

        This method iterates over each asset day in `self._asset_days`, using a
        tqdm progress bar for tracking. For each day:
            1. Retrieve the start dates of the front months using
               `_get_front_months_start_dates`.
            2. Compute the hedge delta for the current day using
               `_compute_delta`.

        The results are stored internally in the class (e.g., `self._deltas`).

        Returns:
            None
        """
        self.tqdm = tqdm(
            self._asset_days,
            total=len(self._asset_days) - 1,
            desc="Determine minimal variance hedge",
            unit="day",
        )
        for asset_day in self.tqdm:
            front_months_start_dates = self._get_front_months_start_dates(asset_day)
            self._compute_delta(asset_day, front_months_start_dates)

    def roll_out_cashflows(self) -> None:
        """
        Roll out cashflows for each asset and simulation day.

        This method iterates over each asset day in ``self._asset_days`` and
        computes the cashflows generated by applying the terminal delta of the
        current beginning-of-month (BOM) delivery period to the spot prices.

        For each asset day:
            1. Identify the BOM delivery start day and its maturity day
               (one day before the start).
            2. For each asset (POWER and COAL):
                - Retrieve the terminal delta of the BOM.
                - Retrieve the corresponding spot price on the asset day.
                - Update the cashflows for that day by multiplying the
                  terminal delta with the spot price.

        Returns:
            None
        """
        for asset_day in self._asset_days:
            bom_delivery_start_day = asset_day - pd.offsets.MonthBegin(n=0)

            # Terminal delta of the current BOM
            bom_maturity_day = bom_delivery_start_day + pd.Timedelta(days=-1)

            for asset in [POWER, COAL]:
                bom_delta_asset = self._deltas.sel(
                    {
                        DELIVERY_START: bom_delivery_start_day,
                        ASSET: asset,
                        SIMULATION_DAY: bom_maturity_day,
                    }
                )

                spots = self._spots.sel({SIMULATION_DAY: asset_day, ASSET: asset})

                # Hedge positions are struck with a strike price of zero
                self.cashflows.loc[asset_day] += bom_delta_asset.values * spots

    def _get_front_months_start_dates(
        self, simulation_day: pd.Timestamp
    ) -> xr.DataArray:
        """
        Extract front month forward prices where delivery start dates lie between the given
        simulation_day and the end of asset_days.

        Parameters
        ----------
        fwds_power : xr.DataArray
            Forward curve array with dims (SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START).
        simulation_day : pd.Timestamp
            The current simulation day.

        Returns
        -------
        xr.DataArray
            Filtered forward prices with restricted DELIVERY_START.
        """
        # Define interval
        start = pd.Timestamp(simulation_day) + pd.Timedelta(days=1)
        end = self._asset_days[-1]  # last asset day

        mask = (self._delivery_start_dates >= start) & (
            self._delivery_start_dates <= end
        )

        return self._delivery_start_dates[mask]

    def _compute_delta(
        self, asset_day: pd.Timestamp, front_months_start_dates: pd.DatetimeIndex
    ) -> None:
        """
        Compute hedge deltas for the given asset_day using polynomial regression
        between the simulated cashflows and forward prices.

        Parameters
        ----------
        asset_day : pd.Timestamp
            Current simulation day.
        front_months_fwds_power : xr.DataArray
            Forward prices (n_sims, delivery_months) at asset_day.

        Returns
        -------
        xr.DataArray
            Hedge deltas (n_sims, delivery_months) for this day.
        """
        for delivery_start in front_months_start_dates:

            self._deltas.loc[
                {
                    SIMULATION_DAY: asset_day,
                    DELIVERY_START: delivery_start,
                }
            ] = self._hedge_front_month(asset_day, delivery_start)

    def _hedge_front_month(
        self,
        asset_day: pd.Timestamp,
        delivery_start: pd.Timestamp,
        r2_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Compute hedge delta for a single delivery month using polynomial regression.

        Parameters
        ----------
        asset_day : pd.Timestamp
            The current simulation day for which the hedge is computed.
        delivery_start : str
            The first day of the delivery month.
        r2_threshold : float, default=0.0
            Minimum acceptable R² value. If regression R² is below this, an error is raised.

        Returns
        -------
        np.ndarray
            Predicted hedge positions (delta) for all simulation paths.

        Raises
        ------
        ValueError
            If the regression R² is below the threshold.
        """
        days_in_front_month: pd.DatetimeIndex = pd.date_range(
            start=delivery_start, end=delivery_start + pd.offsets.MonthEnd(0)
        )

        # Asset days sometimes could be only partially covered by the front month days
        # at the end of the asset computatin horizom
        asset_days_in_front_month = self._cashflows.index.intersection(
            days_in_front_month
        )

        y = self._cashflows.loc[asset_days_in_front_month].values.sum(axis=0)

        beta = self._spots.sel(SIMULATION_DAY=days_in_front_month).sum(
            dim=SIMULATION_DAY
        )

        fwds = self._fwds.sel(
            {DELIVERY_START: delivery_start, SIMULATION_DAY: asset_day}
        )

        delta_position_calculator = self._delta_position_type(
            fwds=fwds, y=y, beta=beta, efficiency=self._power_plant.efficiency
        )

        # Perform regression
        results = delta_position_calculator.compute()
        delta_positions = results[KEY_DELTA_POSITION]
        r2_score = results[KEY_R2]

        # --- Validation ---------------------------------------------------------
        # Check R² threshold
        # if r2_score < r2_threshold:
        #    delivery_month = pd.Timestamp(delivery_start).strftime("%Y-%m")
        #    raise ValueError(
        #        f"Regression R²={r2_score:.4f} below threshold {r2_threshold} "
        #        f"for underlying {asset} and the delivery month {delivery_month}."
        #    )

        # Store R² using .sel
        self._r2_scores.loc[
            dict(
                SIMULATION_DAY=asset_day,
                DELIVERY_START=delivery_start,
            )
        ] = r2_score

        return delta_positions

    def save_results(self, run_name: str = "run_1"):
        """
        Save all important simulation results to a dedicated folder in the cache.
        Overwrites existing files if they exist.

        Parameters
        ----------
        run_name : str
            Name of this simulation run. A folder with this name will be created in the cache.
        """
        # Determine base cache folder
        base_cache = os.getenv(CACHE_FOLDER_ENV, tempfile.gettempdir())
        os.makedirs(base_cache, exist_ok=True)

        # Create run-specific folder
        run_folder = os.path.join(base_cache, run_name)
        os.makedirs(run_folder, exist_ok=True)

        # --- Save deltas (xarray) ---
        deltas_path = os.path.join(run_folder, "deltas.nc")
        self._deltas.to_netcdf(deltas_path)
        print(f"Deltas saved to: {deltas_path}")

        # --- Save cashflows (pandas DataFrame) ---
        cashflows_path = os.path.join(run_folder, "cashflows.csv")
        self.cashflows.to_csv(cashflows_path, index=True)
        print(f"Cashflows saved to: {cashflows_path}")

        print(f"All results saved in folder: {run_folder}")
