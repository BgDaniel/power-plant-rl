import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

from regression.min_var_regression import MinVarPolynomialRegression
from market_simulation.constants import SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START
from market_simulation.two_factor_model.simulation_caching import CACHE_FOLDER_ENV
from regression.polynomial_base_builder import PolynomialBasisBuilder

from valuation.power_plant.power_plant import PowerPlant
from valuation.power_plant.power_plant import POWER, COAL

from regression.regression_helpers import KEY_R2, KEY_PREDICTED


DELTA_POWER = "DELTA_POWER"
DELTA_COAL = "DELTA_COAL"
ASSET = "ASSET"
DELTA_POSITION = "DELTA_POSITION"


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
    ):
        self._n_sims = n_sims
        self._simulation_days = simulation_days
        self._n_steps = len(simulation_days)

        self.as_of_date = simulation_days[0]

        self._power_plant = power_plant

        self._asset_days = power_plant.asset_days

        self._cashflows = self._power_plant.optimal_cashflows

        self._spots_power = spots_power
        self._spots_coal = spots_coal
        self._fwds_power = fwds_power
        self._fwds_coal = fwds_coal

        self._spots = {POWER: self._spots_power, COAL: self._spots_coal}

        delivery_start_dates_power = fwds_power.coords[DELIVERY_START].values
        delivery_start_dates_coal = fwds_coal.coords[DELIVERY_START].values

        # Check equality and raise error if they differ
        if not np.array_equal(delivery_start_dates_power , delivery_start_dates_coal):
            raise ValueError(
                "DELIVERY_START coordinates of fwds_power and fwds_coal are not equal!"
            )

        self._delivery_start_dates = delivery_start_dates_power

        self._n_front_months = len(self._delivery_start_dates)

        self._polynomial_type = polynomial_type
        self._polynomial_degree = polynomial_degree

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

    def hedge(self):
        self.tqdm = tqdm(
            self._asset_days,
            total=len(self._asset_days) - 1,
            desc="Determine minimal variance hedge",
            unit="day",
        )
        for asset_day in self.tqdm:
            front_months_start_dates = self._get_front_months_start_dates(
                asset_day
            )
            self._compute_delta(asset_day, front_months_start_dates)

            self._roll_out_cashflows(asset_day)

    def _roll_out_cashflows(self, asset_day: pd.Timestamp):
        bom_delivery_start_day = asset_day - pd.offsets.MonthBegin(n=0)

        # look for terminal delta of current BOM

        bom_maturity_day = bom_delivery_start_day + pd.Timedelta(days=-1)

        for asset in [POWER, COAL]:

            bom_delta_asset = self._deltas.sel({
                DELIVERY_START: bom_delivery_start_day,
                ASSET: asset,
                SIMULATION_DAY: bom_maturity_day
            })

            spots = self._spots_power.loc[asset_day] if asset == POWER else self._spots_coal.loc[asset_day].values

            # hedge poistion are struck with strike price zero
            self.cashflows.loc[
                asset_day
            ] += bom_delta_asset* spots

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

        mask = (self._delivery_start_dates >= start) & \
               (self._delivery_start_dates <= end)

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
            ] = self._hedge_front_month(
                asset_day, delivery_start
            )

    def _hedge_front_month(
            self,
            asset_day: pd.Timestamp,
            delivery_start: pd.Timestamp,
            r2_threshold: float = 0.0
    ) -> np.ndarray:
        """
        Compute hedge delta for a single delivery month using polynomial regression.

        Parameters
        ----------
        asset_day : pd.Timestamp
            The current simulation day for which the hedge is computed.
        asset : str
            Either POWER or COAL, representing the asset type.
        delivery_start : str
            The first day of the delivery month.
        front_months_fwds_power : pd.DataFrame
            Forward prices for this delivery month (shape: n_sims x n_delivery_days).
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
        asset_days_in_front_month = self._cashflows.index.intersection(days_in_front_month)

        cashflows_from_asset = self._cashflows.loc[asset_days_in_front_month].values
        cashflow_sum_from_asset = cashflows_from_asset.sum(axis=0)  # shape (n_sims,)

        beta_power = (
            self._spots[POWER].loc[days_in_front_month].values.sum(axis=0)
        )  # shape (n_sims,)

        beta_coal = (
            self._spots[COAL].loc[days_in_front_month].values.sum(axis=0)
        )  # shape (n_sims,)

        fwds_power_front_months = self._fwds_power.sel({DELIVERY_START: delivery_start, SIMULATION_DAY: asset_day}).to_pandas()
        fwds_coal_front_months = self._fwds_coal.sel(
            {DELIVERY_START: delivery_start, SIMULATION_DAY: asset_day}).to_pandas()

        x_fwd_power = fwds_power_front_months.values
        x_fwd_coal = fwds_coal_front_months.values
        x_dark_spread = x_fwd_power - self._power_plant.efficiency * x_fwd_coal

        y = cashflow_sum_from_asset  # target cashflows

        # --- Fit minimum-variance polynomial regression -------------------------
        min_var_reg = MinVarPolynomialRegression(n_samples=self._n_sims, x_fwd_power=x_fwd_power,x_fwd_coal=x_fwd_coal,
        x_spread=x_dark_spread, y=y, beta_power=beta_power, beta_coal=beta_coal, degree=self._polynomial_degree,
            poly_type=self._polynomial_type)

        # Perform regression
        results = min_var_reg.regress()
        delta_positions = results[KEY_PREDICTED]
        r2_score = results[KEY_R2]

        # --- Validation ---------------------------------------------------------
        # Check R² threshold
        #if r2_score < r2_threshold:
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

