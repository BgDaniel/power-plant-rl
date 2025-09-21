import os
import pickle
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

from forward_curve.forward_curve import generate_yearly_seasonal_curve, ForwardCurve
from market_simulation.constants import SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START
from market_simulation.spread_model.spread_model import SpreadModel
from market_simulation.two_factor_model.simulation_caching import CACHE_FOLDER_ENV
from regression.polynomial_regressor import PolynomialRegression, KEY_PREDICTED, KEY_R2
from valuation.operations_states import OperationalState
from valuation.power_plant.power_plant import PowerPlant
from valuation.power_plant.power_plant import POWER, COAL


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
        polynomial_type: str = PolynomialRegression.POLY_LEGENDRE,
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

        self._delivery_start_dates = fwds_power.coords[DELIVERY_START].values

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
            front_months_power = self._get_front_months(
                self._fwds_power, asset_day
            )
            front_months_coal = self._get_front_months(self._fwds_coal, asset_day)

            self._compute_delta(POWER, asset_day, front_months_power)

            self._compute_delta(COAL, asset_day, front_months_coal)

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

    def _get_front_months(
        self, fwds: xr.DataArray, simulation_day: pd.Timestamp
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

        return fwds.sel(
            {DELIVERY_START: slice(start, end), SIMULATION_DAY: simulation_day}
        )

    def _compute_delta(
        self, asset: str, asset_day: pd.Timestamp, front_months_fwds: xr.DataArray
    ) -> None:
        """
        Compute hedge deltas for the given asset_day using polynomial regression
        between the simulated cashflows and forward prices.

        Parameters
        ----------
        asset_day : pd.Timestamp
            Current simulation day.
        front_months_fwds : xr.DataArray
            Forward prices (n_sims, delivery_months) at asset_day.

        Returns
        -------
        xr.DataArray
            Hedge deltas (n_sims, delivery_months) for this day.
        """
        # --- get cashflows of the plant at this day ---
        front_months_start_dates = front_months_fwds.coords[DELIVERY_START].values

        for delivery_start in front_months_start_dates:

            self._deltas.loc[
                {
                    SIMULATION_DAY: asset_day,
                    DELIVERY_START: delivery_start,
                    ASSET: asset,
                }
            ] = self._hedge_front_month(
                asset_day, asset, delivery_start, front_months_fwds.to_pandas()
            )

    def _hedge_front_month(
        self,
        asset_day: pd.Timestamp,
        asset: str,
        delivery_start: str,
        front_months_fwds: pd.DataFrame,
        r2_threshold=0.0,
    ):

        days_in_front_month: pd.DatetimeIndex = pd.date_range(
            start=delivery_start, end=delivery_start + pd.offsets.MonthEnd(0)
        )

        # Asset days sometimes could be only partially covered by the front month days
        # at the end of the asset computatin horizom
        asset_days_in_front_month = self._cashflows.index.intersection(days_in_front_month)

        cashflows_from_asset = self._cashflows.loc[asset_days_in_front_month].values

        alpha = cashflows_from_asset.sum(axis=0)  # shape (n_sims,)

        beta = (
            self._spots[asset].loc[days_in_front_month].values.sum(axis=0)
        )  # shape (n_sims,)

        y = (
            alpha / beta
        )  # the sum of the spots over the delivery months are always non zero

        # Construct features: beta_i * gamma_i^k for k=0..degree
        x = front_months_fwds.values

        # Fit polynomial regression
        poly_reg = PolynomialRegression(
            x=x, y=y, degree=self._polynomial_degree, poly_type=self._polynomial_type
        )

        # Perform regression
        results = poly_reg.regress()
        delta_positions = results[KEY_PREDICTED]
        r2_score = results[KEY_R2]

        # Check R² threshold
        if r2_score < r2_threshold:
            delivery_month = delivery_start.strftime("%Y-%m")
            raise ValueError(
                f"Regression R²={r2_score:.4f} below threshold {r2_threshold} "
                f"for underlying {asset} and the delivery month {delivery_month}."
            )

        # Store R² using .sel
        self._r2_scores.loc[
            dict(
                SIMULATION_DAY=asset_day,
                DELIVERY_START=delivery_start,
                ASSET=asset
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

    def plot_r2(self):
        """
        Plot R² values over simulation days for both POWER and COAL in a two-row plot,
        with each delivery start date shown as a separate line.
        """
        assets = [POWER, COAL]
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for ax, asset in zip(axes, assets):
            asset_index = 0 if asset == POWER else 1
            for delivery_index, delivery_start in enumerate(self._delivery_start_dates):
                r2_values = self._r2_scores[:, delivery_index, asset_index].values
                ax.plot(
                    self._simulation_days,
                    r2_values,
                    label=f"Delivery {pd.Timestamp(delivery_start).strftime('%Y-%m')}"
                )
            ax.set_ylabel("R² Value")
            ax.set_title(f"{asset} R² over time")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        axes[1].set_xlabel("Simulation Day")
        plt.tight_layout()
        plt.show(block=True)


if __name__ == "__main__":
    config_path_spread_model = "model_configs/spread_model.json"

    as_of_date = pd.Timestamp("2025-09-30")

    simulation_start = as_of_date
    simulation_end = pd.Timestamp("2026-12-30")

    simulation_days = pd.date_range(
        start=simulation_start, end=simulation_end, freq="D"
    )

    spread_model = SpreadModel(
        as_of_date, simulation_days, config_path=config_path_spread_model
    )

    n_sims = 1000

    power_fwd_0 = generate_yearly_seasonal_curve(
        as_of_date=as_of_date,
        start_date=simulation_start,
        end_date=simulation_end,
        winter_value=120,
        summer_value=60.0,
    )

    coal_fwd_0 = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=simulation_start,
        end_date=simulation_end,
        start_value=90.0,
        end_value=70.0,
        name="Coal Forward Curve",
    )

    (
        power_fwd,
        power_month_ahead,
        power_spot,
        coal_fwd,
        coal_month_ahead,
        coal_spot,
    ) = spread_model.simulate(
        power_fwd_0=power_fwd_0, coal_fwd_0=coal_fwd_0, n_sims=n_sims, use_cache=True
    )

    asset_start = pd.Timestamp(2025, 10, 1)
    asset_end = simulation_start + pd.Timedelta(days=365)

    asset_days = pd.date_range(start=asset_start, end=asset_end, freq="D")

    initial_state = OperationalState.IDLE

    power_plant = PowerPlant(
        n_sims=n_sims,
        asset_days=asset_days,
        initial_state=initial_state,
        spots_power=power_spot,
        spots_coal=coal_spot,
        fwds_power=power_fwd,
        fwds_coal=coal_fwd,
        fwd_0_power=power_fwd_0,
        fwd_0_coal=coal_fwd_0,
        config_path="asset_configs/power_plant_config.yaml",
    )

    # Run the simulation
    power_plant.optimize()

    min_var_hedge = MinVarHedge(
        n_sims=n_sims,
        simulation_days=simulation_days,
        power_plant=power_plant,
        spots_power=power_spot,
        spots_coal=coal_spot,
        fwds_power=power_fwd,
        fwds_coal=coal_fwd,
    )

    min_var_hedge.hedge()

    min_var_hedge.save_results(run_name='hedge_run_oct_25-oct26')
    min_var_hedge.plot_r2()
