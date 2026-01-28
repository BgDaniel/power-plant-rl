from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from ml_hedging.nn_delta.feature_builder import FeatureBuilder
from delta.min_var_delta.min_var_delta import MinVarDeltaCalculator
from delta.delta_calculator import DeltaCalculator
from valuation.power_plant.power_plant import PowerPlant
from delta.hedge_calculator import HedgeCalculator  # or your MinVarHedge wrapper


class SampleGenerator:
    """
    Generates training samples (features, targets) for NN delta.

    Steps:
    1. Run Monte Carlo simulation of spot and forwards.
    2. Run MinVar hedge to compute target deltas.
    3. Build input features for each delivery month.
    """

    def __init__(
        self,
        power_plant: PowerPlant,
        n_sims: int,
        efficiency: float,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.power_plant = power_plant
        self.n_sims = n_sims
        self.efficiency = efficiency
        self.feature_builder = feature_builder or FeatureBuilder(efficiency)

    def generate(
        self,
        simulation_days: pd.DatetimeIndex,
        power_fwd: xr.DataArray,
        coal_fwd: xr.DataArray,
        power_spot: xr.DataArray,
        coal_spot: xr.DataArray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
        delivery_months: Optional[np.ndarray] = None,
    ):
        """
        Generate features and targets for NN training.

        Parameters
        ----------
        simulation_days : pd.DatetimeIndex
            Simulation time steps
        power_fwd, coal_fwd : xr.DataArray
            Forward curves (n_sims, n_days, n_delivery_months)
        power_spot, coal_spot : xr.DataArray
            Simulated spot paths (n_days, n_sims)
        beta_power, beta_coal : np.ndarray
            Beta exposures (n_sims,)
        delivery_months : np.ndarray, optional
            Array of delivery start dates to generate samples for.
            Defaults to all months in forward curve.

        Returns
        -------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Targets (n_samples, 2) -> delta_power, delta_coal
        """
        # Initialize HedgeCalculator to compute MinVar deltas
        hedge = HedgeCalculator(
            n_sims=self.n_sims,
            simulation_days=simulation_days,
            power_plant=self.power_plant,
            spots_power=power_spot,
            spots_coal=coal_spot,
            fwds_power=power_fwd,
            fwds_coal=coal_fwd,
            delta_calculator=MinVarDeltaCalculator,
        )

        # Run the MinVar hedge
        hedge.hedge()

        # Decide delivery months
        if delivery_months is None:
            delivery_months = hedge._delivery_start_dates

        X_list = []
        y_list = []

        for t in delivery_months:
            # Extract forward prices for this delivery month
            fwd_p = power_fwd.sel(delivery_start=t).values
            fwd_c = coal_fwd.sel(delivery_start=t).values

            # Extract delta targets for this month
            delta_res = hedge.deltas.sel(delivery_start=t)
            delta_targets = np.stack([
                delta_res.sel(asset='POWER').values.flatten(),
                delta_res.sel(asset='COAL').values.flatten()
            ], axis=1)

            # Build input features
            features = self.feature_builder.build(
                fwd_power=fwd_p,
                fwd_coal=fwd_c,
                beta_power=beta_power,
                beta_coal=beta_coal,
                t=float(pd.Timestamp(t).toordinal())  # convert date to numeric
            )

            X_list.append(features)
            y_list.append(delta_targets)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import xarray as xr
    from valuation.power_plant.power_plant import PowerPlant
    from sample_generator import SampleGenerator

    # ----------------------------
    # Initialize a simple PowerPlant
    # ----------------------------
    asset_days = pd.date_range("2026-01-01", periods=5)
    power_plant = PowerPlant(asset_days=asset_days, efficiency=0.4)

    n_sims = 10
    simulation_days = pd.date_range("2026-01-01", periods=5)

    # ----------------------------
    # Dummy forward & spot data
    # ----------------------------
    power_fwd = xr.DataArray(np.random.rand(n_sims, 5, 1),
                             dims=["simulation_path", "simulation_day", "delivery_start"],
                             coords={"simulation_path": np.arange(n_sims),
                                     "simulation_day": simulation_days,
                                     "delivery_start": [simulation_days[0]]})
    coal_fwd = xr.DataArray(np.random.rand(n_sims, 5, 1),
                            dims=["simulation_path", "simulation_day", "delivery_start"],
                            coords={"simulation_path": np.arange(n_sims),
                                    "simulation_day": simulation_days,
                                    "delivery_start": [simulation_days[0]]})
    power_spot = xr.DataArray(np.random.rand(n_sims, 5),
                              dims=["simulation_path", "simulation_day"],
                              coords={"simulation_path": np.arange(n_sims),
                                      "simulation_day": simulation_days})
    coal_spot = xr.DataArray(np.random.rand(n_sims, 5),
                             dims=["simulation_path", "simulation_day"],
                             coords={"simulation_path": np.arange(n_sims),
                                     "simulation_day": simulation_days})

    # ----------------------------
    # Generate training samples
    # ----------------------------
    generator = SampleGenerator(power_plant, n_sims=n_sims, efficiency=0.4)
    X, y = generator.generate(
        simulation_days,
        power_fwd,
        coal_fwd,
        power_spot,
        coal_spot,
        beta_power=np.ones(n_sims),
        beta_coal=np.ones(n_sims)
    )

    print("Feature matrix shape:", X.shape)
    print("Target (delta) shape:", y.shape)