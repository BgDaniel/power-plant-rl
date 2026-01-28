from typing import Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from delta_position.min_var_delta.nn_delta.feature_builder import FeatureBuilder
from delta_position.min_var_delta.polynomial_regression_delta.polynomial_regression_delta import PolynomialRegressionDelta
from valuation.power_plant.power_plant import PowerPlant
from delta_position.min_var_delta import HedgeCalculator  # or your MinVarHedge wrapper


class SampleGenerator:
    """
    Generates training samples (features, targets, beta) for NN polynomial_regression_delta.

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
        """
        Initialize SampleGenerator.

        Parameters
        ----------
        power_plant : PowerPlant
            The power plant object used for simulation.
        n_sims : int
            Number of Monte Carlo simulation paths.
        efficiency : float
            Efficiency factor used in spread calculation.
        feature_builder : FeatureBuilder, optional
            Optional custom feature builder. If None, uses default.
        """
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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            delta_calculator=PolynomialRegressionDelta,
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

            # Extract polynomial_regression_delta targets for this month
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

    def get_training_set(
        self,
        simulation_days: pd.DatetimeIndex,
        power_fwd: xr.DataArray,
        coal_fwd: xr.DataArray,
        power_spot: xr.DataArray,
        coal_spot: xr.DataArray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
        delivery_months: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the full training set including features, targets, and beta weights.

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

        Returns
        -------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Targets (n_samples, 2)
        beta_power_full : np.ndarray
            Expanded beta_power for each sample (n_samples,)
        beta_coal_full : np.ndarray
            Expanded beta_coal for each sample (n_samples,)
        """
        X, y = self.generate(
            simulation_days,
            power_fwd,
            coal_fwd,
            power_spot,
            coal_spot,
            beta_power,
            beta_coal,
            delivery_months
        )

        # Repeat beta arrays for each delivery month
        n_repeats = X.shape[0] // beta_power.shape[0]
        beta_power_full = np.tile(beta_power, n_repeats)
        beta_coal_full = np.tile(beta_coal, n_repeats)

        return X, y, beta_power_full, beta_coal_full
