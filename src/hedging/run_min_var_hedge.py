import pandas as pd
import os

from delta_position.min_var_delta.polynomial_regression_delta.polynomial_regression_delta import \
    PolynomialRegressionDelta
from forward_curve.forward_curve import generate_yearly_seasonal_curve, ForwardCurve
from hedging.min_var_hedge import MinVarHedge
from hedging.ops_plot_min_var_hedge import OpsPlotMinVarHedge
from market_simulation.spread_model.spread_model import SpreadModel
from valuation.operations_states import OperationalState
from valuation.power_plant.power_plant import PowerPlant
from base_config import CONFIG_FOLDER_ENV


# Set the environment variable for this notebook session
os.environ[CONFIG_FOLDER_ENV] = "C:\Projects\power-plant-rl\config"


# Simulation configuration
config_path_spread_model = "model_configs/spread_model.json"
as_of_date = pd.Timestamp("2025-09-30")

simulation_start = as_of_date
simulation_end = pd.Timestamp("2026-12-31")
simulation_days = pd.date_range(start=simulation_start, end=simulation_end, freq="D")

n_sims = 1000


# Generate initial forward curves
power_fwd_0 = generate_yearly_seasonal_curve(
    as_of_date=as_of_date,
    start_date=simulation_start,
    end_date=simulation_end,
    winter_value=130,
    summer_value=40.0,
)

coal_fwd_0 = ForwardCurve.generate_curve(
    as_of_date=as_of_date,
    start_date=simulation_start,
    end_date=simulation_end,
    start_value=90,
    end_value=80.0,
    name="Coal Forward Curve",
)

# Initialize spread model
spread_model = SpreadModel(as_of_date, simulation_days, config_path=config_path_spread_model)

# Simulate spot and forward prices
power_fwd, power_month_ahead, power_spot, coal_fwd, coal_month_ahead, coal_spot = spread_model.simulate(
    power_fwd_0=power_fwd_0, coal_fwd_0=coal_fwd_0, n_sims=n_sims, use_cache=True
)


# Power plant simulation period
asset_start = pd.Timestamp(2025, 10, 1)
asset_end = asset_start + pd.Timedelta(days=364)
asset_days = pd.date_range(start=asset_start, end=asset_end, freq="D")

# Initial operational state
initial_state = OperationalState.IDLE

# Initialize PowerPlant
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

# Optimize plant operations
power_plant.optimize()


# Initialize hedging strategy
delta_calculator = PolynomialRegressionDelta()

hedge_calculator = MinVarHedge(
    n_sims=n_sims,
    simulation_days=simulation_days,
    power_plant=power_plant,
    spots_power=power_spot,
    spots_coal=coal_spot,
    fwds_power=power_fwd,
    fwds_coal=coal_fwd,
    delta_calculator=delta_calculator
)

# Execute hedge and calculate cashflows
hedge_calculator.hedge()
hedge_calculator.roll_out_cashflows()

# Plot the hedge effectiveness and R2
ops_plot_min_var_hedge = OpsPlotMinVarHedge(hedge_calculator)
ops_plot_min_var_hedge.plot_r2()
ops_plot_min_var_hedge.plot_hedge_effectiveness(power_plant.cashflows)
ops_plot_min_var_hedge.plot_mean_delta_positions()