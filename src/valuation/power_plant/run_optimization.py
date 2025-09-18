import pandas as pd

from forward_curve.forward_curve import ForwardCurve
from market_simulation.spread_model.spread_model import SpreadModel
from valuation.operations_states import OperationalState
from valuation.power_plant.ops_plot import OpsPlot
from valuation.power_plant.power_plant import PowerPlant

config_path_spread_model = "model_configs/spread_model.json"

as_of_date = pd.Timestamp("2025-09-13")
n_days = 365

simulation_start = as_of_date
simulation_end = simulation_start + pd.Timedelta(days=365 + 31)

simulation_days = pd.date_range(start=simulation_start, end=simulation_end, freq="D")

spread_model = SpreadModel(
    as_of_date, simulation_days, config_path=config_path_spread_model
)

n_sims = 1000

power_fwd_0 = ForwardCurve.generate_curve(
    as_of_date=as_of_date,
    start_date=simulation_start,
    end_date=simulation_end,
    start_value=80.0,
    end_value=110.0,
    name="Power Forward Curve",
)

coal_fwd_0 = ForwardCurve.generate_curve(
    as_of_date=as_of_date,
    start_date=simulation_start,
    end_date=simulation_end,
    start_value=120.0,
    end_value=70.0,
    name="Coal Forward Curve",
)

(power_fwd,
power_month_ahead,
power_spot,
coal_fwd,
coal_month_ahead,
coal_spot) = spread_model.simulate(
    power_fwd_0=power_fwd_0, coal_fwd_0=coal_fwd_0, n_sims=n_sims, use_cache=True
)

asset_start = simulation_start
asset_end = simulation_start + pd.Timedelta(days=365)

asset_days = pd.date_range(start=asset_start, end=asset_end, freq="D")

initial_state = OperationalState.IDLE

power_plant = PowerPlant(
    n_sims=n_sims,
    asset_days=asset_days,
    initial_state=initial_state ,
    spots_power=power_spot,
    spots_coal=coal_spot,
    fwds_power= power_fwd,
    fwds_coal=coal_fwd,
    fwd_0_power=power_fwd_0,
    fwd_0_coal=coal_fwd_0,
    config_path="asset_configs/power_plant_config.yaml",
)

# Run the simulation
power_plant.optimize()

ops_plot = OpsPlot(power_plant)

#ops_plot.plot_r2()
ops_plot.plot_simulation_summary(path_index=0)
