import pandas as pd


from forward_curve.forward_curve import generate_yearly_seasonal_curve, ForwardCurve
from hedging.min_var_hedge import MinVarHedge
from hedging.ops_plot_min_var_hedge import OpsPlotMinVarHedge
from market_simulation.spread_model.spread_model import SpreadModel
from valuation.operations_states import OperationalState
from valuation.power_plant.ops_plot_power_plant import OpsPlotPowerPlant
from valuation.power_plant.power_plant import PowerPlant


config_path_spread_model = "model_configs/spread_model.json"

as_of_date = pd.Timestamp("2025-09-30")

simulation_start = as_of_date
simulation_end = pd.Timestamp("2026-12-30")

simulation_days = pd.date_range(start=simulation_start, end=simulation_end, freq="D")

spread_model = SpreadModel(
    as_of_date, simulation_days, config_path=config_path_spread_model
)

n_sims = 1000

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

ops_plot_power_plant = OpsPlotPowerPlant(power_plant)
# ops_plot_power_plant.plot_r2()
# ops_plot_power_plant.plot_simulation_summary(path_index=0)

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
# min_var_hedge.save_results(run_name='hedge_run_oct_25-oct26')

min_var_hedge.roll_out_cashflows()

ops_plot_min_var_hedge = OpsPlotMinVarHedge(min_var_hedge)
ops_plot_min_var_hedge.plot_r2()
ops_plot_min_var_hedge.plot_hedge_effectiveness(power_plant.cashflows)
