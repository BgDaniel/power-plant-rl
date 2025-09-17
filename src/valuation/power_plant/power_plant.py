import xarray as xr
from typing import Optional
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import time

from forward_curve.forward_curve import ForwardCurve
from market_simulation.spread_model.spread_model import SpreadModel
from valuation.operations_states import OperationalState
from valuation.operational_control import OptimalControl

from valuation.power_plant.power_plant_config import (
    PowerPlantConfigLoader,
    PowerPlantConfig,
)
from valuation.regression.polynomial_regressor import (
    PolynomialRegression,
    KEY_PREDICTED,
    KEY_R2,
    KEY_CONDITION_NUMBER,
)


logger = logging.getLogger(__name__)


# -----------------------
# Dimension name constants
# -----------------------
DIM_SIMULATION_DAY = "simulation_day"
DIM_SIMULATION_PATH = "simulation_path"
DIM_OPERATIONAL_STATE = "operational_state"


OPTIMAL_CONTROL = "optimal_control"
VALUE = "value"
CASHFLOWS = "cashflows"
REGRESSED_VALUE = "regressed_value"
REGRESSION_RESULTS = "regression_results"


class PowerPlant:
    """
    Class representing a power plant with operational costs, ramping constraints,
    and market data inputs (forward/day-ahead prices).

    Attributes
    ----------
    n_sims : int
        Number of Monte Carlo simulation paths.
    simulation_days : pd.DatetimeIndex
        The timeline of the simulation (one entry per day).
    n_steps : int
        Number of time steps in the simulation (= len(simulation_days)).

    cashflows : xr.DataArray
        Cashflows for each (day, simulation path, operational state).
        Shape: (n_steps, n_sims, 4).
    value : xr.DataArray
        Value of the power plant for each (day, simulation path, operational state).
        Shape: (n_steps, n_sims, 4).
    optimal_control_decision : xr.DataArray
        Optimal control decision for each (day, simulation path, operational state).
        Shape: (n_steps - 1, n_sims, 4).
    states : list[str]
        List of operational states as lowercase strings, aligned with `OperationalState`.
    """

    def __init__(
        self,
        n_sims: int,
        asset_days: pd.DatetimeIndex,
        fwds_power: xr.DataArray,
        fwds_coal: xr.DataArray,
        spots_power: pd.DataFrame,
        spots_coal: pd.DataFrame,
        config: Optional[PowerPlantConfig] = None,
        config_path: Optional[str] = None,
        polynomial_type: str = PolynomialRegression.POLY_LEGENDRE,
        polynomial_degree: int = 3,
    ) -> None:
        """
        Initialize a PowerPlant instance.

        Parameters
        ----------
        n_sims : int
            Number of Monte Carlo simulation paths.
        asset_days : pd.DatetimeIndex
            The timeline of the simulation (one entry per day).
        fwds_power : xr.DataArray
            Forward prices for power (simulated).
        fwds_coal : xr.DataArray
            Forward prices for coal (simulated).
        spots_power : pd.DataFrame
            Day-ahead power prices (simulated).
        spots_coal : pd.DataFrame
            Day-ahead coal prices (simulated).
        config : PowerPlantConfig, optional
            Operational parameter configuration.
        config_path : str, optional
            Path to YAML config file (if config is not directly provided).
        """
        if config is None:
            if config_path is not None:
                config = PowerPlantConfigLoader.from_yaml(config_path)
            else:
                raise ValueError("Either config or config_path must be provided.")

        # Simulation setup
        self.n_sims: int = n_sims
        self.simulation_days: pd.DatetimeIndex = asset_days
        self.n_steps: int = len(asset_days)

        # Market data
        self.power_fwd_prices: xr.DataArray = fwds_power
        self.coal_fwd_prices: xr.DataArray = fwds_coal
        self.power_day_ahead_prices: pd.DataFrame = spots_power
        self.coal_day_ahead_prices: pd.DataFrame = spots_coal

        # Operational parameters
        self.config: "PowerPlantConfig" = config
        self.operation_costs: float = config.operation_costs
        self.efficiency: float = config.efficiency
        self.ramping_up_costs: float = config.ramping_up_costs
        self.ramping_down_costs: float = config.ramping_down_costs
        self.n_days_ramping_up: int = config.n_days_ramping_up
        self.n_days_ramping_down: int = config.n_days_ramping_down
        self.idle_costs: float = config.idle_costs

        self.polynomial_type = polynomial_type
        self.polynomial_degree = polynomial_degree

        # Operational states (string labels, consistent with enum)
        self.states: list[OperationalState] = list(OperationalState)

        # -----------------------
        # Cashflows
        # -----------------------
        self.cashflows: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.simulation_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.states,
            },
            name=CASHFLOWS,
        )

        # -----------------------
        # Values
        # -----------------------
        self.value: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.simulation_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.states,
            },
            name=VALUE,
        )

        # -----------------------
        # Optimal control decisions
        # -----------------------
        self.optimal_control: xr.DataArray = xr.DataArray(
            np.full(
                (self.n_steps - 1, self.n_sims, len(self.states)),
                OptimalControl.DO_NOTHING,
            ),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.simulation_days[
                    :-1
                ],  # no decision on last day
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.states,
            },
            name=OPTIMAL_CONTROL,
        )

        self.regressed_value: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.simulation_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.states,
            },
            name=REGRESSED_VALUE,
        )

        self.regression_results: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.simulation_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.states,
            },
            name=REGRESSION_RESULTS,
        )

        # Compute spread: power - efficiency * coal
        self.spread: pd.DataFrame = spots_power.sub(self.efficiency * spots_coal)

    def _initialize_terminal_state(self) -> None:
        """
        Initialize terminal state (end of simulation).
        Sets cashflows and values for the last day across all operational states.
        """
        last_day = self.simulation_days[-1]
        spread_last = self.spread.loc[last_day]

        self.cashflows.loc[
            {DIM_SIMULATION_DAY: last_day, DIM_OPERATIONAL_STATE: OperationalState.IDLE}
        ] = -self.idle_costs
        self.cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RAMPING_UP,
            }
        ] = -self.ramping_up_costs
        self.cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RAMPING_DOWN,
            }
        ] = -self.ramping_down_costs
        self.cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RUNNING,
            }
        ] = (
            -self.operation_costs + self.efficiency * spread_last.values
        )

        self.value.loc[{DIM_SIMULATION_DAY: last_day}] = self.cashflows.loc[
            {DIM_SIMULATION_DAY: last_day}
        ]

    def _optimize(self, simulation_day: pd.Timestamp) -> None:
        """
        Perform backward induction optimization for a single simulation day.
        It updates the cashflows, values, and optimal control decisions for each
        operational state.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to optimize.
        """
        first_day = self.simulation_days[0]

        if simulation_day == first_day:
            # Special case: first day, no regression
            next_day = self.simulation_days[1]
            for state in OperationalState:
                mean_value = (
                    self.value.sel(simulation_day=next_day, operational_state=state)
                    .mean(dim="simulation_path")
                    .values
                )
                # Broadcast the mean to all simulation paths
                self.regressed_value.loc[
                    dict(simulation_day=simulation_day, operational_state=state)
                ] = np.full(self.n_sims, mean_value)
        else:
            # Normal case: regress for all states
            for state in OperationalState:
                self.regressed_value.loc[
                    dict(simulation_day=simulation_day, operational_state=state)
                ] = self._regress(simulation_day, state)

        # Optimize each state separately
        self._optimize_idle(simulation_day)
        self._optimize_ramping_up(simulation_day)
        self._optimize_ramping_down(simulation_day)
        self._optimize_running(simulation_day)

    def _optimize_idle(self, simulation_day: pd.Timestamp) -> None:
        """
        Optimize the idle state for a single simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to optimize.
        """
        idle = OperationalState.IDLE
        ramp_up = OperationalState.RAMPING_UP

        cont_val = self.regressed_value.loc[
            dict(simulation_day=simulation_day, operational_state=idle)
        ]
        exer_val = self.regressed_value.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ]

        self.cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=idle)
        ] = -self.idle_costs
        self.value.loc[dict(simulation_day=simulation_day, operational_state=idle)] = (
            -self.idle_costs + np.maximum(cont_val, exer_val)
        )
        self.optimal_control.loc[
            dict(simulation_day=simulation_day, operational_state=idle)
        ] = np.where(
            cont_val > exer_val, OptimalControl.DO_NOTHING, OptimalControl.RAMPING_UP
        )

    def _optimize_ramping_up(self, simulation_day: pd.Timestamp) -> None:
        """
        Optimize the ramping-up state for a single simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to optimize.
        """
        ramp_up = OperationalState.RAMPING_UP
        running = OperationalState.RUNNING

        self.cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ] = -self.ramping_up_costs
        self.value.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ] = (
            -self.ramping_up_costs
            + self.regressed_value.loc[
                dict(simulation_day=simulation_day, operational_state=running)
            ]
        )
        # No control decision to be made for ramping up

    def _optimize_ramping_down(self, simulation_day: pd.Timestamp) -> None:
        """
        Optimize the ramping-down state for a single simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to optimize.
        """
        ramp_down = OperationalState.RAMPING_DOWN
        idle = OperationalState.IDLE

        self.cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ] = -self.ramping_down_costs
        self.value.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ] = (
            -self.ramping_down_costs
            + self.regressed_value.loc[
                dict(simulation_day=simulation_day, operational_state=idle)
            ]
        )
        # No control decision to be made for ramping down

    def _optimize_running(self, simulation_day: pd.Timestamp) -> None:
        """
        Optimize the running state for a single simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to optimize.
        """
        running = OperationalState.RUNNING
        ramp_down = OperationalState.RAMPING_DOWN

        cont_val = self.regressed_value.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ]
        exer_val = self.regressed_value.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ]

        self.cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = (-self.operation_costs + self.efficiency * self.spread.loc[simulation_day])
        self.value.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = (
            -self.operation_costs
            + self.efficiency * self.spread.loc[simulation_day]
            + np.maximum(cont_val, exer_val)
        )
        self.optimal_control.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = np.where(
            cont_val > exer_val, OptimalControl.DO_NOTHING, OptimalControl.RAMPING_DOWN
        )

    def _regress(
        self,
        simulation_day: pd.Timestamp,
        state: OperationalState,
        r2_threshold: float = 0.9,
    ) -> np.ndarray:
        """
        Regress the optimal value for a given day and operational state
        against the polynomial features of day-ahead power and coal prices.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day to perform regression on.
        state : OperationalState
            The operational state (IDLE, RAMPING_UP, RAMPING_DOWN, RUNNING).
        r2_threshold : float, default=0.8
            Minimum acceptable R² score; an exception is raised if the regression
            falls below this threshold.

        Returns
        -------
        np.ndarray
            Regressed optimal values for all simulation paths on the given day.

        Raises
        ------
        ValueError
            If the R² score of the regression is below the specified threshold.
        """
        # Extract day-ahead power and coal prices for the given day
        day_ahead_power = self.power_day_ahead_prices.loc[simulation_day].values
        day_ahead_coal = self.coal_day_ahead_prices.loc[simulation_day].values

        # Stack features
        x = np.vstack([day_ahead_power, day_ahead_coal]).T

        # Extract target values: optimal value for the next day
        y = (
            self.value.sel(simulation_day=simulation_day + pd.Timedelta(days=1))
            .sel(operational_state=state)
            .values
        )

        # Initialize PolynomialRegression with features, target, degree, and polynomial type
        poly_reg = PolynomialRegression(
            x=x,
            y=y,
            degree=self.polynomial_degree,
            poly_type=self.polynomial_type,
        )

        # Perform regression
        results = poly_reg.regress()
        regressed_value = results[KEY_PREDICTED]
        r2_score = results[KEY_R2]

        # Logging
        logger.info(
            f"Regression for {state.name} on {simulation_day.date()}: "
            f"R²={r2_score:.4f}, Condition number={results[KEY_CONDITION_NUMBER]:.2e}"
        )

        # Check R² threshold
        if r2_score < r2_threshold:
            raise ValueError(
                f"Regression R²={r2_score:.4f} below threshold {r2_threshold} "
                f"for state {state.name} on {simulation_day.date()}"
            )

        # Store regression results in the class xarray field
        self.regression_results.loc[
            dict(simulation_day=simulation_day, operational_state=state)
        ] = regressed_value

        return regressed_value

    def _update_r2_statistics(self, simulation_day: pd.Timestamp) -> None:
        """
        Update the minimal and maximal R² statistics for the given simulation day.

        Parameters
        ----------
        simulation_day : pd.Timestamp
            The simulation day for which to check R² values.
        """
        for state in OperationalState:
            r2_value = self.regression_results.sel(
                simulation_day=simulation_day, operational_state=state
            ).attrs.get("r2", None)

            if r2_value is not None:
                if r2_value < self._min_r2:
                    self._min_r2 = r2_value
                    self._min_r2_day = (simulation_day, state)
                if r2_value > self._max_r2:
                    self._max_r2 = r2_value
                    self._max_r2_day = (simulation_day, state)

    def optimize(self) -> None:
        """
        Optimize the power plant operations using backward induction.

        This method first initializes the terminal state (last day of simulation).
        Then it iterates in reverse over all simulation days, starting from the
        second-to-last day, and applies the optimization logic for each day.

        Workflow
        --------
        1. Initialize terminal state (last day).
        2. Loop backward through all previous days.
        3. Apply `_optimize` for each simulation day.

        Modifies
        --------
        self.cashflows : xr.DataArray
            Updated cashflows for each day, simulation path, and operational state.
        self.value : xr.DataArray
            Updated values for each day, simulation path, and operational state.
        self.optimal_control : xr.DataArray
            Updated optimal control decisions for each day, path, and state.
        """
        logging.info("Starting power plant optimization...")
        start_time = time.time()

        self._initialize_terminal_state()

        for simulation_day in tqdm(
            reversed(self.simulation_days[:-1]),
            total=len(self.simulation_days) - 1,
            desc="Optimizing simulation days",
            unit="day",
        ):
            self._optimize(simulation_day)

        elapsed_time = time.time() - start_time
        logging.info(
            "Power plant optimization successfully finished. "
            f"Elapsed time: {elapsed_time:.2f} seconds."
        )


# Example usage of the PowerPlant class:
if __name__ == "__main__":
    config_path_spread_model = "model_configs/spread_model.json"

    as_of_date = pd.Timestamp("2025-09-13")
    n_days = 365

    simulation_start = as_of_date
    simulation_end = simulation_start + pd.Timedelta(days=365 + 31)

    simulation_days = pd.date_range(
        start=simulation_start, end=simulation_end, freq="D"
    )

    spread_model = SpreadModel(
        as_of_date, simulation_days, config_path=config_path_spread_model
    )

    n_sims = 5000

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
        start_value=60.0,
        end_value=85.0,
        name="Coal Forward Curve",
    )

    power_fwd, _, power_day_ahead, coal_fwd, _, coal_day_ahead = spread_model.simulate(
        power_fwd_0=power_fwd_0, coal_fwd_0=coal_fwd_0, n_sims=n_sims
    )

    asset_start = simulation_start
    asset_end = simulation_start + pd.Timedelta(days=365)

    asset_days = pd.date_range(start=asset_start, end=asset_end, freq="D")

    power_plant = PowerPlant(
        n_sims=n_sims,
        asset_days=asset_days,
        fwds_power=power_fwd,
        fwds_coal=coal_fwd,
        spots_power=power_day_ahead,
        spots_coal=coal_day_ahead,
        config_path="asset_configs/power_plant_config.yaml",
    )

    # Run the simulation
    power_plant.optimize()
