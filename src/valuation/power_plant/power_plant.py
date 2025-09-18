import xarray as xr
from typing import Optional
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import time
from typing import Tuple

from forward_curve.forward_curve import ForwardCurve
from market_simulation.constants import DELIVERY_START, SIMULATION_DAY
from valuation.operations_states import OperationalState, get_next_state
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
logger.setLevel(logging.INFO)


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
    asset_days : pd.DatetimeIndex
        The timeline of the simulation (one entry per day).
    n_steps : int
        Number of time steps in the simulation (= len(simulation_days)).

    _cashflows : xr.DataArray
        Cashflows for each (day, simulation path, operational state).
        Shape: (n_steps, n_sims, 4).
    value : xr.DataArray
        Value of the power plant for each (day, simulation path, operational state).
        Shape: (n_steps, n_sims, 4).
    optimal_control_decision : xr.DataArray
        Optimal control decision for each (day, simulation path, operational state).
        Shape: (n_steps - 1, n_sims, 4).
    operational_states : list[str]
        List of operational states as lowercase strings, aligned with `OperationalState`.
    """

    def __init__(
        self,
        n_sims: int,
        asset_days: pd.DatetimeIndex,
        initial_state: OperationalState,
        spots_power: pd.DataFrame,
        spots_coal: pd.DataFrame,
        fwds_power: xr.DataArray,
        fwds_coal: xr.DataArray,
        fwd_0_power: ForwardCurve,
        fwd_0_coal: ForwardCurve,
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
        self.asset_days: pd.DatetimeIndex = asset_days
        self.n_steps: int = len(asset_days)

        self._initial_state = initial_state

        # Market data
        self.spots_power: pd.DataFrame = spots_power
        self.spots_coal: pd.DataFrame = spots_coal

        self.fwds_power: xr.DataArray = fwds_power
        self.fwds_coal: xr.DataArray = fwds_coal

        self.fwd_0_power = fwd_0_power
        self.fwd_0_coal = fwd_0_coal

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
        self.operational_states: list[OperationalState] = list(OperationalState)

        # -----------------------
        # Cashflows
        # -----------------------
        self._cashflows: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.operational_states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.asset_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.operational_states,
            },
            name=CASHFLOWS,
        )

        # -----------------------
        # Values
        # -----------------------
        self.values: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.operational_states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.asset_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.operational_states,
            },
            name=VALUE,
        )

        # -----------------------
        # Optimal control decisions
        # -----------------------
        self._optimal_control: xr.DataArray = xr.DataArray(
            np.full(
                (self.n_steps - 1, self.n_sims, len(self.operational_states)),
                OptimalControl.DO_NOTHING,
            ),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.asset_days[
                    :-1
                ],  # no decision on last day
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.operational_states,
            },
            name=OPTIMAL_CONTROL,
        )

        self._regressed_values: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, self.n_sims, len(self.operational_states))),
            dims=[DIM_SIMULATION_DAY, DIM_SIMULATION_PATH, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.asset_days,
                DIM_SIMULATION_PATH: range(self.n_sims),
                DIM_OPERATIONAL_STATE: self.operational_states,
            },
            name=REGRESSED_VALUE,
        )

        self._r2_scores: xr.DataArray = xr.DataArray(
            np.zeros((self.n_steps, len(self.operational_states))),
            dims=[DIM_SIMULATION_DAY, DIM_OPERATIONAL_STATE],
            coords={
                DIM_SIMULATION_DAY: self.asset_days,
                DIM_OPERATIONAL_STATE: self.operational_states,
            },
            name=REGRESSION_RESULTS,
        )

        # Compute spread: power - efficiency * coal
        self._spread: pd.DataFrame = spots_power.sub(self.efficiency * spots_coal)

        self._optimal_value = pd.DataFrame(np.zeros((self.n_steps, self.n_sims)),
                                           index=self.asset_days, columns=range(self.n_sims))
        self._optimal_state = pd.DataFrame(np.full((self.n_steps, self.n_sims),
                                                   OperationalState.IDLE, dtype=object), index=self.asset_days,
                                           columns=range(self.n_sims))
        self._optimal_cashflow = pd.DataFrame(np.zeros((self.n_steps, self.n_sims)), index=self.asset_days,
                                              columns=range(self.n_sims))

    def _initialize_terminal_state(self) -> None:
        """
        Initialize terminal state (end of simulation).
        Sets cashflows and values for the last day across all operational states.
        """
        last_day = self.asset_days[-1]
        spread_last = self._spread.loc[last_day]

        self._cashflows.loc[
            {DIM_SIMULATION_DAY: last_day, DIM_OPERATIONAL_STATE: OperationalState.IDLE}
        ] = -self.idle_costs
        self._cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RAMPING_UP,
            }
        ] = -self.ramping_up_costs
        self._cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RAMPING_DOWN,
            }
        ] = -self.ramping_down_costs
        self._cashflows.loc[
            {
                DIM_SIMULATION_DAY: last_day,
                DIM_OPERATIONAL_STATE: OperationalState.RUNNING,
            }
        ] = (
            -self.operation_costs + self.efficiency * spread_last.values
        )

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
        first_day = self.asset_days[0]

        if simulation_day == first_day:
            # Special case: first day, no regression
            next_day = self.asset_days[1]
            for state in OperationalState:
                self._value_0 = (
                    self.values.sel(simulation_day=next_day, operational_state=state)
                    .mean(dim="simulation_path")
                    .values
                )
                # Broadcast the mean to all simulation paths
                self._regressed_values.loc[
                    dict(simulation_day=simulation_day, operational_state=state)
                ] = np.full(self.n_sims, self._value_0)
        else:
            # Normal case: regress for all states
            for state in OperationalState:
                self._regressed_values.loc[
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

        cont_val = self._regressed_values.loc[
            dict(simulation_day=simulation_day, operational_state=idle)
        ]
        exer_val = self._regressed_values.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ]

        self._cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=idle)
        ] = -self.idle_costs
        self.values.loc[dict(simulation_day=simulation_day, operational_state=idle)] = (
            -self.idle_costs + np.maximum(cont_val, exer_val)
        )
        self._optimal_control.loc[
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

        self._cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ] = -self.ramping_up_costs
        self.values.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_up)
        ] = (
            -self.ramping_up_costs
            + self._regressed_values.loc[
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

        self._cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ] = -self.ramping_down_costs
        self.values.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ] = (
            -self.ramping_down_costs
            + self._regressed_values.loc[
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

        cont_val = self._regressed_values.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ]
        exer_val = self._regressed_values.loc[
            dict(simulation_day=simulation_day, operational_state=ramp_down)
        ]

        self._cashflows.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = (-self.operation_costs + self.efficiency * self._spread.loc[simulation_day])
        self.values.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = (
            -self.operation_costs
            + self.efficiency * self._spread.loc[simulation_day]
            + np.maximum(cont_val, exer_val)
        )
        self._optimal_control.loc[
            dict(simulation_day=simulation_day, operational_state=running)
        ] = np.where(
            cont_val > exer_val, OptimalControl.DO_NOTHING, OptimalControl.RAMPING_DOWN
        )

    def _extract_front_months(self, fwds: xr.DataArray, simulation_day: pd.Timestamp) -> xr.DataArray:
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
        start = pd.Timestamp(simulation_day)
        end = self.asset_days[-1]  # last asset day

        return fwds.sel({DELIVERY_START: slice(start, end), SIMULATION_DAY: simulation_day})

    def _regress(
        self,
        simulation_day: pd.Timestamp,
        state: OperationalState,
        r2_threshold: float = 0.0,
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
        spots_power = self.spots_power.loc[simulation_day].values
        spots_coal = self.spots_coal.loc[simulation_day].values

        fwds_power_front_months = self._extract_front_months(self.fwds_power, simulation_day).values
        fwds_coal_front_months = self._extract_front_months(self.fwds_coal, simulation_day).values

        # Stack features
        x = np.hstack([spots_power.reshape(-1, 1), spots_coal.reshape(-1, 1),
                       fwds_power_front_months, fwds_coal_front_months])

        # Extract target values: optimal value for the next day
        y = (
            self.values.sel(simulation_day=simulation_day + pd.Timedelta(days=1))
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

        # Check R² threshold
        if r2_score < r2_threshold:
            raise ValueError(
                f"Regression R²={r2_score:.4f} below threshold {r2_threshold} "
                f"for state {state.name} on {simulation_day.date()}"
            )

        # Store regression results in the class xarray field
        self._regressed_values.loc[
            dict(simulation_day=simulation_day, operational_state=state)
        ] = regressed_value

        # Store regression results in the class xarray field
        self._r2_scores.loc[
            dict(simulation_day=simulation_day, operational_state=state)
        ] = r2_score

        return regressed_value

    def determine_optimal_dispatch(self) -> None:
        """
        Determine optimal dispatch along all simulation paths using date-based indexing.

        Updates the pre-allocated DataFrames directly.
        Uses vectorized operations to speed up next-state calculations.
        Logs elapsed time per simulation day and total time.
        """
        start_time = time.time()
        logging.info("Starting optimal dispatch calculation...")

        # First day
        first_day = self.asset_days[0]
        self._optimal_state.loc[first_day, :] = self._initial_state
        self._optimal_value.loc[first_day, :] = self.values.sel(
            simulation_day=first_day,
            operational_state=self._initial_state
        ).values
        self._optimal_cashflow.loc[first_day, :] = self._cashflows.sel(
            simulation_day=first_day,
            operational_state=self._initial_state
        ).values

        # Remaining days
        for idx, (current_day, prev_day) in enumerate(zip(self.asset_days[1:], self.asset_days[:-1])):
            day_start = time.time()

            # Previous states as Series of enums
            prev_optimal_states = self._optimal_state.loc[prev_day, :]

            # Previous optimal control as DataArray
            prev_optimal_control = self._optimal_control.sel(
                simulation_day=prev_day,
                operational_state=xr.DataArray(prev_optimal_states, dims=DIM_SIMULATION_PATH)
            )

            # Vectorized next state computation
            current_optimal_state_values = get_next_state(prev_optimal_states, prev_optimal_control)

            # Store next states
            self._optimal_state.loc[current_day, :] = current_optimal_state_values

            # Wrap into DataArray for selection
            current_optimal_states = xr.DataArray(current_optimal_state_values, dims=DIM_SIMULATION_PATH)

            # Fill value and cashflow
            self._optimal_value.loc[current_day, :] = self.values.sel(
                simulation_day=current_day,
                operational_state=current_optimal_states
            ).values
            self._optimal_cashflow.loc[current_day, :] = self._cashflows.sel(
                simulation_day=current_day,
                operational_state=current_optimal_states
            ).values

        total_elapsed = time.time() - start_time
        logging.info(f"Optimal dispatch calculation completed in {total_elapsed:.3f}s")

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
        self.values : xr.DataArray
            Updated values for each day, simulation path, and operational state.
        self.optimal_control : xr.DataArray
            Updated optimal control decisions for each day, path, and state.
        """
        logging.info("Starting power plant optimization...")
        start_time = time.time()

        self._initialize_terminal_state()

        for simulation_day in tqdm(
            reversed(self.asset_days[:-1]),
            total=len(self.asset_days) - 1,
            desc="Optimizing simulation days",
            unit="day",
        ):
            self._optimize(simulation_day)

        self.determine_optimal_dispatch()

        elapsed_time = time.time() - start_time
        logging.info(
            "Power plant optimization successfully finished. "
            f"Elapsed time: {elapsed_time:.2f} seconds."
        )
