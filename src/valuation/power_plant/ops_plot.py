import matplotlib

from valuation.operations_states import OperationalState

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from valuation.power_plant.power_plant import PowerPlant
from valuation.regression.polynomial_regressor import KEY_R2


class OpsPlot:
    """
    Class for plotting results from a PowerPlant simulation.

    Provides visualization utilities for regression results, cashflows,
    values, and optimal control decisions.

    Attributes
    ----------
    power_plant : PowerPlant
        The PowerPlant instance containing the simulation results.
    """

    def __init__(self, power_plant: PowerPlant) -> None:
        """
        Initialize the OpsPlot instance.

        Parameters
        ----------
        power_plant : PowerPlant
            The PowerPlant instance with completed simulation and regression results.
        """
        self.power_plant: PowerPlant = power_plant

    def plot_r2(self) -> None:
        """
        Plot R² values over simulation days in a single plot with one curve per operational state.

        Colors:
        - IDLE: red
        - RUNNING: green
        - RAMPING_DOWN: orange
        - RAMPING_UP: yellow

        Line styles:
        - IDLE: solid
        - RUNNING: dashed
        - RAMPING_DOWN: dash-dot
        - RAMPING_UP: dotted

        UNDEFINED states are ignored.
        """
        asset_days = list(self.power_plant.asset_days[1:])  # Skip first day if needed
        states = [s for s in self.power_plant.operational_states if s != OperationalState.UNDEFINED]

        color_map = {
            OperationalState.IDLE: "red",
            OperationalState.RUNNING: "green",
            OperationalState.RAMPING_DOWN: "orange",
            OperationalState.RAMPING_UP: "yellow"
        }
        linestyle_map = {
            OperationalState.IDLE: "-",
            OperationalState.RUNNING: "--",
            OperationalState.RAMPING_DOWN: "-.",
            OperationalState.RAMPING_UP: ":"
        }

        plt.figure(figsize=(12, 5))

        for state in states:
            r2_values = self.power_plant._r2_scores.sel(
                simulation_day=asset_days,
                operational_state=state
            ).values
            plt.plot(
                asset_days,
                r2_values,
                label=state.name,  # Use enum name for legend
                color=color_map.get(state, "blue"),
                linestyle=linestyle_map.get(state, "-")
            )

        plt.xlabel("Simulation Day")
        plt.ylabel("R²")
        plt.title("R² over Time for Different Operational States")
        plt.grid(True)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(states))
        plt.tight_layout()
        plt.show(block=True)

    def plot_operation_along_path(self, path_index: int) -> None:
        """
        Detailed three-row plot for a single simulation path.

        1. Optimal value + cashflows
        2. Spread with ramp-up/down markers
        3. Operational state timeline

        Parameters
        ----------
        path_index : int
            Index of the simulation path to visualize.
        """
        asset_days = self.power_plant.asset_days

        # Row 1: Optimal value and cashflows
        optimal_values = self.power_plant._optimal_value.iloc[:,path_index]
        cashflows = self.power_plant._optimal_cashflow.iloc[:,path_index]

        # Row 2: Spread and ramp markers
        spreads = self.power_plant._spread.iloc[:,path_index]
        states = self.power_plant._optimal_state.iloc[:,path_index]
        ramp_up_idx = [i for i, s in enumerate(states) if s == OperationalState.RAMPING_UP]
        ramp_down_idx = [i for i, s in enumerate(states) if s == OperationalState.RAMPING_DOWN]

        # Row 3: Operational state timeline
        color_map = {'RUNNING': 'green', 'IDLE': 'red', 'RAMPING_UP': 'yellow', 'RAMPING_DOWN': 'yellow'}
        state_colors = [color_map.get(s, 'gray') for s in states]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 2, 2, 0.5]})

        # --- Row 1 ---
        ax1.plot(asset_days, optimal_values, color='blue', label='Optimal Value')
        ax1.grid(True)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(asset_days, cashflows, color='orange', alpha=0.5, label='Cashflows')
        ax1.set_title(f"Asset Value and Cashflows along path {path_index}")
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # --- Row 2 ---
        ax2.plot(asset_days, spreads.loc[asset_days], color='purple', label='Spread')
        ax2.scatter(asset_days[ramp_up_idx], spreads[ramp_up_idx], color='green', marker='^', label='ramping up', zorder=5)
        ax2.scatter(asset_days[ramp_down_idx], spreads[ramp_down_idx], color='red', marker='v', label='ramping down', zorder=5)
        ax2.set_title("Clean Dark Spread")
        ax2.grid(True)
        ax2.legend()

        # Third row: difference between power and coal forward curves
        ax3.plot(asset_days, self.fwd_0_power.loc[asset_days], label="Power", color="blue", linestyle="-")
        ax3.plot(asset_days, self.fwd_0_coal.loc[asset_days], label="Coal", color="brown", linestyle="--")
        ax3.set_xlabel("Simulation Day")
        ax3.set_title("Initial Forward Curves")
        ax3.legend(loc="upper right")
        ax3.grid(True)

        # Fourth row: operational state as numeric series with color segments
        state_numeric = []
        state_colors = []
        for s in self._optimal_state.loc[asset_days, path_index]:  # replace path_index with the desired path
            if s == OperationalState.IDLE:
                state_numeric.append(0)
                state_colors.append("red")
            elif s == OperationalState.RUNNING:
                state_numeric.append(1)
                state_colors.append("green")
            elif s in (OperationalState.RAMPING_UP, OperationalState.RAMPING_DOWN):
                state_numeric.append(0.5)
                state_colors.append("yellow")
            else:
                state_numeric.append(np.nan)
                state_colors.append("gray")

        # Plot colored line segments
        for i in range(len(asset_days)-1):
            ax4.plot(
                asset_days[i:i+2],
                state_numeric[i:i+2],
                color=state_colors[i],
                linewidth=2
            )

        ax4.set_ylim(-0.1, 1.1)
        ax4.set_yticks([0, 0.5, 1])
        ax4.set_yticklabels(["IDLE", "RAMPING", "RUNNING"])
        ax4.set_xlabel("Simulation Day")
        ax4.set_title("Operational State Timeline (Colored)")
        ax4.grid(True)

        plt.tight_layout()
        plt.show(block=True)

def plot_simulation_statistics_with_state_heatmap(
    self,
    confidence_levels: tuple[float, float] = (0.01, 0.05)
) -> None:
    """
    Plot simulation statistics aggregated over all simulation paths with a state heatmap.

    Row 1: Power plant value mean ± 1%/5% percentiles, cashflows mean ± percentiles.
    Row 2: Spread price mean ± 1%/5% percentiles.
    Row 3: Heatmap of operational states (red=idle, yellow=ramping, green=running)
           showing fraction of simulation paths in each state per day.

    Parameters
    ----------
    confidence_levels : tuple[float, float], default=(0.01, 0.05)
        Lower percentile levels for confidence shading.
    """
    import matplotlib.dates as mdates

    lower1, lower2 = confidence_levels
    upper1, upper2 = 1 - lower1, 1 - lower2

    sim_days = self.power_plant.asset_days[1:]

    # -----------------------------
    # Row 1: Power plant value + cashflows
    # -----------------------------
    values = self.power_plant.asset_values  # shape: (n_paths, n_days)
    cashflows = self.power_plant._cashflows  # shape: (n_paths, n_days)

    value_mean = values.mean(axis=0)
    value_lower1 = np.percentile(values, lower1 * 100, axis=0)
    value_upper1 = np.percentile(values, upper1 * 100, axis=0)
    value_lower2 = np.percentile(values, lower2 * 100, axis=0)
    value_upper2 = np.percentile(values, upper2 * 100, axis=0)

    cash_mean = cashflows.mean(axis=0)
    cash_lower1 = np.percentile(cashflows, lower1 * 100, axis=0)
    cash_upper1 = np.percentile(cashflows, upper1 * 100, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(sim_days, value_mean, color="blue", label="Mean Value")
    ax1.fill_between(sim_days, value_lower2, value_upper2, color="blue", alpha=0.2, label="5% CI")
    ax1.fill_between(sim_days, value_lower1, value_upper1, color="blue", alpha=0.4, label="1% CI")
    ax1.set_ylabel("Power Plant Value")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.bar(sim_days, cash_mean, width=0.8, alpha=0.5, color="gray", label="Mean Cashflow")
    ax2.set_ylabel("Cashflows")
    ax2.grid(False)

    # -----------------------------
    # Row 2: Spread price
    # -----------------------------
    spread = self.power_plant.spread_prices  # shape: (n_paths, n_days)
    spread_mean = spread.mean(axis=0)
    spread_lower1 = np.percentile(spread, lower1 * 100, axis=0)
    spread_upper1 = np.percentile(spread, upper1 * 100, axis=0)
    spread_lower2 = np.percentile(spread, lower2 * 100, axis=0)
    spread_upper2 = np.percentile(spread, upper2 * 100, axis=0)

    ax3 = axes[1]
    ax3.plot(sim_days, spread_mean, color="purple", label="Mean Spread")
    ax3.fill_between(sim_days, spread_lower2, spread_upper2, color="purple", alpha=0.2, label="5% CI")
    ax3.fill_between(sim_days, spread_lower1, spread_upper1, color="purple", alpha=0.4, label="1% CI")
    ax3.set_ylabel("Spread Price")
    ax3.grid(True)

    # -----------------------------
    # Row 3: State heatmap
    # -----------------------------
    # Map operational states to numbers: 0=idle, 1=ramping (up/down), 2=running
    state_map = {"IDLE": 0, "RAMPING_UP": 1, "RAMPING_DOWN": 1, "RUNNING": 2}
    states_array = np.vectorize(state_map.get)(self.power_plant.operational_state_matrix)  # shape: (n_paths, n_days)

    # Compute fractions
    frac_idle = (states_array == 0).mean(axis=0)
    frac_ramping = (states_array == 1).mean(axis=0)
    frac_running = (states_array == 2).mean(axis=0)

    # Combine fractions to RGB colors: red=idle, yellow=ramping, green=running
    colors = np.stack([frac_idle, frac_ramping, frac_running], axis=1)

    ax4 = axes[2]
    for i, day in enumerate(sim_days):
        ax4.axvspan(day, day + pd.Timedelta(days=1), color=colors[i], linewidth=0)

    ax4.set_ylabel("State Fraction")
    ax4.set_yticks([])
    ax4.grid(False)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    ax1.legend(loc="upper left")
    ax3.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()