import matplotlib
import pandas as pd

from valuation.operations_states import OperationalState

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from valuation.power_plant.power_plant import PowerPlant


class OpsPlotPowerPlant:
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
        states = [
            s
            for s in self.power_plant.operational_states
            if s != OperationalState.UNDEFINED
        ]

        color_map = {
            OperationalState.IDLE: "red",
            OperationalState.RUNNING: "green",
            OperationalState.RAMPING_DOWN: "orange",
            OperationalState.RAMPING_UP: "yellow",
        }
        linestyle_map = {
            OperationalState.IDLE: "-",
            OperationalState.RUNNING: "--",
            OperationalState.RAMPING_DOWN: "-.",
            OperationalState.RAMPING_UP: ":",
        }

        plt.figure(figsize=(12, 5))

        for state in states:
            r2_values = self.power_plant._r2_scores.sel(
                operational_state=state
            )
            plt.plot(
                r2_values,
                label=state.name,  # Use enum name for legend
                color=color_map.get(state, "blue"),
                linestyle=linestyle_map.get(state, "-"),
            )

        plt.xlabel("Simulation Day")
        plt.ylabel("R²")
        plt.title("R² over Time for Different Operational States")
        plt.grid(True)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(states))
        plt.tight_layout()
        plt.show(block=True)

    def plot_simulation_summary(
        self,
        confidence_levels: tuple[float, float] = (0.01, 0.05),
        path_index: int | None = None,
    ) -> None:
        """
        Plot simulation statistics aggregated over all simulation paths with optional single-path overlay.

        Row 1: Cashflows mean ± 1%/5% percentiles (shaded bands).
               If path_index is provided: overlay the cashflow of that path.
        Row 2: Power plant values mean ± 1%/5% percentiles (bar plot style).
               If path_index is provided: overlay the path's values.
        Row 3: Spread mean ± 1%/5% percentiles.
               If path_index is provided: overlay the spread along this path + markers for ramping up/down.
        Row 4: Left blank (reserved for later).

        Parameters
        ----------
        confidence_levels : tuple[float, float], default=(0.01, 0.05)
            Percentile levels for confidence shading/bands. Example: (0.01, 0.05) → 1% and 5% bands.
        path_index : int or None, optional
            Index of the simulation path to overlay. If None, only aggregates are plotted.
        """

        lower1, lower2 = confidence_levels

        label_ci1 = f"{int(lower1 * 100)}% CI"
        label_ci2 = f"{int(lower2 * 100)}% CI"

        upper1, upper2 = 1 - lower1, 1 - lower2
        asset_days = self.power_plant.asset_days

        # Prepare figure with 4 rows
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(8, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1, 0.1]},  # fourth row is half height
        )

        # --- Row 1: Values ---
        values = self.power_plant.values

        value_mean = values.mean(axis=1)
        value_lower1 = np.percentile(values, lower1 * 100, axis=1)
        value_upper1 = np.percentile(values, upper1 * 100, axis=1)
        value_lower2 = np.percentile(values, lower2 * 100, axis=1)
        value_upper2 = np.percentile(values, upper2 * 100, axis=1)

        ax1 = axes[0]
        ax1.plot(
            asset_days, value_mean, lw=1.5, color="yellow", label="Mean", linestyle="--"
        )
        ax1.fill_between(
            asset_days,
            value_lower2,
            value_upper2,
            color="blue",
            alpha=0.05,
            label=label_ci2,
        )
        ax1.fill_between(
            asset_days,
            value_lower1,
            value_upper1,
            color="blue",
            alpha=0.1,
            label=label_ci1,
        )

        if path_index is not None:
            ax1.plot(
                asset_days,
                values.iloc[:, path_index],
                color="red",
                lw=1,
                label=f"Path {path_index}",
            )

        ax1.set_title("Asset Value")
        ax1.grid(True)
        ax1.legend(
            loc="upper center",  # Position relative to bbox_to_anchor
            bbox_to_anchor=(
                0.5,
                -0.15,
            ),  # x=0.5 centers horizontally, y=-0.15 moves below the axes
            fontsize="small",
            markerscale=0.4,
            ncol=4,  # Optional: spread legend items across multiple columns
        )

        # --- Row 2: Cashflows ---
        cashflows = (
            self.power_plant.cashflows
        )  # shape (n_days, n_paths) or (n_paths, n_days)

        cash_mean = cashflows.mean(axis=1)
        cash_lower1 = np.percentile(cashflows, lower1 * 100, axis=1)
        cash_upper1 = np.percentile(cashflows, upper1 * 100, axis=1)
        cash_lower2 = np.percentile(cashflows, lower2 * 100, axis=1)
        cash_upper2 = np.percentile(cashflows, upper2 * 100, axis=1)

        ax2 = axes[1]
        ax2.plot(
            asset_days, cash_mean, lw=1.5, color="yellow", label="Mean", linestyle="--"
        )
        ax2.fill_between(
            asset_days,
            cash_lower2,
            cash_upper2,
            color="blue",
            alpha=0.05,
            label=label_ci2,
        )
        ax2.fill_between(
            asset_days,
            cash_lower1,
            cash_upper1,
            color="blue",
            alpha=0.1,
            label=label_ci1,
        )

        if path_index is not None:
            ax2.plot(
                asset_days,
                cashflows.iloc[:, path_index],
                color="red",
                lw=1,
                label=f"Path {path_index}",
            )

        ax2.set_title("Cashlows")
        ax2.grid(True)
        ax2.legend(
            loc="upper center",  # Position relative to bbox_to_anchor
            bbox_to_anchor=(
                0.5,
                -0.15,
            ),  # x=0.5 centers horizontally, y=-0.15 moves below the axes
            fontsize="small",
            markerscale=0.4,
            ncol=4,  # Optional: spread legend items across multiple columns
        )

        # --- Row 3: Spread ---
        spread = self.power_plant.spreads.loc[asset_days]

        spread_mean = spread.mean(axis=1)
        spread_lower1 = np.percentile(spread, lower1 * 100, axis=1)
        spread_upper1 = np.percentile(spread, upper1 * 100, axis=1)
        spread_lower2 = np.percentile(spread, lower2 * 100, axis=1)
        spread_upper2 = np.percentile(spread, upper2 * 100, axis=1)

        ax3 = axes[2]
        ax3.plot(
            asset_days,
            spread_mean,
            lw=1.5,
            color="yellow",
            label="Mean",
            linestyle="--",
        )
        ax3.fill_between(
            asset_days,
            spread_lower2,
            spread_upper2,
            color="blue",
            alpha=0.05,
            label=label_ci2,
        )
        ax3.fill_between(
            asset_days,
            spread_lower1,
            spread_upper1,
            color="blue",
            alpha=0.1,
            label=label_ci1,
        )

        if path_index is not None:
            spreads_path = spread.iloc[:, path_index]
            ax3.plot(
                asset_days,
                spreads_path,
                color="red",
                lw=1.0,
                label=f"Path {path_index}",
            )

            # Add markers for ramping states
            states_path = self.power_plant._optimal_state.iloc[:, path_index]
            ramp_up_idx = [
                i for i, s in enumerate(states_path) if s == OperationalState.RAMPING_UP
            ]
            ramp_down_idx = [
                i
                for i, s in enumerate(states_path)
                if s == OperationalState.RAMPING_DOWN
            ]
            # Ramping Up: green circle, slightly larger
            ax3.scatter(
                asset_days[ramp_up_idx],
                spreads_path.iloc[ramp_up_idx],
                color="green",
                marker="o",
                s=40,  # marker size
                label="Ramping Up",
            )

            # Ramping Down: red cross, slightly larger
            ax3.scatter(
                asset_days[ramp_down_idx],
                spreads_path.iloc[ramp_down_idx],
                color="red",
                marker="X",
                s=40,  # marker size
                label="Ramping Down",
            )

        ax3.set_title("Dark Spread")
        ax3.grid(True)
        ax3.legend(
            loc="upper center",  # Position relative to bbox_to_anchor
            bbox_to_anchor=(
                0.5,
                -0.15,
            ),  # x=0.5 centers horizontally, y=-0.15 moves below the axes
            fontsize="small",
            markerscale=0.4,
            ncol=6,  # Optional: spread legend items across multiple columns
        )

        # --- Row 4: State fraction heatmap ---
        ax4 = axes[3]

        # Map OperationalState to numeric categories
        state_map = {
            OperationalState.IDLE: 0,
            OperationalState.RAMPING_UP: 1,
            OperationalState.RAMPING_DOWN: 1,
            OperationalState.RUNNING: 2,
        }

        # Convert states to numeric array: shape (n_days, n_paths)
        states_array = np.vectorize(state_map.get)(
            self.power_plant._optimal_state.values
        )  # shape: (n_days, n_paths)

        # Compute fractions for each simulation day
        frac_idle = (states_array == 0).mean(axis=1)
        frac_ramping = (states_array == 1).mean(axis=1)
        frac_running = (states_array == 2).mean(axis=1)

        # Stack fractions into RGB colors: ramping split between red and green for yellow
        colors = np.zeros((len(asset_days), 3))
        colors[:, 0] = (
            frac_idle + 0.5 * frac_ramping
        )  # red channel: idle + half ramping
        colors[:, 1] = (
            frac_running + 0.5 * frac_ramping
        )  # green channel: running + half ramping
        colors[:, 2] = 0  # blue channel remains zero

        # Plot as colored band
        for i, day in enumerate(asset_days):
            ax4.axvspan(day, day + pd.Timedelta(days=1), color=colors[i], linewidth=0)

        ax4.set_yticks([])

        ax4.set_title("Operational State")
        ax4.grid(False)

        plt.tight_layout()
        plt.show(block=True)
