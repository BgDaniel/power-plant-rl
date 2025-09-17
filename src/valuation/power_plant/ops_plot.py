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
        Plot the R² values over simulation days for each operational state.

        This method creates a figure with one subplot per operational state
        (IDLE, RAMPING_UP, RAMPING_DOWN, RUNNING), showing the R² time series
        of the regression results over the simulation period.

        Notes
        -----
        - Assumes `self.power_plant.r2_values` is an xarray.DataArray with
          dimensions ('simulation_day', 'operational_state').
        - Handles the edge case of a single operational state.
        """
        simulation_days: List[np.datetime64] = list(self.power_plant.simulation_days)
        states: List[str] = self.power_plant.operational_states
        n_states: int = len(states)

        fig, axes = plt.subplots(n_states, 1, figsize=(12, 3 * n_states), sharex=True)

        # Ensure axes is iterable
        if n_states == 1:
            axes = [axes]

        for i, state in enumerate(states):
            r2_values = self.power_plant.r2_scores.sel(
                simulation_day=self.power_plant.simulation_days[1:],
                operational_state=state,
            ).values
            axes[i].plot(simulation_days, r2_values, linestyle="-", color="blue")
            axes[i].set_title(f"R² over time - {state}")
            axes[i].set_ylabel("R²")
            axes[i].grid(True)

        axes[-1].set_xlabel("Simulation Day")
        plt.tight_layout()
        plt.show()


# Example usage:
# ops_plot = OpsPlot(power_plant)
# ops_plot.plot_r2()
