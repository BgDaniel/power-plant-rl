from valuation.power_plant.power_plant import PowerPlant


import matplotlib.pyplot as plt
import numpy as np


from valuation.operations_states import OperationalState
from valuation.operational_control import OptimalControl


class OpmizationVisualization(PowerPlant):
    def __init__(self):
        pass

    def plot_optimization_results(
        self, initial_state: OperationalState, simulation_path: int
    ) -> None:
        """Plots the spot prices, optimal values, control decisions, and states along a given simulation path.

        Args:
            initial_state (OperationalState): The initial state to start from in the simulation.
            simulation_path (int): The index of the simulation path to plot.
        """
        # Retrieve the spot prices (power and coal) along the simulation path
        power_prices = self.day_ahead_power.iloc[:, simulation_path]
        coal_prices = self.day_ahead_coal.iloc[:, simulation_path]

        # Get the optimal value and control decisions along the path
        (
            value_along_path,
            optimal_states_along_path,
            optimal_control_along_path,
            cashflow_along_path,
        ) = self.value_along_path(initial_state, simulation_path)

        # Create the plot figure with four subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 20))

        # Plot the spot prices (Power and Coal)
        axes[0].plot(
            power_prices.index, power_prices.values, label="Power Price", color="blue"
        )
        axes[0].plot(
            coal_prices.index, coal_prices.values, label="Coal Price", color="orange"
        )
        axes[0].set_title("Spot Prices along Simulation Path")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Price")
        axes[0].legend()

        # Plot the optimal value along the simulation path
        axes[1].plot(
            power_prices.index, value_along_path, label="Optimal Value", color="green"
        )
        axes[1].set_title("Optimal Value along Simulation Path")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Optimal Value")

        # Plot the optimal value along the simulation path
        axes[2].plot(
            power_prices.index,
            cashflow_along_path,
            label="Cashflow along path",
            color="red",
        )

        # Plot the control decisions along the simulation path as markers
        has_ramping_up_label_written = False
        has_ramping_down_label_written = False
        has_idle_label_written = False
        has_do_nothing_label_written = False

        for i, control in enumerate(optimal_control_along_path):
            if control == OptimalControl.RAMPING_UP:

                if not has_ramping_up_label_written:
                    has_ramping_up_label_written = True
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="green",
                        marker="x",
                        label="Ramping Up",
                        s=30.0,
                    )
                else:
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="green",
                        marker="x",
                        s=15.0,
                    )

            elif control == OptimalControl.RAMPING_DOWN:
                if not has_ramping_down_label_written:
                    has_ramping_down_label_written = True
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="red",
                        marker="o",
                        label="Ramping Down",
                        s=15.0,
                    )
                else:
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="red",
                        marker="o",
                        s=15.0,
                    )
            else:  # DO_NOTHING
                if not has_do_nothing_label_written:
                    has_do_nothing_label_written = True
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="yellow",
                        marker="o",
                        label="Ramping Down",
                        s=5.0,
                    )
                else:
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color="yellow",
                        marker="o",
                        s=5.0,
                    )

        axes[3].set_title("Control Decisions along Simulation Path")
        axes[3].set_xlabel("Date")
        axes[3].set_ylabel("Control Decision")
        axes[3].legend()

        # Plot the states along the simulation path as markers
        for i, state in enumerate(optimal_states_along_path):
            if state == OperationalState.IDLE:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color="blue",
                    marker="o",
                    label="Idle" if i == 0 else "",
                )
            elif state == OperationalState.RAMPING_UP:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color="green",
                    marker="x",
                    label="Ramping Up" if i == 0 else "",
                )
            elif state == OperationalState.RAMPING_DOWN:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color="red",
                    marker="s",
                    label="Ramping Down" if i == 0 else "",
                )
            elif state == OperationalState.RUNNING:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color="yellow",
                    marker="o",
                    label="Running" if i == 0 else "",
                )

        axes[3].set_title("States along Simulation Path")
        axes[3].set_xlabel("Date")
        axes[3].set_ylabel("State")
        axes[3].legend()

        plt.tight_layout()
        plt.show()

    def _plot_3d_regression(self, X, y, model, X_poly, poly, i_day, j_state):
        """Plots the actual vs predicted values and the regression surface in 3D."""
        # Predict the optimal values using the model
        y_pred = model.predict(X_poly)

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot of actual values
        ax.scatter(X[:, 0], X[:, 1], y, label="Actual Values", color="blue", alpha=0.6)

        # Scatter plot of predicted values
        ax.scatter(
            X[:, 0], X[:, 1], y_pred, label="Predicted Values", color="red", alpha=0.6
        )

        # Create a grid for plotting the regression surface
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = model.predict(
            poly.transform(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T)
        ).reshape(X_grid.shape)

        # Plot the regression surface
        ax.plot_surface(X_grid, Y_grid, Z_grid, color="gray", alpha=0.3)

        # Add labels and title
        ax.set_xlabel("Day Ahead Power")
        ax.set_ylabel("Day Ahead Coal")
        ax.set_zlabel("Optimal Value")
        ax.set_title(f"3D Regression: Day {i_day}, State {j_state}")

        # Show the legend
        ax.legend()

        # Show the plot
        plt.show()
