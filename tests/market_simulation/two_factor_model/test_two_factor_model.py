import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from forward_curve.forward_curve import ForwardCurve
from market_simulation.two_factor_model.two_factor_model import TwoFactorForwardModel
from market_simulation.market_helpers import DELIVERY_START
from math_helpers import instant_fwd_vol, log_var
from tests.test_helpers import assert_relative_difference


# -------------------------------
# Pytest fixture
# -------------------------------
@pytest.fixture
def example_model_and_data() -> (
    tuple[TwoFactorForwardModel, pd.Series, pd.DatetimeIndex, "xr.DataArray"]
):
    """
    Build a TwoFactorForwardModel, generate initial forward curve,
    and simulate multiple paths.

    Returns
    -------
    tuple
        - TwoFactorForwardModel : calibrated model instance
        - pd.Series : initial forward curve
        - pd.DatetimeIndex : simulation days
        - xarray.DataArray : simulated forward curves (n_sims x n_days x delivery)
    """
    as_of_date = pd.Timestamp("2025-09-13")

    simulation_start = as_of_date
    simulation_end = simulation_start + pd.Timedelta(days=365)

    simulation_days = pd.date_range(
        start=simulation_start, end=simulation_end, freq="D"
    )

    # Model initialization
    two_factor_model = TwoFactorForwardModel(
        as_of_date=as_of_date,
        simulation_days=simulation_days,
        config_path="model_configs/power_2_factor_model_config.json",
    )

    # Forward curve setup
    fwd_0 = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=simulation_start,
        end_date=simulation_end,
        start_value=60.0,
        end_value=85.0,
        name="Coal Forward Curve",
    )

    # Simulation
    n_sims = 10000

    fwds, month_aheads, spots = two_factor_model.simulate(
        fwd_0=fwd_0, n_sims=n_sims
    )

    return two_factor_model, fwd_0, simulation_days, fwds, month_aheads, spots


class TestTwoFactorForwardModel:
    """
    Test suite for the TwoFactorForwardModel using a generic test helper.

    Provides tests for empirical vs analytical statistics including:
        - instantaneous forward volatility
        - log variance
        - variance
        - mean forward price
    """

    DELIVERY_STARTS = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-06-01"),
    ]

    @staticmethod
    def _bootstrap_ci(
        values: pd.DataFrame,
        func: callable,
        n_resample: int = 1000,
        percentiles: tuple[float, float] = (1, 5),
    ) -> dict[int, dict[str, pd.Series]]:
        """
        Compute bootstrap confidence intervals from a DataFrame of simulation paths.

        Parameters
        ----------
        values : pd.DataFrame
            DataFrame of simulated values (rows=time, columns=simulations).
        func : callable
            Function to apply to each resampled set (e.g., mean, variance).
        n_resample : int, optional
            Number of bootstrap resamples (default=1000).
        percentiles : tuple of float, optional
            Percentiles to compute lower and upper bounds (default=(1, 5)).

        Returns
        -------
        dict[int, dict[str, pd.Series]]
            Dictionary of confidence intervals for each percentile with keys 'lower' and 'upper'.
        """
        n_days, n_sims = values.shape
        resampled_results: np.ndarray = np.zeros((n_days, n_resample))

        # Resample columns (simulation paths)
        resample_idx = np.random.randint(0, n_sims, size=(n_resample, n_sims))
        for r in range(n_resample):
            resampled = values.iloc[:, resample_idx[r]]
            resampled_results[:, r] = func(resampled)

        conf: dict[int, dict[str, pd.Series]] = {}
        for p in percentiles:
            lower = pd.Series(
                np.percentile(resampled_results, p, axis=1), index=values.index
            )
            upper = pd.Series(
                np.percentile(resampled_results, 100 - p, axis=1), index=values.index
            )
            conf[p] = {"lower": lower, "upper": upper}

        return conf

    @staticmethod
    def _plot_results(
        empirical_data: pd.Series,
        analytical_data: pd.Series,
        conf: Optional[dict[int, dict[str, pd.Series]]],
        title: str,
    ) -> None:
        """
        Plot simulation results against analytical benchmarks with confidence intervals.

        Parameters
        ----------
        empirical_data : pd.Series
            Series of empirical statistics from simulations.
        analytical_data : pd.Series
            Series of analytical statistics for comparison.
        conf : dict[int, dict[str, pd.Series]]
            Bootstrap confidence intervals.
        title : str
            Plot title.
        """
        plt.plot(
            empirical_data.index,
            empirical_data,
            color="deepskyblue",
            lw=1,
            label="Empirical",
        )
        plt.plot(
            analytical_data.index,
            analytical_data,
            color="red",
            lw=1,
            linestyle="--",
            label="Analytical",
        )

        if conf:
            conf_1_lower, conf_1_upper = conf[1]["lower"], conf[1]["upper"]
            conf_5_lower, conf_5_upper = conf[5]["lower"], conf[5]["upper"]

            plt.fill_between(
                empirical_data.index,
                conf_5_lower,
                conf_5_upper,
                color="green",
                alpha=0.2,
                label="5% CI",
            )
            plt.fill_between(
                empirical_data.index,
                conf_1_lower,
                conf_1_upper,
                color="limegreen",
                alpha=0.4,
                label="1% CI",
            )

            min_ci_lower = min(
                np.nanmin(conf_1_lower),
                np.nanmin(conf_5_lower),
                np.nanmin(empirical_data),
                np.nanmin(analytical_data),
            )
            max_ci_upper = max(
                np.nanmax(conf_1_upper),
                np.nanmax(conf_5_upper),
                np.nanmax(empirical_data),
                np.nanmax(analytical_data),
            )
            plt.ylim(min_ci_lower * 0.9, max_ci_upper * 1.05)

        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    @pytest.mark.parametrize("delivery_start", DELIVERY_STARTS)
    def test_mean(
        self,
        delivery_start: pd.Timestamp,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.01,
        plot: bool = True,
    ) -> None:
        """
        Test empirical forward mean against analytical flat forward mean
        for multiple delivery start dates.

        Parameters
        ----------
        delivery_start : pd.Timestamp
            The start date of the delivery period being tested.
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        plot : bool, optional
            Whether to plot the results.
        """
        two_factor_model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Extract simulated forward values
        fwd_values: np.ndarray = fwds.sel({DELIVERY_START: delivery_start}).values
        fwd: pd.DataFrame = pd.DataFrame(fwd_values.T, index=simulation_days)

        # Keep only rows strictly before delivery start
        fwd = fwd.loc[fwd.index < delivery_start]

        # Empirical mean across simulations
        mean_emp: pd.Series = fwd.mean(axis=1)

        # Analytical mean: flat forward until delivery_start, NaN afterwards
        mean_exp: pd.Series = pd.Series(
            [fwd_0.loc[delivery_start]] * len(fwd),
            index=fwd.index,
        )

        # Bootstrap confidence intervals (optional)
        conf = None
        if bootstrap_ci:
            conf = self._bootstrap_ci(fwd, lambda sample: sample.mean(axis=1))

        # Plot results
        observable_name = "Mean Fwd"
        if plot:
            self._plot_results(
                mean_emp,
                mean_exp,
                conf,
                title=f"{observable_name} - Delivery {delivery_start.date()}",
            )

        # Assert similarity
        assert_relative_difference(
            mean_emp,
            mean_exp,
            max_diff=max_diff,
            name=f"{observable_name}_{delivery_start.date()}",
        )

    @pytest.mark.parametrize("delivery_start", DELIVERY_STARTS)
    def test_variance(
        self,
        delivery_start: pd.Timestamp,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.05,
        plot: bool = True,
    ) -> None:
        """
        Test empirical forward variance against analytical variance
        for multiple delivery start dates.

        Parameters
        ----------
        delivery_start : pd.Timestamp
            The start date of the delivery period being tested.
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        plot : bool, optional
            Whether to plot the results.
        """
        two_factor_model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Extract simulated forward values
        fwd_values: np.ndarray = fwds.sel({DELIVERY_START: delivery_start}).values
        fwd: pd.DataFrame = pd.DataFrame(fwd_values.T, index=simulation_days)

        # Keep only rows strictly before delivery start
        fwd = fwd.loc[fwd.index < delivery_start]

        # Empirical variance across simulations
        var_emp: pd.Series = fwd.var(axis=1, ddof=1)

        # Analytical variance from the model
        var_exp: pd.Series = two_factor_model.var(fwd_0, maturity_date=delivery_start)
        var_exp = var_exp.loc[fwd.index]  # align with empirical dates

        # Bootstrap confidence intervals (optional)
        conf = None
        if bootstrap_ci:
            conf = self._bootstrap_ci(fwd, lambda sample: sample.var(axis=1, ddof=1))

        # Plot results
        observable_name = "Variance"
        if plot:
            self._plot_results(
                var_emp,
                var_exp,
                conf,
                title=f"{observable_name} - Delivery {delivery_start.date()}",
            )

        # Assert similarity
        assert_relative_difference(
            var_emp,
            var_exp,
            max_diff=max_diff,
            name=f"{observable_name}_{delivery_start.date()}",
        )

    @pytest.mark.parametrize("delivery_start", DELIVERY_STARTS)
    def test_instant_fwd_vol(
        self,
        delivery_start: pd.Timestamp,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.05,
        plot: bool = True,
    ) -> None:
        """
        Test instantaneous forward volatility against its analytical formula
        for multiple delivery start dates.

        Parameters
        ----------
        delivery_start : pd.Timestamp
            The start date of the delivery period being tested.
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        """
        two_factor_model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Extract simulated forward values
        fwd_values: np.ndarray = fwds.sel({DELIVERY_START: delivery_start}).values
        fwd: pd.DataFrame = pd.DataFrame(fwd_values.T, index=simulation_days)

        # Empirical and analytical instantaneous forward vol
        instant_fwd_vol_emp: pd.Series = instant_fwd_vol(fwd)
        instant_fwd_vol_exp: pd.Series = two_factor_model.instant_fwd_vol(
            maturity_date=delivery_start
        )

        instant_fwd_vol_emp = instant_fwd_vol_emp.loc[fwd.index < delivery_start]
        instant_fwd_vol_exp = instant_fwd_vol_exp.loc[fwd.index < delivery_start]

        # Bootstrap confidence intervals (optional)
        conf = None
        if bootstrap_ci:
            conf = self._bootstrap_ci(fwd, instant_fwd_vol)

        # Plot results (with delivery_start in title)
        observable_name = "Instant Fwd Vol"

        if plot:
            self._plot_results(
                instant_fwd_vol_emp,
                instant_fwd_vol_exp,
                conf,
                title=f"{observable_name} - Delivery {delivery_start.date()}",
            )

        # Assert similarity
        assert_relative_difference(
            instant_fwd_vol_emp,
            instant_fwd_vol_exp,
            max_diff=max_diff,
            name=f"{observable_name}_{delivery_start.date()}",
        )

    def test_mean_month_ahead(
        self,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.05,
        plot: bool = True,
    ):
        """
        Test the mean of the exponential OU day-ahead process over multiple simulation dates.

        Parameters
        ----------
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards, month-ahead, spots).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        plot : bool, optional
            Whether to produce a plot comparing empirical vs analytical results.
        """
        model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Compute empirical mean from simulated day-ahead series
        day_ahead_values = (
            (month_aheads / spots).sel(SIMULATION_DAY=simulation_days).values
        )
        empirical_mean = pd.Series(day_ahead_values.mean(axis=0), index=simulation_days)

        # Compute analytical mean
        analytical_mean = model.mean_day_ahead(simulation_days)

        # Bootstrap CI (optional)
        conf = None
        if bootstrap_ci:
            # Use a helper similar to _bootstrap_ci if you have it
            conf = model._bootstrap_ci(
                pd.DataFrame(day_ahead_values.T, index=simulation_days), np.mean
            )

        observable_name = "Day-Ahead Mean"

        # Plot (optional)
        if plot:
            self._plot_results(
                empirical_mean, analytical_mean, conf, title=observable_name
            )

        # Relative difference assertion
        assert_relative_difference(
            empirical_mean, analytical_mean, max_diff=max_diff, name=observable_name
        )

    def test_mean_day_ahead(
        self,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.05,
        plot: bool = True,
    ):
        """
        Test the mean of the exponential OU day-ahead process over multiple simulation dates.

        Parameters
        ----------
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards, month-ahead, spots).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        plot : bool, optional
            Whether to produce a plot comparing empirical vs analytical results.
        """
        model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Compute empirical mean from simulated day-ahead series
        day_ahead_values = (
            (month_aheads / spots).sel(SIMULATION_DAY=simulation_days).values
        )
        empirical_mean = pd.Series(day_ahead_values.mean(axis=0), index=simulation_days)

        # Compute analytical mean
        analytical_mean = model.mean_day_ahead(simulation_days)

        # Bootstrap CI (optional)
        conf = None
        if bootstrap_ci:
            # Use a helper similar to _bootstrap_ci if you have it
            conf = model._bootstrap_ci(
                pd.DataFrame(day_ahead_values.T, index=simulation_days), np.mean
            )

        observable_name = "Day-Ahead Mean"

        # Plot (optional)
        if plot:
            self._plot_results(
                empirical_mean, analytical_mean, conf, title=observable_name
            )

        # Relative difference assertion
        assert_relative_difference(
            empirical_mean, analytical_mean, max_diff=max_diff, name=observable_name
        )

    def test_variance_day_ahead(
        self,
        example_model_and_data: tuple,
        bootstrap_ci: bool = False,
        max_diff: float = 0.05,
        plot: bool = True,
    ):
        """
        Test the variance of the exponential OU day-ahead process over multiple simulation dates.

        Parameters
        ----------
        example_model_and_data : tuple
            Tuple of (model, initial forward, simulation days, simulated forwards, month-ahead, spots).
        bootstrap_ci : bool, optional
            Whether to compute bootstrap confidence intervals for plotting.
        max_diff : float, optional
            Maximum allowed relative difference between empirical and analytical results.
        plot : bool, optional
            Whether to produce a plot comparing empirical vs analytical results.
        """
        model, fwd_0, simulation_days, fwds, month_aheads, spots = (
            example_model_and_data
        )

        # Compute empirical variance from simulated day-ahead series
        day_ahead_values = (
            (month_aheads / spots).sel(SIMULATION_DAY=simulation_days).values
        )
        empirical_var = pd.Series(
            day_ahead_values.var(axis=0, ddof=1), index=simulation_days
        )

        # Compute analytical variance
        month_ahead_mean = month_aheads.sel(SIMULATION_DAY=simulation_days).values.mean(
            axis=0
        )
        analytical_var = model.variance_day_ahead(month_ahead_mean, simulation_days)

        # Bootstrap CI (optional)
        conf = None
        if bootstrap_ci:
            conf = model._bootstrap_ci(
                pd.DataFrame(day_ahead_values.T, index=simulation_days), np.var
            )

        observable_name = "Day-Ahead Var"

        # Plot (optional)
        if plot:
            self._plot_results(
                empirical_var, analytical_var, conf, title=f"{observable_name}"
            )

        # Relative difference assertion
        assert_relative_difference(
            empirical_var,
            analytical_var,
            max_diff=max_diff,
            name=observable_name,
        )
