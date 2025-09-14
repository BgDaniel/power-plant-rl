import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from market_simulation.two_factor_model import TwoFactorForwardModel
from market_simulation.market_helpers import DELIVERY_START
from market_simulation.constants import DT
from math_helpers import (
    instantaneous_forward_volatility,
    log_variance,
    variance,
    mean,
)
from test_helpers import assert_relative_difference


# -------------------------------
# Pytest fixture
# -------------------------------
@pytest.fixture
def example_model_and_data() -> tuple:
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
    two_factor_model = TwoFactorForwardModel(
        config_path="power_2_factor_model_config.json"
    )
    start_date = pd.Timestamp("2025-09-13")
    n_rel_fwds = 6
    fwd_dates = pd.date_range(start=start_date, periods=n_rel_fwds, freq="MS")
    fwd_values = 50 + np.arange(n_rel_fwds)
    fwd_0 = pd.Series(data=fwd_values, index=fwd_dates)

    n_days = 365
    simulation_days = pd.date_range(start=start_date, periods=n_days, freq="D")

    n_sims = 5000
    fwds, _ = two_factor_model.simulate(
        fwd_0=fwd_0, n_sims=n_sims, simulation_days=simulation_days
    )

    return two_factor_model, fwd_0, simulation_days, fwds


class TestTwoFactorForwardModel:
    """
    Test suite for the TwoFactorForwardModel using a generic test helper.
    """

    @staticmethod
    def _bootstrap_ci(
        values: pd.DataFrame,
        func: callable,
        n_resample: int = 2500,
        percentiles: tuple[float, float] = (1, 5),
    ) -> dict[int, dict[str, pd.Series]]:
        """
        Compute bootstrap confidence intervals from a DataFrame of simulation paths.
        """
        n_days, n_sims = values.shape
        resampled_results = np.zeros((n_days, n_resample))

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
        conf: dict[int, dict[str, pd.Series]],
        title: str,
    ) -> None:
        """
        Plot simulation results against analytical benchmarks with confidence intervals.
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

        plt.xlabel("Date")
        plt.ylabel(title)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _generic_test(
        self,
        example_model_and_data: tuple,
        delivery_start: pd.Timestamp,
        empirical_func: callable,
        analytical_func: callable,
        title: str,
        name: str,
        max_diff: float = 0.05,
        n_resample: int = 1000,
    ) -> None:
        """
        Generic test helper for empirical vs analytical statistics.
        """
        model, fwd_0, simulation_days, fwds = example_model_and_data
        fwd_values = fwds.sel({DELIVERY_START: delivery_start}).values
        sim_df = pd.DataFrame(
            fwd_values.T, index=simulation_days
        )  # rows=time, cols=sims

        # Compute empirical and analytical series
        empirical_series: pd.Series = empirical_func(sim_df)
        analytical_series: pd.Series = analytical_func(
            model, fwd_0, simulation_days, delivery_start
        )

        # Compute bootstrap confidence intervals
        conf = self._bootstrap_ci(sim_df, empirical_func, n_resample=n_resample)

        # Plot results
        self._plot_results(empirical_series, analytical_series, conf, title=title)

        # Relative difference assertion
        assert_relative_difference(
            empirical_series, analytical_series, max_diff=max_diff, name=name
        )

    # -------------------------------
    # Individual test cases
    # -------------------------------
    def test_instantaneous_forward_volatility(
        self, example_model_and_data: tuple
    ) -> None:
        self._generic_test(
            example_model_and_data,
            delivery_start=pd.Timestamp("2025-12-01"),
            empirical_func=instantaneous_forward_volatility,
            analytical_func=lambda model, fwd_0, sim_days, delivery_start: model.instantaneous_forward_vol(
                sim_days, delivery_start
            ),
            title="Inst. Fwd Vol",
            name="instantaneous_forward_volatility",
        )

    def test_log_variance(self, example_model_and_data: tuple) -> None:
        self._generic_test(
            example_model_and_data,
            delivery_start=pd.Timestamp("2025-12-01"),
            empirical_func=log_variance,
            analytical_func=lambda model, fwd_0, sim_days, delivery_start: model.log_var(
                sim_days, delivery_start
            ),
            title="Log Fwd Var",
            name="log_variance",
        )

    def test_variance(self, example_model_and_data: tuple) -> None:
        self._generic_test(
            example_model_and_data,
            delivery_start=pd.Timestamp("2025-12-01"),
            empirical_func=variance,
            analytical_func=lambda model, fwd_0, sim_days, delivery_start: model.var(
                fwd_0, sim_days, delivery_start
            ),
            title="Variance",
            name="variance",
        )

    def test_mean(self, example_model_and_data: tuple) -> None:
        self._generic_test(
            example_model_and_data,
            delivery_start=pd.Timestamp("2025-12-01"),
            empirical_func=mean,
            analytical_func=lambda model, fwd_0, sim_days, delivery_start: pd.Series(
                fwd_0.loc[delivery_start], index=sim_days
            ),
            title="Mean Fwd",
            name="mean",
        )
