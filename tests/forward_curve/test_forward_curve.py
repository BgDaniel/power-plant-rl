import pytest
import pandas as pd
import numpy as np
from forward_curve import ForwardCurve


class TestForwardCurve:
    """
    Unit tests for the ForwardCurve class.

    Tests:
    - Initialization of ForwardCurve.
    - Generation of curves with different shapes.
    - Plotting functionality runs without errors.
    """

    def test_initialization(self) -> None:
        """Test that ForwardCurve initializes correctly and supports indexing and length."""
        start_date = pd.Timestamp("2025-01-01")
        series = pd.Series(
            [10, 20, 30], index=pd.date_range(start_date, periods=3, freq="D")
        )
        fc = ForwardCurve(series, start_date, "TestCurve")

        assert isinstance(fc.series, pd.Series)
        assert fc.start_date == start_date
        assert fc.name == "TestCurve"
        assert len(fc) == 3
        assert fc[0] == 10

    @pytest.mark.parametrize("shape", ["linear", "flat", "contango", "backwardation"])
    def test_generate_curve_shapes(self, shape: str) -> None:
        """Test generating forward curves of various shapes."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        fc = ForwardCurve.generate_curve(
            start_date, end_date, 50, 100, n_points=12, shape=shape, name="TestCurve"
        )

        assert isinstance(fc, ForwardCurve)
        assert len(fc.series) == 12
        assert fc.series.index[0] == start_date
        assert fc.series.index[-1] == end_date
        assert fc.name == "TestCurve"
        # Values should start near start_value
        np.testing.assert_allclose(fc.series.iloc[0], 50, rtol=1e-12)

    def test_plot_runs_without_error(self) -> None:
        """Ensure that the plot method executes without throwing an exception."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        fc = ForwardCurve.generate_curve(
            start_date, end_date, 50, 100, n_points=12, shape="linear", name="TestCurve"
        )

        # Just ensure the plot function runs without raising an exception
        fc.plot(title="Test Forward Curve Plot")
