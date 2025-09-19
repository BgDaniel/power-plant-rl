from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from typing import Any, Dict
import matplotlib.pyplot as plt


class ForwardCurve:
    """
    A simple wrapper around a pandas Series representing a forward curve.

    Attributes
    ----------
    series : pd.Series
        The forward curve values indexed by dates.
    start_date : pd.Timestamp
        The start date of the forward curve.
    name : str
        A name for the forward curve (e.g., 'Power', 'Coal').
    """

    # String constants for curve shapes
    LINEAR = "linear"
    FLAT = "flat"
    CONTANGO = "contango"
    BACKWARDATION = "backwardation"

    def __init__(
        self,
        series: pd.Series,
        as_of_date: pd.Timestamp,
        name: str,
        freq: Optional[str] = "MS",  # default monthly start
    ) -> None:
        """
        Initialize the ForwardCurve.

        Parameters
        ----------
        series : pd.Series
            Forward curve values indexed by dates.
        as_of_date : pd.Timestamp
            The reference date of the curve.
        name : str
            Name of the forward curve.
        freq : str, optional
            Frequency to resample the series. Defaults to 'MS' (month start).
        """
        # Validate freq
        self.freq = freq
        if freq is not None:
            try:
                # Try creating a dummy date range to validate frequency
                pd.date_range(start=series.index[0], periods=2, freq=freq)
            except ValueError:
                raise ValueError(f"Invalid freq '{freq}'. Must be a valid pandas offset alias.")

            # Resample series if freq is provided
            series = series.resample(freq).mean()

        self.series: pd.Series = series
        self.as_of_date: pd.Timestamp = as_of_date
        self.name: str = name

    def __getitem__(self, key) -> float:
        return self.series[key]

    def __len__(self) -> int:
        return len(self.series)

    def __repr__(self) -> str:
        return f"ForwardCurve(name={self.name}, as_of_date={self.as_of_date}, len={len(self.series)})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize key attributes of the object to a dictionary.

        Only includes:
        - `as_of_date` as string
        - `name` as string
        - `series` as a dict with 'index' and 'values'

        Returns
        -------
        Dict[str, Any]
            Dictionary containing a serializable representation of the object.
        """
        result: Dict[str, Any] = {
            "as_of_date": str(getattr(self, "as_of_date", None)),
            "name": getattr(self, "name", None),
        }

        series: pd.Series = getattr(self, "series", None)
        if series is not None:
            result["series"] = {
                "index": [str(d) for d in series.index],
                "values": series.values.tolist(),
            }

        return result

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        Get the index of the forward curve series.

        Returns
        -------
        pd.DatetimeIndex
            Dates corresponding to the forward curve values.
        """
        return self.series.index

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the forward curve series.

        Returns
        np.ndarray
            Forward curve values.
        """
        return self.series.values

    @property
    def loc(self):
        """Return the loc accessor for the underlying series."""
        return self.series.loc

    def plot(self, title: Optional[str] = None) -> None:
        """
        Plot the forward curve with y-axis starting at 0 and upper limit 1.1 times the max value.

        Parameters
        ----------
        title : str, optional
            Title of the plot. Defaults to the forward curve's name.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.series.index,
            self.series.values,
            color="blue",
            label=self.name,
        )
        plt.xlabel("Date")
        plt.title(
            title or f"Forward Curve: {self.name}, as of {self.as_of_date.date()}"
        )
        plt.ylim(0, 1.1 * self.series.max())  # Set y-axis from 0 to 1.1×max value
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_curve(
        as_of_date: pd.Timestamp,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        start_value: float,
        end_value: float,
        shape: str = LINEAR,  # default to linear
        name: str = "ForwardCurve",
    ) -> ForwardCurve:
        """
        Generate a forward curve with a given shape and daily resolution.

        Parameters
        ----------
        start_date : pd.Timestamp
            Start date of the forward curve.
        end_date : pd.Timestamp
            End date of the forward curve.
        start_value : float
            Value at the start date.
        end_value : float
            Value at the end date.
        shape : str
            Shape of the forward curve. One of 'linear', 'flat', 'contango', 'backwardation'.
            Defaults to 'linear'.
        name : str
            Name of the forward curve.

        Returns
        -------
        ForwardCurve
            A ForwardCurve instance with the generated series.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n_points = len(dates)

        if shape == ForwardCurve.LINEAR:
            values = np.linspace(start_value, end_value, n_points)
        elif shape == ForwardCurve.FLAT:
            values = np.full(n_points, start_value)
        elif shape == ForwardCurve.CONTANGO:
            values = start_value + (end_value - start_value) * np.sqrt(
                np.linspace(0, 1, n_points)
            )
        elif shape == ForwardCurve.BACKWARDATION:
            values = start_value + (end_value - start_value) * (
                1 - np.sqrt(np.linspace(0, 1, n_points))
            )
        else:
            raise ValueError(f"Unsupported shape '{shape}'")

        series = pd.Series(data=values, index=dates)
        return ForwardCurve(series=series, as_of_date=as_of_date, name=name)


def generate_yearly_seasonal_curve(
    as_of_date: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    winter_value: float,
    summer_value: float,
    name: Optional[str] = "SeasonalForwardCurve",
) -> ForwardCurve:
    """
    Generate a yearly seasonal ForwardCurve with peak in summer and trough in winter.

    Parameters
    ----------
    as_of_date : pd.Timestamp
        Reference date of the forward curve.
    start_date : pd.Timestamp
        Start date of the curve.
    end_date : pd.Timestamp
        End date of the curve.
    winter_value : float
        Minimum value (attained in mid-winter, ~Dec 21).
    summer_value : float
        Maximum value (attained in mid-summer, ~Jun 21).
    name : str, optional
        Name of the forward curve.

    Returns
    -------
    ForwardCurve
        ForwardCurve instance with daily seasonal values.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Day of year
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])

    # Approximate peak/trough in northern hemisphere
    peak_day = 172  # ~June 21
    trough_day = 355  # ~Dec 21

    # Map day_of_year to sine function (peak at summer, trough at winter)
    theta = 2 * np.pi * (day_of_year - peak_day) / 365
    sin_norm = 0.5 * (1 + np.sin(theta))  # scale [0,1]

    # Scale to winter → summer values
    values = winter_value + sin_norm * (summer_value - winter_value)

    series = pd.Series(data=values, index=dates)
    return ForwardCurve(series=series, as_of_date=as_of_date, name=name)


if __name__ == "__main__":
    # -------------------------------
    # Example usage of ForwardCurve
    # -------------------------------

    # Define start and end dates
    as_of_date: pd.Timestamp = pd.Timestamp("2024-12-31")
    start_date: pd.Timestamp = pd.Timestamp("2025-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31")

    # Define start and end values
    start_value: float = 50.0
    end_value: float = 100.0

    # Generate linear forward curve
    linear_curve: ForwardCurve = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        start_value=start_value,
        end_value=end_value,
        shape="linear",
        name="LinearForwardCurve",
    )
    print(linear_curve)
    linear_curve.plot(title="Linear Forward Curve Example")

    # Generate contango forward curve
    contango_curve: ForwardCurve = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        start_value=start_value,
        end_value=end_value,
        shape="contango",
        name="ContangoForwardCurve",
    )
    print(contango_curve)
    contango_curve.plot(title="Contango Forward Curve Example")

    # Generate backwardation forward curve
    backwardation_curve: ForwardCurve = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        start_value=start_value,
        end_value=end_value,
        shape="backwardation",
        name="BackwardationForwardCurve",
    )
    print(backwardation_curve)
    backwardation_curve.plot(title="Backwardation Forward Curve Example")

    # Generate flat forward curve
    flat_curve: ForwardCurve = ForwardCurve.generate_curve(
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        start_value=start_value,
        end_value=end_value,
        shape="flat",
        name="FlatForwardCurve",
    )
    print(flat_curve)
    flat_curve.plot(title="Flat Forward Curve Example")

    seasonal_curve: ForwardCurve = generate_yearly_seasonal_curve(
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        winter_value=50.0,
        summer_value=100.0,
    )
    print(seasonal_curve)
    seasonal_curve.plot(title="Seasonal Forward Curve Example")
