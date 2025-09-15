from __future__ import annotations
from typing import Optional, Literal
import pandas as pd
import numpy as np
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

    def __init__(self, series: pd.Series, as_of_date: pd.Timestamp, name: str) -> None:
        """
        Initialize the ForwardCurve.

        Parameters
        ----------
        series : pd.Series
            Forward curve values indexed by dates.
        start_date : pd.Timestamp
            The start date of the forward curve.
        name : str
            Name of the forward curve.
        """
        self.series: pd.Series = series
        self.as_of_date: pd.Timestamp = as_of_date
        self.name: str = name

    def __getitem__(self, key) -> float:
        return self.series[key]

    def __len__(self) -> int:
        return len(self.series)

    def __repr__(self) -> str:
        return f"ForwardCurve(name={self.name}, as_of_date={self.as_of_date}, len={len(self.series)})"

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
