import pandas as pd
import numpy as np
from typing import Optional


class MonthlyForward:
    def __init__(
        self,
        delivery_start: pd.Timestamp,
        strike: float,
        nominal: float,
        long: bool = True
    ):
        """
        Represents a monthly forward contract.

        Args:
            delivery_start: Start of the delivery period (must be the first day of a month).
            strike: Strike price of the forward.
            nominal: Contract size.
            long: True for long position (default), False for short position.
        """
        # Ensure delivery_start is a Timestamp
        self.start_start = delivery_start

        # Validate that delivery_start is the 1st of the month
        if self.start.day != 1:
            raise ValueError("delivery_start must be the first day of a month")

        # Infer name as YYYY_MM
        self.name: str = self.start.strftime("%Y_%m")

        # Infer delivery_end as the last day of that month
        self.delivery_end: pd.Timestamp = self.start + pd.offsets.MonthEnd(0)

        self.delivery_days = pd.date_range(start=self.delivery_start, end=self.delivery_end, freq='D')

        # Infer maturity as the day before start
        self.maturity: pd.Timestamp = self.delivery_start - pd.Timedelta(days=1)

        # Store contract parameters
        self.nominal: float = nominal
        self.strike: float = strike
        self.long: bool = long

        # Internal multiplier (+1 for long, -1 for short)
        self._direction: int = 1 if long else -1

    def value(self, current_price: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
        """
        Compute mark-to-market value relative to the strike.

        Args:
            current_price: Current forward price(s) (scalar, numpy array, or pandas Series).

        Returns:
            Mark-to-market value(s) = ± nominal * (current_price - strike).
        """
        return self._direction * self.nominal * (current_price - self.strike)

    def cashflow(self, spots: pd.Series | pd.DataFrame) -> pd.Series:
        return self.nominal * (spots - self.strike)

    def __repr__(self) -> str:
        pos = "Long" if self.long else "Short"
        return (f"MonthlyForward(name={self.name}, start={self.start.date()}, "
                f"end={self.end.date()}, strike={self.strike}, nominal={self.nominal}, "
                f"position={pos})")
