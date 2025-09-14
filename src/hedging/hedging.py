import numpy as np
import pandas as pd
from typing import List, Optional


from hedging.forward_product import ForwardProduct
from math_helpers import shifted_legendre_basis_dynamic


class BaseHedge:
    def __init__(
        self,
        plant_value: pd.Series,
        forwards_df: pd.DataFrame,
        degree: int = 2,
        ridge: float = 1e-6,
    ):
        """
        Base class for polynomial forward hedging.

        Args:
            plant_value: Plant values indexed by date.
            forwards_df: Forward prices indexed by date, columns = forward products.
            degree: Maximum polynomial degree per forward.
            ridge: Regularization for regression.
        """
        self.V: pd.Series = plant_value.sort_index()
        self.F: pd.DataFrame = forwards_df.sort_index()
        self.M: int = self.F.shape[1]
        self.degree: int = degree
        self.ridge: float = ridge
        self.dates: pd.Index = self.V.index
        self.alpha: Optional[np.ndarray] = None

    def build_design_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build regression design matrix. Must be implemented in derived classes.

        Returns:
            X: Regression matrix.
            y: Target vector.
        """
        raise NotImplementedError

    def fit(self) -> None:
        """
        Fit the polynomial hedge by solving the linear regression.
        """
        X, y = self.build_design_matrix()
        XtX = X.T @ X + self.ridge * np.eye(X.shape[1])
        Xty = X.T @ y
        self.alpha = np.linalg.solve(XtX, Xty)

    def compute_weights(self) -> pd.DataFrame:
        """
        Compute hedge weights for all dates using fitted coefficients.

        Returns:
            DataFrame with daily hedge weights for each forward.
        """
        if self.alpha is None:
            raise RuntimeError("Must fit first")
        weights = pd.DataFrame(index=self.dates, columns=self.F.columns, dtype=float)
        deg = self.degree
        for t in range(1, len(self.dates)):
            F_past = self.F.iloc[:t]
            x_max_t = max(F_past.max().max(), 1e-6)
            row = []
            for i, f_prev in enumerate(self.F.iloc[t - 1]):
                phi = shifted_legendre_basis_dynamic(f_prev, deg, x_max_t)
                coeff_slice = self.alpha[i * (deg + 1) : (i + 1) * (deg + 1)]
                w_i = np.dot(coeff_slice, phi)
                row.append(w_i)
            weights.iloc[t] = row
        weights.iloc[0] = 0.0
        return weights

    def simulate_cashflows(
        self,
        forwards_products: List[ForwardProduct],
        day_ahead: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Simulate financial cashflows from rebalancing the hedge.

        Args:
            forwards_products: List of ForwardProduct objects in same order as self.F columns.
            day_ahead: Optional day-ahead prices for mark-to-market P/L.

        Returns:
            DataFrame with daily hedge weights and cashflow.
        """
        if self.alpha is None:
            raise RuntimeError("Must fit first")

        weights = self.compute_weights()
        cashflow = []
        w_prev = np.zeros(self.M)

        for t in range(len(self.dates)):
            w_curr = weights.iloc[t].values.astype(float)

            # Rebalancing cost: buy/sell old positions
            if t == 0:
                rebal_cost = 0.0
            else:
                rebal_cost = 0.0
                for i, fwd in enumerate(forwards_products):
                    f_prev = self.F.iloc[t - 1, i]
                    rebal_cost += (w_curr[i] - w_prev[i]) * fwd.get_mark_to_market(
                        f_prev
                    )

            # Mark-to-market for held positions vs day-ahead price
            if day_ahead is not None and t > 0:
                mtm = 0.0
                for i, fwd in enumerate(forwards_products):
                    mtm += w_curr[i] * fwd.get_mark_to_market(day_ahead.iloc[t])
            else:
                mtm = 0.0

            total_cf = rebal_cost + mtm
            cashflow.append(total_cf)
            w_prev = w_curr

        df = weights.copy()
        df["cashflow"] = cashflow
        return df


# --- Delta hedge class -------------------------------------------------------
class DeltaHedge(BaseHedge):
    def build_design_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build regression matrix for delta hedge (differences approach).

        Returns:
            X: Regression matrix
            y: Target vector
        """
        X_list, y_list = [], []
        for t in range(1, len(self.dates)):
            deltaV = self.V.iloc[t] - self.V.iloc[t - 1]
            deltaF = self.F.iloc[t] - self.F.iloc[t - 1]
            F_past = self.F.iloc[:t]
            x_max_t = max(F_past.max().max(), 1e-6)
            row = []
            for i, f_prev in enumerate(self.F.iloc[t - 1]):
                phi = shifted_legendre_basis_dynamic(f_prev, self.degree, x_max_t)
                row.extend(phi * deltaF[i])
            X_list.append(row)
            y_list.append(deltaV)
        return np.array(X_list), np.array(y_list)


# --- Absolute / cashflow hedge class ----------------------------------------
class AbsoluteHedge(BaseHedge):
    def __init__(
        self,
        plant_value: pd.Series,
        forwards_df: pd.DataFrame,
        day_ahead: pd.Series,
        degree: int = 2,
        ridge: float = 1e-6,
    ):
        """
        Absolute value hedge minimizing V_t - sum(weights * DA_t).

        Args:
            plant_value: Plant values indexed by date.
            forwards_df: Forward prices indexed by date.
            day_ahead: Day-ahead prices for regression.
            degree: Max polynomial degree.
            ridge: Regularization.
        """
        super().__init__(plant_value, forwards_df, degree, ridge)
        self.DA: pd.Series = day_ahead.sort_index()

    def build_design_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for t in range(1, len(self.dates)):
            V_t = self.V.iloc[t]
            DA_t = self.DA.iloc[t]
            F_past = self.F.iloc[:t]
            x_max_t = max(F_past.max().max(), 1e-6)
            row = []
            for i, f_prev in enumerate(self.F.iloc[t - 1]):
                phi = shifted_legendre_basis_dynamic(f_prev, self.degree, x_max_t)
                row.extend(phi * DA_t)
            X_list.append(row)
            y_list.append(V_t)
        return np.array(X_list), np.array(y_list)
