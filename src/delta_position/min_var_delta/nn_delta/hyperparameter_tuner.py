from typing import Dict, List, Any, Optional
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from delta_position.min_var_delta.nn_delta.nn_delta_model import NNDeltaModel


class HyperparameterTuner:
    """
    Simple hyperparameter tuning class for NNDeltaModel using grid search.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Feature matrix for training (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Target polynomial_regression_delta values (n_samples, output_dim).
    device : str, optional
        PyTorch device to use ("cuda" or "cpu"). Defaults to CUDA if available.
    """

    def __init__(self, X: Any, y: Any, device: Optional[str] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def tune(
        self,
        param_grid: Dict[str, List[Any]],
        batch_size: int = 128,
        epochs: int = 20,
        criterion: nn.Module = nn.MSELoss(),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a grid search over hyperparameter combinations.

        Parameters
        ----------
        param_grid : dict
            Dictionary of hyperparameters with lists of values to try. Example:
            {
                "hidden_dim": [16, 32, 64],
                "lr": [1e-3, 1e-4],
                "weight_decay": [0.0, 1e-5]
            }
        batch_size : int
            Batch size for training.
        epochs : int
            Number of epochs per combination.
        criterion : nn.Module
            Loss function (default: MSELoss).
        verbose : bool
            Print progress for each combination.

        Returns
        -------
        Dict[str, Any]
            Best hyperparameter combination found.
        """
        best_loss = float("inf")
        best_params: Optional[Dict[str, Any]] = None
        dataset = TensorDataset(self.X, self.y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            hidden_dim = params.get("hidden_dim", 64)
            lr = params.get("lr", 1e-3)
            weight_decay = params.get("weight_decay", 0.0)

            model = NNDeltaModel(input_dim=self.X.shape[1], hidden_dim=hidden_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            model.train()

            # Training loop
            for epoch in range(epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataset)
            if verbose:
                print(f"Params: {params}, Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params

        print(f"\nBest Hyperparameters: {best_params}, Loss: {best_loss:.6f}")
        return best_params


if __name__ == "__main__":
    import numpy as np

    # ----------------------------
    # Simple sanity test
    # ----------------------------
    n_samples = 100
    n_features = 6
    output_dim = 2

    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, output_dim)

    param_grid = {
        "hidden_dim": [16, 32],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-5]
    }

    tuner = HyperparameterTuner(X, y)
    best_params = tuner.tune(param_grid, batch_size=16, epochs=5, verbose=True)
    print("Best params found:", best_params)
