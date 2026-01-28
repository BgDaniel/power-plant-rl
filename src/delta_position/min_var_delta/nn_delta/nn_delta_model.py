import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, Optional
import numpy as np


class NNDeltaModel(nn.Module):
    """
    Feed-forward neural network to approximate hedge deltas for power and coal.

    This network takes 4 observable inputs per delivery month:
        1. fwd_power : forward power price
        2. fwd_coal  : forward coal price
        3. spread    : fwd_power - efficiency * fwd_coal
        4. t         : delivery month (numeric)

    Output is 2-dimensional:
        - delta_power : predicted hedge delta for power
        - delta_coal  : predicted hedge delta for coal

    Hyperparameters and training options are stored in the constructor
    for easier hyperparameter tuning.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        activation: Literal["relu", "softplus", "tanh", "gelu"] = "softplus",
        batch_size: int = 128,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the NNDeltaModel.

        Parameters
        ----------
        hidden_dim : int, default=64
            Number of neurons in each hidden layer.
        activation : {'relu', 'softplus', 'tanh', 'gelu'}, default='softplus'
            Activation function between layers.
        batch_size : int, default=128
            Batch size for training.
        epochs : int, default=20
            Number of epochs.
        lr : float, default=1e-3
            Learning rate.
        weight_decay : float, default=0.0
            L2 regularization.
        device : str or None, optional
            Torch device ("cpu" or "cuda"); auto-select if None.
        save_path : str or None, optional
            Path to save the trained model after training.
        """
        super().__init__()

        # Training hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        # Map string activation to PyTorch module
        act_fn: nn.Module = {
            "relu": nn.ReLU(),
            "softplus": nn.Softplus(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }[activation]

        # Define network architecture
        self.net: nn.Module = nn.Sequential(
            nn.Linear(4, hidden_dim),  # input layer: 4 features
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),  # hidden layer
            act_fn,
            nn.Linear(hidden_dim, 2),  # output layer: delta_power, delta_coal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, 4)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n_samples, 2) containing delta_power and delta_coal
        """
        return self.net(x)

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        device: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Train the neural network using minimum-variance loss.

        The loss is computed as:
            y_hat = beta_power * delta_power + beta_coal * delta_coal
            loss = MSE(y_hat, y_cashflow)

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, 4)
        y : np.ndarray
            Target cashflows (n_samples,)
        beta_power : np.ndarray
            Beta weights for power (n_samples,)
        beta_coal : np.ndarray
            Beta weights for coal (n_samples,)
        batch_size : int, optional
            Batch size (defaults to self.batch_size)
        epochs : int, optional
            Number of epochs (defaults to self.epochs)
        lr : float, optional
            Learning rate (defaults to self.lr)
        weight_decay : float, optional
            Weight decay (defaults to self.weight_decay)
        device : str, optional
            Torch device (defaults to self.device)
        save_path : str, optional
            Path to save trained model (defaults to self.save_path)
        """
        batch_size = batch_size or self.batch_size
        epochs = epochs or self.epochs
        lr = lr or self.lr
        weight_decay = weight_decay or self.weight_decay
        device = device or self.device
        save_path = save_path or self.save_path

        self.to(device)
        self.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        beta_p_tensor = torch.tensor(beta_power, dtype=torch.float32).to(device)
        beta_c_tensor = torch.tensor(beta_coal, dtype=torch.float32).to(device)

        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_tensor, beta_p_tensor, beta_c_tensor
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0

            for xb, yb, bp, bc in loader:
                optimizer.zero_grad()
                delta_pred = self.forward(xb)
                y_hat = bp * delta_pred[:, 0] + bc * delta_pred[:, 1]
                loss = criterion(y_hat, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataset)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if save_path:
            torch.save(self.state_dict(), save_path)

            print(f"Trained model saved at {save_path}")
