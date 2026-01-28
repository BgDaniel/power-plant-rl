import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nn_delta.nn_delta_model import NNDeltaModel


def train_model(X, y, hidden_dim=64, lr=1e-3, batch_size=128, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NNDeltaModel(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(dataset):.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_X", type=str, required=True)
    parser.add_argument("--train_y", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_output", type=str, default="/opt/ml/model/")
    args = parser.parse_args()

    # Load dataset
    X = np.load(args.train_X)
    y = np.load(args.train_y)

    # Train model
    model = train_model(X, y, hidden_dim=args.hidden_dim, lr=args.lr,
                        batch_size=args.batch_size, epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), args.model_output + "/nn_delta_model.pt")
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
