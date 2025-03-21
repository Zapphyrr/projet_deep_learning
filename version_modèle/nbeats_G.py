import torch as t
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, input_size: int, forecast_size: int):
        self.data = data
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.samples = len(data) - input_size - forecast_size

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_size]
        y = self.data[idx + self.input_size:idx + self.input_size + self.forecast_size]
        return t.tensor(x, dtype=t.float32), t.tensor(y, dtype=t.float32)

class NBeatsBlock(t.nn.Module):
    def __init__(self, input_size, theta_size: int, layers: int, layer_size: int):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = theta_size - input_size
        self.layers = t.nn.ModuleList([
            t.nn.Linear(input_size * 7, layer_size)] +
            [t.nn.Linear(layer_size, layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = t.nn.Linear(layer_size, theta_size * 7)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x.flatten(start_dim=1)
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        theta = self.basis_parameters(block_input)
        theta = theta.view(x.size(0), -1, 7)
        backcast = theta[:, :self.input_size]
        forecast = theta[:, self.input_size:]
        return backcast, forecast

class NBeats(t.nn.Module):
    def __init__(self, blocks: t.nn.ModuleList, forecast_size: int):
        super().__init__()
        self.blocks = blocks
        self.forecast_size = forecast_size

    def forward(self, x: t.Tensor) -> t.Tensor:
        residuals = x
        forecast = t.zeros(x.size(0), self.forecast_size, x.size(2)).to(x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            if backcast.shape != residuals.shape:
                backcast = backcast[:, :residuals.shape[1], :residuals.shape[2]]
            residuals = residuals - backcast
            forecast = forecast + block_forecast[:, :self.forecast_size]
        return forecast

def create_nbeats_g(input_size: int, forecast_size: int, num_blocks: int, layers_per_block: int, layer_size: int):
    blocks = t.nn.ModuleList([
        NBeatsBlock(input_size=input_size, theta_size=input_size + forecast_size,
                    layers=layers_per_block, layer_size=layer_size)
        for _ in range(num_blocks)
    ])
    return NBeats(blocks, forecast_size=forecast_size)

def load_dataset(csv_path, input_size, forecast_size):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.drop(columns=['date'])
    data = df.values.astype(np.float32)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return TimeSeriesDataset(data, input_size, forecast_size)

def smape_loss(y_pred, y_true):
    return 100 * t.mean(2 * t.abs(y_pred - y_true) / (t.abs(y_pred) + t.abs(y_true) + 1e-8))

def mape_loss(y_pred, y_true):
    return t.mean(t.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def mase_loss(y_pred, y_true, y_train, seasonality=1):
    n = y_train.size(0)
    d = t.sum(t.abs(y_train[seasonality:] - y_train[:-seasonality])) / (n - seasonality)
    errors = t.abs(y_true - y_pred)
    return t.mean(errors / d)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def train_nbeats(dataset, input_size, forecast_size, epochs=100, batch_size=32, lr=0.001, loss_fn='mse'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = create_nbeats_g(input_size, forecast_size, num_blocks=3, layers_per_block=4, layer_size=128)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    writer = SummaryWriter(log_dir=f"run")  # Initialize TensorBoard writer

    if loss_fn == 'smape':
        loss_function = smape_loss
    elif loss_fn == 'mape':
        loss_function = mape_loss
    elif loss_fn == 'mase':
        loss_function = lambda y_pred, y_true: mase_loss(y_pred, y_true, dataset.data)
    else:
        loss_function = t.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)  # Log training loss to TensorBoard

        model.eval()
        val_loss = 0
        with t.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(dataloader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Loss/val', val_loss, epoch)  # Log validation loss to TensorBoard

        scheduler.step(val_loss)

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    writer.close()  # Close the TensorBoard writer
    return model

if __name__ == "__main__":
    input_size = 7
    forecast_size = 12
    dataset = load_dataset("ETTm1.csv", input_size, forecast_size)
    model = train_nbeats(dataset, input_size, forecast_size, loss_fn='smape')  # Change loss_fn as needed
