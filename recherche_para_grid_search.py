import torch as t
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid

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

class GenericNBeatsBlock(t.nn.Module):
    def __init__(self, input_size: int, forecast_size: int, layers: int, layer_size: int, num_features: int):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_features = num_features
        
        # Première couche
        self.fc1 = t.nn.Linear(input_size * num_features, layer_size)
        
        # Couches cachées
        self.hidden_layers = t.nn.ModuleList()
        for _ in range(layers - 1):
            self.hidden_layers.append(t.nn.Linear(layer_size, layer_size))
        
        # Projections pour backcast et forecast
        self.backcast_layer = t.nn.Linear(layer_size, input_size * num_features)
        self.forecast_layer = t.nn.Linear(layer_size, forecast_size * num_features)
        
    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # x est de forme [batch_size, input_size, num_features]
        batch_size = x.size(0)
        
        # Aplatir les dimensions input_size et num_features
        block_input = x.reshape(batch_size, -1)
        
        # Première couche
        block_input = t.relu(self.fc1(block_input))
        
        # Couches cachées
        for layer in self.hidden_layers:
            block_input = t.relu(layer(block_input))
        
        # Génération des backcast et forecast
        backcast = self.backcast_layer(block_input)
        forecast = self.forecast_layer(block_input)
        
        # Restructurer les sorties
        backcast = backcast.reshape(batch_size, self.input_size, self.num_features)
        forecast = forecast.reshape(batch_size, self.forecast_size, self.num_features)
        
        return backcast, forecast

class GenericNBeats(t.nn.Module):
    def __init__(self, input_size: int, forecast_size: int, num_blocks: int, 
                 layers_per_block: int, layer_size: int, num_features: int):
        super().__init__()
        self.blocks = t.nn.ModuleList([
            GenericNBeatsBlock(
                input_size=input_size,
                forecast_size=forecast_size,
                layers=layers_per_block,
                layer_size=layer_size,
                num_features=num_features
            ) for _ in range(num_blocks)
        ])
        self.forecast_size = forecast_size
        self.num_features = num_features
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # x est de forme [batch_size, input_size, num_features]
        residuals = x
        forecast = t.zeros(x.size(0), self.forecast_size, self.num_features).to(x.device)
        
        # Propagation à travers les blocs
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast  # Connexion résiduelle
            forecast = forecast + block_forecast  # Agrégation des prévisions
            
        return forecast

def init_weights(m):
    if isinstance(m, t.nn.Linear):
        t.nn.init.orthogonal_(m.weight)
        if m.bias is not None, t.nn.init.zeros_(m.bias)

def load_dataset(csv_path, input_size, forecast_size):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.drop(columns=['date'])
    data = df.values.astype(np.float32)
    
    # Normaliser les données
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
    def __init__(self, patience=15, min_delta=0.001):
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


def train_generic_nbeats(dataset, input_size, forecast_size, epochs=100, batch_size=32, lr=0.001, loss_fn='mse'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Obtenir le nombre de caractéristiques à partir du premier élément du dataset
    x_sample, _ = dataset[0]
    num_features = x_sample.shape[-1]  # Nombre de colonnes dans vos données
    
    # Création du modèle générique
    model = GenericNBeats(
        input_size=input_size,
        forecast_size=forecast_size,
        num_blocks=5,
        layers_per_block=4,
        layer_size=256,
        num_features=num_features
    )
    
    # Initialisation des poids
    model.apply(init_weights)
    
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs, pct_start=0.2)
    
    # Choix de la fonction de perte
    if loss_fn == 'smape':
        loss_function = smape_loss
    elif loss_fn == 'mape':
        loss_function = mape_loss
    else:
        loss_function = t.nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    writer = SummaryWriter(log_dir=f"runs/generic_nbeats")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        iteration = 0  # Initialiser un compteur d'itérations
        
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            
            loss.backward()
            # Gradient clipping pour éviter l'explosion des gradients
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Ajouter le learning rate au TensorBoard
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning_Rate', current_lr, epoch * len(dataloader) + iteration)
            
            iteration += 1  # Incrémenter le compteur d'itérations
        
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        with t.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(dataloader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Monitoring des gradients et poids pour diagnostiquer le vanishing gradient
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                writer.add_histogram(f'weights/{name}', param, epoch)
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
    
    writer.close()
    return model

def grid_search(train_dataset, val_dataset, input_size, forecast_size, param_grid, epochs=100, loss_fn='mse'):
    best_params = None
    best_loss = float('inf')
    
    for params in ParameterGrid(param_grid):
        print(f"Training with parameters: {params}")
        model = train_generic_nbeats(
            dataset=train_dataset,
            input_size=input_size,
            forecast_size=forecast_size,
            epochs=epochs,
            batch_size=params['batch_size'],
            lr=params['lr'],
            loss_fn=loss_fn
        )
        
        # Évaluation du modèle
        _, _, val_loss = evaluate_model(model, val_dataset, loss_fn=loss_fn)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
        
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_loss}")
    return best_params

def predict(model, x):
    model.eval()
    with t.no_grad():
        return model(x)

def evaluate_model(model, test_dataset, loss_fn='mse'):
    dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    
    if loss_fn == 'smape':
        loss_function = smape_loss
    elif loss_fn == 'mape':
        loss_function = mape_loss
    else:
        loss_function = t.nn.MSELoss()
    
    total_loss = 0
    predictions = []
    actuals = []
    
    with t.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            total_loss += loss.item()
            
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    print(f"Test Loss ({loss_fn}): {avg_loss}")
    
    return np.vstack(predictions), np.vstack(actuals), avg_loss

if __name__ == "__main__":
    # Exemple d'utilisation
    input_size = 7
    forecast_size = 12
    
    # Chargement des données
    dataset = load_dataset("ETTm1.csv", input_size, forecast_size)
    
    # Séparation en train/validation/test
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = t.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Définir la grille de paramètres
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'num_blocks': [3, 5, 7],
        'layer_size': [128, 256, 512],
        'layers_per_block': [2, 4, 6],
        'batch_size': [16, 32, 64]
    }
    
    # Exécuter la recherche en grille
    best_params = grid_search(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_size=input_size,
        forecast_size=forecast_size,
        param_grid=param_grid,
        epochs=50,
        loss_fn='smape'
    )
    
    # Entraîner le modèle avec les meilleurs paramètres
    model = train_generic_nbeats(
        dataset=train_dataset,
        input_size=input_size,
        forecast_size=forecast_size,
        epochs=100,
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        loss_fn='smape'
    )
    
    # Évaluation du modèle
    predictions, actuals, test_loss = evaluate_model(model, test_dataset, loss_fn='smape')
    
    # Sauvegarde du modèle
    t.save(model.state_dict(), "nbeats_generic_model.pth")
    
    print("Entraînement et évaluation terminés")