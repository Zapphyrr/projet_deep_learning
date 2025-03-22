import torch as t
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from typing import Tuple
from torch.optim.lr_scheduler import OneCycleLR
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

class GenericNBeatsBlock(t.nn.Module):
    def __init__(self, input_size: int, forecast_size: int, layers: int, layer_size: int, num_features: int):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_features = num_features
        
        # Première couche
        self.fc1 = t.nn.Linear(input_size * num_features, layer_size)
        self.bn1 = t.nn.BatchNorm1d(layer_size)
        
        # Couches cachées avec BatchNorm
        self.hidden_layers = t.nn.ModuleList()
        self.bn_layers = t.nn.ModuleList()
        
        for _ in range(layers - 1):
            self.hidden_layers.append(t.nn.Linear(layer_size, layer_size))
            self.bn_layers.append(t.nn.BatchNorm1d(layer_size))
        
        # Ajout de Dropout pour la régularisation
        self.dropout = t.nn.Dropout(0.2)
        
        # Projections pour backcast et forecast
        self.backcast_layer = t.nn.Linear(layer_size, input_size * num_features)
        self.forecast_layer = t.nn.Linear(layer_size, forecast_size * num_features)
        
    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # x est de forme [batch_size, input_size, num_features]
        batch_size = x.size(0)
        
        # Aplatir les dimensions input_size et num_features
        block_input = x.reshape(batch_size, -1)
        
        # Première couche avec BatchNorm
        block_input = self.fc1(block_input)
        block_input = self.bn1(block_input)
        block_input = t.relu(block_input)
        
        # Couches cachées avec BatchNorm
        for i, layer in enumerate(self.hidden_layers):
            block_input = layer(block_input)
            block_input = self.bn_layers[i](block_input)
            block_input = t.relu(block_input)
            block_input = self.dropout(block_input)  # Appliquer le dropout
        
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
        if m.bias is not None:
            t.nn.init.zeros_(m.bias)

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

def test_overfitting(dataset, input_size, forecast_size, sample_size=32, epochs=500, lr=0.001, loss_fn='mse'):
    """
    Test la capacité du modèle à apprendre en l'entraînant sur un petit échantillon fixe
    avec des modifications pour éviter le blocage de la perte à 200
    """
    # Créer un petit échantillon fixe pour tester l'overfitting
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    small_dataset = Subset(dataset, indices)
    
    # Créer un DataLoader sans shuffle pour garantir que les mêmes données sont vues à chaque époque
    dataloader = DataLoader(small_dataset, batch_size=sample_size, shuffle=False)
    
    # Obtenir le nombre de caractéristiques
    x_sample, _ = dataset[0]
    num_features = x_sample.shape[-1]
    
    # Création d'un modèle avec plus de capacité pour faciliter l'overfitting
    model = GenericNBeats(
        input_size=input_size,
        forecast_size=forecast_size,
        num_blocks=6,  # Réduit de 8 à 6 pour diminuer le risque de gradient explosif
        layers_per_block=4,  # Réduit de 5 à 4
        layer_size=384,  # Réduit de 512 à 384
        num_features=num_features
    )
    
    # Modifier l'initialisation des poids pour Kaiming au lieu d'Orthogonal
    def init_weights_kaiming(m):
        if isinstance(m, t.nn.Linear):
            t.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                t.nn.init.zeros_(m.bias)
    
    # Appliquer la nouvelle initialisation des poids
    model.apply(init_weights_kaiming)
    
    # Optimiseur avec un taux d'apprentissage plus modéré
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Ajout de weight_decay pour la régularisation
    
    # Ajouter un scheduler pour réduire progressivement le learning rate
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Choix de la fonction de perte modifiée
    if loss_fn == 'smape':
        def modified_smape_loss(y_pred, y_true):
            # Version modifiée de SMAPE qui n'est pas limitée à 200
            return t.mean(t.abs(y_pred - y_true) / (t.abs(y_true) + 1e-8)) * 100
        loss_function = modified_smape_loss
    elif loss_fn == 'mape':
        loss_function = mape_loss
    else:
        loss_function = t.nn.MSELoss()
    
    writer = SummaryWriter(log_dir=f"runs/overfitting_test_modified")
    
    # Stocker les batch pour visualisation
    x_batch_vis = None
    y_batch_vis = None
    
    # Variables pour suivre l'évolution de la perte
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(epochs):
        model.train()
        
        # Puisque nous n'avons qu'un seul batch, on peut simplifier
        for x_batch, y_batch in dataloader:
            if x_batch_vis is None:
                x_batch_vis = x_batch.clone()
                y_batch_vis = y_batch.clone()
                
            optimizer.zero_grad()
            
            # Ajout d'une vérification pour les NaN dans les entrées
            if t.isnan(x_batch).any() or t.isnan(y_batch).any():
                print("Attention: NaN détecté dans les données d'entrée!")
                continue
            
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            
            # Vérifier si la perte est NaN ou Inf
            if not t.isfinite(loss):
                print(f"La perte est devenue {loss.item()} à l'époque {epoch + 1}. Arrêt de l'entraînement.")
                break
            
            loss.backward()
            
            # Ajout du gradient clipping pour éviter l'explosion des gradients
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Mettre à jour le learning rate avec le scheduler
            scheduler.step(loss)
            
            # Afficher la perte et les détails supplémentaires pour le débogage
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            # Surveillance des gradients pour débogage
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar('Gradients/norm', total_grad_norm, epoch)
            
            # Si la perte est très basse, on peut considérer que le modèle a réussi à faire de l'overfitting
            if loss.item() < 1.0:  # Seuil de succès
                print(f"Overfitting réussi à l'époque {epoch + 1} avec loss={loss.item():.6f}")
                
                # Afficher quelques prédictions vs réalité
                model.eval()
                with t.no_grad():
                    final_preds = model(x_batch_vis)
                
                # Afficher les résultats pour quelques exemples
                for i in range(min(3, len(x_batch_vis))):
                    print(f"\nExemple {i+1}:")
                    print(f"Entrée: {x_batch_vis[i, -5:, 0].numpy()}")  # Dernières 5 valeurs de la première feature
                    print(f"Cible: {y_batch_vis[i, :5, 0].numpy()}")    # 5 premières valeurs prédites
                    print(f"Prédiction: {final_preds[i, :5, 0].numpy()}")
                
                writer.close()
                return model, loss.item()
            
            # Vérifier si la perte s'améliore
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Si la perte ne s'améliore pas pendant 50 époques, réinitialiser l'optimiseur
            if no_improvement_count >= 50:
                print(f"Pas d'amélioration depuis 50 époques. Réinitialisation de l'optimiseur.")
                optimizer = t.optim.Adam(model.parameters(), lr=lr * 0.5, weight_decay=1e-5)
                scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=20, verbose=True
                )
                no_improvement_count = 0
    
    # Évaluation finale
    model.eval()
    with t.no_grad():
        final_preds = model(x_batch_vis)
        final_loss = loss_function(final_preds, y_batch_vis)
        
    print(f"\nPerte finale après {epochs} époques: {final_loss.item():.6f}")
    
    # Afficher les résultats pour quelques exemples
    for i in range(min(3, len(x_batch_vis))):
        print(f"\nExemple {i+1}:")
        print(f"Entrée: {x_batch_vis[i, -5:, 0].numpy()}")  # Dernières 5 valeurs de la première feature
        print(f"Cible: {y_batch_vis[i, :5, 0].numpy()}")    # 5 premières valeurs prédites
        print(f"Prédiction: {final_preds[i, :5, 0].numpy()}")
    
    writer.close()
    return model, final_loss.item()

if __name__ == "__main__":
    # Paramètres
    input_size = 7
    forecast_size = 12
    
    # Chargement des données
    dataset = load_dataset("ETTm1.csv", input_size, forecast_size)
    
    # Test d'overfitting sur un petit échantillon
    print("=== TEST D'OVERFITTING SUR UN PETIT ÉCHANTILLON ===")
    overfitted_model, final_loss = test_overfitting(
        dataset=dataset,
        input_size=input_size,
        forecast_size=forecast_size,
        sample_size=32,     # Très petit échantillon
        epochs=500,         # Beaucoup d'époques
        lr=0.001,           # Taux d'apprentissage modéré
        loss_fn='mse'       # Utiliser MSE au lieu de SMAPE pour commencer
    )
    
    # Si MSE fonctionne bien, essayer avec la version modifiée de SMAPE
    if final_loss < 0.1:  # Bon signe avec MSE
        print("\n=== ESSAI AVEC SMAPE MODIFIÉ ===")
        overfitted_model_smape, final_loss_smape = test_overfitting(
            dataset=dataset,
            input_size=input_size,
            forecast_size=forecast_size,
            sample_size=32,
            epochs=500,
            lr=0.001,
            loss_fn='smape'  # Utiliser la version modifiée de SMAPE
        )
        
        # Sauvegarder le modèle overfitté
        t.save(overfitted_model_smape.state_dict(), "nbeats_overfitted_model_smape.pth")
    
    # Sauvegarder le modèle overfitté avec MSE
    t.save(overfitted_model.state_dict(), "nbeats_overfitted_model.pth")
    
    print("\nTest d'overfitting terminé!")
    print(f"Perte finale: {final_loss:.6f}")
    
    # Critère de succès adapté selon la fonction de perte utilisée
    success_threshold = 0.1 if loss_fn == 'mse' else 5.0
    if final_loss < success_threshold:
        print("✓ Le modèle est capable d'apprendre correctement les patterns des données.")
    else:
        print("✗ Le modèle a des difficultés à apprendre même sur un petit échantillon fixe.")
        print("  Cela pourrait indiquer un problème avec l'architecture ou les hyperparamètres.")