import pandas as pd
import numpy as np

# Chargement des données
df = pd.read_csv("ETTm1.csv")
print(df.head())
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

# Vérification des statistiques de base
print(df.describe())

# Normalisation des données
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(columns=['date']))
print(scaled_data)

# Visualisation des valeurs normalisées
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['date'], scaled_data)
plt.legend(df.columns[1:], loc='upper right')
plt.title('Valeurs Normalisées des Caractéristiques')
plt.xlabel('Date')
plt.ylabel('Valeurs Normalisées')
plt.show()