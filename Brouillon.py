# ==============================================================================
# NOTEBOOK BROUILLON (DRAFT) : ANALYSE DES ERREURS ET EXPÉRIMENTATIONS
# Projet Encadré par M. Abdallah Khemais
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. PRÉPARATION DE L'ENVIRONNEMENT
# ------------------------------------------------------------------------------
!pip install yfinance -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

# ------------------------------------------------------------------------------
# PARTIE A : L'ERREUR INITIALE (Modèle Paresseux / Biais de Majorité)
# ------------------------------------------------------------------------------
print("="*80)
print("ÉTAPE 1 : REPRODUCTION DE L'ERREUR AVEC LES PRIX BRUTS")
print("="*80)

# Téléchargement des données classiques
data_fausse = yf.download("BTC-USD", start="2020-01-01", end="2026-01-01")

# Target binaire classique
data_fausse['Target'] = (data_fausse['Close'].shift(-1) > data_fausse['Close']).astype(int)

# FAUTE COMMISE : Utiliser les prix bruts (Open, High, Low, Close, Volume)
features_fausses = ['Open', 'High', 'Low', 'Close', 'Volume']
df_features_fausses = data_fausse[features_fausses].values
df_target_fausses = data_fausse['Target'].values

# Split 80/20
train_size = int(len(df_features_fausses) * 0.8)
X_train_f, X_test_f = df_features_fausses[:train_size], df_features_fausses[train_size:]
y_train_f, y_test_f = df_target_fausses[:train_size], df_target_fausses[train_size:]

# Normalisation brute
scaler_f = MinMaxScaler(feature_range=(0, 1))
X_train_f_scaled = scaler_f.fit_transform(X_train_f)
X_test_f_scaled = scaler_f.transform(X_test_f)

# Fonction de séquences (60 jours)
def create_sequences(data, targets, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

X_train_f_seq, y_train_f_seq = create_sequences(X_train_f_scaled, y_train_f, 60)
X_test_f_seq, y_test_f_seq = create_sequences(X_test_f_scaled, y_test_f, 60)

# Construction d'un LSTM simple pour le test
model_fausse = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X_train_f_seq.shape[1], X_train_f_seq.shape[2])),
    Dense(1, activation='sigmoid')
])
model_fausse.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nEntraînement du modèle défaillant...")
model_fausse.fit(X_train_f_seq, y_train_f_seq, epochs=10, batch_size=32, verbose=0)

# Prédiction et constatation du bug
y_pred_f = (model_fausse.predict(X_test_f_seq) > 0.5).astype(int)

print("\n[CONSTATATION DU BUG] Matrice de Confusion obtenue avec les prix bruts :")
print(confusion_matrix(y_test_f_seq, y_pred_f))
print("\nAnalyse critique du brouillon :")
print("-> Le modèle prédit uniquement des 0 (ou uniquement des 1). Les prix bruts écrasent le gradient.")

# ------------------------------------------------------------------------------
# PARTIE B : LA CORRECTION TECHNIQUE (Feature Engineering & Rendements)
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("ÉTAPE 2 : APPLICATION DE LA CORRECTION (RENDEMENTS & RSI)")
print("="*80)

data_propre = yf.download("BTC-USD", start="2020-01-01", end="2026-01-01")

# LA CORRECTION 1 : Passer aux rendements (variations en %) au lieu des valeurs brutes
data_propre['Returns'] = data_propre['Close'].pct_change()

# LA CORRECTION 2 : Ajouter un indicateur de dynamique borné (RSI)
delta = data_propre['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
data_propre['RSI'] = 100 - (100 / (1 + rs))
data_propre.dropna(inplace=True)

# Target
data_propre['Target'] = (data_propre['Close'].shift(-1) > data_propre['Close']).astype(int)

# Sélection des fonctionnalités corrigées
features_propres = ['Returns', 'RSI', 'Volume']
df_features_propres = data_propre[features_propres].values
df_target_propres = data_propre['Target'].values

# Split
train_size_p = int(len(df_features_propres) * 0.8)
X_train_p, X_test_p = df_features_propres[:train_size_p], df_features_propres[train_size_p:]
y_train_p, y_test_p = df_target_propres[:train_size_p], df_target_propres[train_size_p:]

# Normalisation stable
scaler_p = MinMaxScaler(feature_range=(0, 1))
X_train_p_scaled = scaler_p.fit_transform(X_train_p)
X_test_p_scaled = scaler_p.transform(X_test_p)

# LA CORRECTION 3 : Réduire la fenêtre temporelle à 30 jours pour éviter la perte d'information
X_train_p_seq, y_train_p_seq = create_sequences(X_train_p_scaled, y_train_p, 30)
X_test_p_seq, y_test_p_seq = create_sequences(X_test_p_scaled, y_test_p, 30)

# Modèle LSTM Corrigé (Identique au Notebook Final)
model_propre = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_p_seq.shape[1], X_train_p_seq.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_propre.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nEntraînement du modèle corrigé...")
model_propre.fit(X_train_p_seq, y_train_p_seq, epochs=15, batch_size=32, verbose=0)

# Prédiction finale
y_pred_p = (model_propre.predict(X_test_p_seq) > 0.5).astype(int)

print("\n" + "="*60)
print("[RÉSULTAT APRÈS CORRECTION] Matrice de Confusion Finale :")
print("="*60)
cm_finale = confusion_matrix(y_test_p_seq, y_pred_p)
print(cm_finale)

# Visualisation comparative des deux étapes pour le rapport
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(confusion_matrix(y_test_f_seq, y_pred_f), annot=True, fmt='d', cmap='Reds', ax=ax[0], cbar=False)
ax[0].set_title("Avant correction : Biais de majorité (Hasard)")
ax[0].set_xlabel("Prédiction")
ax[0].set_ylabel("Vraie Valeur")

sns.heatmap(cm_finale, annot=True, fmt='d', cmap='Greens', ax=ax[1], cbar=False)
ax[1].set_title("Après correction : Répartition équilibrée")
ax[1].set_xlabel("Prédiction")
ax[1].set_ylabel("Vraie Valeur")
plt.tight_layout()
plt.show()

print("\n[OK] Expérimentations terminées. Ce notebook brouillon valide scientifiquement notre démarche d'optimisation.")