# Projet_DL
# 📈 IA Crypto Predictor : Analyse & Prédiction du Bitcoin par Deep Learning

### Projet Académique de Deep Learning
**Encadré par :** M. Abdallah Khemais  
**Réalisé par :** Raed Alouani et Amira Farrah

---

## 🎯 Présentation du Projet
Ce projet a pour objectif d'appliquer les architectures de Deep Learning aux séries temporelles financières, plus particulièrement pour prédire la tendance du **Bitcoin (BTC-USD)**. 

Plutôt que de tenter de prédire un prix brut exact (tâche hautement instable), l'approche retenue est une **classification binaire** : l'intelligence artificielle analyse l'historique récent du marché pour estimer la probabilité que le prix de clôture du lendemain soit en **Hausse (1)** ou en **Baisse (0)**.

Une interface web interactive développée avec **Streamlit** permet de visualiser les prédictions du meilleur modèle en temps réel.

---

## 🛠️ Pipeline de Traitement des Données (Data Pipeline)

Le traitement des données suit un flux rigoureux afin de garantir la convergence des modèles et d'éviter les pièges classiques de la finance (surapprentissage et biais de majorité) :

1. **Collecte de données :** Récupération automatique de l'historique des prix (OHLCV) du Bitcoin via l'API `yfinance` de 2020 à 2026.
2. **Feature Engineering (Solution Anti-Biais) :** Les prix bruts étant trop chaotiques, nous les transformons en **Rendements Journaliers (variations en %)** et intégrons l'indicateur de momentum **RSI (Relative Strength Index)** pour donner un contexte macroscopique à l'IA.
3. **Split Chronologique :** Séparation des données en **80% Entraînement / 20% Test**. Le mélange aléatoire (*shuffle*) est strictement banni pour respecter la flèche du temps.
4. **Normalisation :** Application d'un `MinMaxScaler` (0, 1) sur les caractéristiques pour stabiliser la descente de gradient.
5. **Fenêtrage Temporel :** Découpage des données en séquences glissantes de **30 jours** pour prédire le comportement du 31ème jour.

---

## 🔬 Architectures Comparées

Pour répondre aux exigences du barème, nous avons implémenté et comparé deux architectures majeures de Deep Learning :

### 1. LSTM (Long Short-Term Memory)
Un réseau récurrent profond équipé de couches de `BatchNormalization` (stabilisation du gradient) et de `Dropout` (régularisation à 30%). Il traite les données de façon séquentielle pour mémoriser les patterns temporels.

### 2. Transformer (Mécanisme d'Attention)
Une architecture moderne basée sur une couche `MultiHeadAttention` (4 têtes) et une normalisation par couche (`LayerNormalization`). Elle tente de l'analyser l'ensemble de la séquence de 30 jours de manière globale.

---

## 📊 Résultats & Analyse Critique

L'évaluation sur l'ensemble de test (405 jours de marché non vus lors de l'entraînement) a donné les résultats suivants :

| Métrique / Modèle | LSTM | Transformer |
| :--- | :---: | :---: |
| **Accuracy Globale** | **51.85%** | 50.86% |
| **F1-Score Baisse (0)**| **0.55** | 0.67 |
| **F1-Score Hausse (1)**| **0.49** | 0.06 |

### Synthèse de la confrontation :
* **Le LSTM est désigné vainqueur.** Bien que son exactitude globale soit de 51.85% (performance réaliste en finance), sa matrice de confusion montre un modèle **très équilibré** capable de prédire efficacement les hausses (92) comme les baisses (118).
* **Le Transformer échoue** en développant un **biais de majorité**. Il prédit presque systématiquement une baisse (396 fois sur 405), menant à un F1-score quasi nul pour la hausse (0.06). Cette architecture très gourmande souffre ici du manque de données (*Data Scarcity*) inhérent aux bougies journalières.

---

## 🖥️ Application Web Interactive (Streamlit)

Le modèle LSTM entraîné a été sauvegardé au format `.h5` et déployé au sein d'une application de démonstration.

### Fonctionnalités de l'application :
* Affichage en direct du graphique en chandeliers japonais (*Candlestick chart*) de la crypto-monnaie via **Plotly**.
* Bouton d'inférence déclenchant l'analyse des 30 derniers jours par le modèle LSTM.
* Restitution dynamique d'un **Signal d'Achat (Acheter/Vendre)** basé sur la probabilité calculée par l'IA.

---

## 📁 Structure du Dépôt

```text
├── app.py                     # Code principal de l'application Streamlit
├── requirements.txt           # Dépendances requises pour le serveur Cloud
├── meilleur_modele_lstm.h5     # Fichier du modèle LSTM final entraîné
└── README.md                  # Documentation du projet
