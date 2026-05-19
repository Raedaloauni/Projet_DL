import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IA Crypto Predictor", layout="wide")

st.title("📈 Application de Soutenance : IA Crypto Predictor")
st.markdown("""
Cette application utilise un modèle **LSTM (Deep Learning)** pour prédire si le prix du Bitcoin va **monter** ou **descendre** demain.
Encadré par M. Abdallah Khemais.
""")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modele_trading_lstm.h5")

try:
    model = load_my_model()
    st.sidebar.success("✅ Modèle LSTM chargé avec succès")
except:
    st.sidebar.error("❌ Modèle non trouvé. Assurez-vous que 'modele_trading_lstm.h5' est dans le même dossier.")

# --- BARRE LATÉRALE ---
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Choisissez l'actif", ["BTC-USD", "ETH-USD"])
period = st.sidebar.slider("Nombre de jours à afficher", 30, 200, 100)

# --- TÉLÉCHARGEMENT DES DONNÉES EN TEMPS RÉEL ---
data = yf.download(ticker, period="1y", interval="1d")

# --- PRÉTRAITEMENT (Identique au Notebook) ---
def get_prediction_data(df):
    # Calcul des rendements et RSI
    df['Returns'] = df['Close'].pct_change()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    
    # Sélection des 30 derniers jours pour la prédiction
    features = ['Returns', 'RSI', 'Volume']
    last_30_days = df[features].tail(30).values
    
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_30_days_scaled = scaler.fit_transform(last_30_days)
    
    # Formatage pour le LSTM (1, 30, 3)
    return np.expand_dims(last_30_days_scaled, axis=0)

# --- INTERFACE PRINCIPALE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Historique récent de {ticker}")
    fig = go.Figure(data=[go.Candlestick(x=data.index[-period:],
                open=data['Open'][-period:],
                high=data['High'][-period:],
                low=data['Low'][-period:],
                close=data['Close'][-period:])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Prédiction IA")
    if st.button("Lancer l'analyse du marché"):
        with st.spinner("L'IA analyse les 30 derniers jours..."):
            # Préparer les données
            input_data = get_prediction_data(data)
            
            # Inférence
            prediction_prob = model.predict(input_data)[0][0]
            
            # Affichage du résultat
            if prediction_prob > 0.5:
                st.success(f"🟩 **HAUSSE PRÉVUE** ({prediction_prob:.1%})")
                st.metric(label="Tendance", value="ACHAT", delta="UP")
                st.info("Le modèle suggère que le prix sera plus élevé demain.")
            else:
                st.error(f"🟥 **BAISSE PRÉVUE** ({1-prediction_prob:.1%})")
                st.metric(label="Tendance", value="VENTE", delta="-DOWN")
                st.info("Le modèle suggère que le prix sera plus bas demain.")

st.divider()
st.caption("Note : Ce modèle est un projet académique. Ne l'utilisez pas pour faire du vrai trading.")