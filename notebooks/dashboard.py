# dashboard.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Cargo datos procesados
df = pd.read_csv("twitter_processed.csv")

st.title("Análisis de sentimiento sobre IA y empleo")

# Gráfico de distribución
st.subheader("Distribución de sentimiento")

fig, ax = plt.subplots()
sns.countplot(x='sentiment_label', data=df, ax=ax)
st.pyplot(fig)

# Polaridad
st.subheader("Distribución de polaridad")

fig2, ax2 = plt.subplots()
ax2.hist(df['sentiment'], bins=30)
st.pyplot(fig2)

# Métricas
st.subheader("Métricas clave")

st.write("Media de sentimiento:", df['sentiment'].mean())
st.write("Desviación estándar:", df['sentiment'].std())