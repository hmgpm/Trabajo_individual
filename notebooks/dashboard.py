# dashboard.py

import streamlit as st
import pandas as pd
import os
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide")

# Ruta base (igual que en main)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_path = os.path.join(base_path, "data", "clean")
img_path = data_path  # las imágenes están en la misma carpeta

# -------------------------------
# CARGA DE DATOS
# -------------------------------
df = pd.read_csv(os.path.join(data_path, "data_unificado.csv"))

# -------------------------------
# COLORES
# -------------------------------
COLORES = {
    "positivo": "green",
    "negativo": "red",
    "neutro": "gold"
}

# -------------------------------
# HEADER
# -------------------------------
st.title("Análisis de Sentimiento sobre IA y Empleo")
st.subheader("Comparativa Twitter vs Reddit + Análisis Global")

# -------------------------------
# KPIs (ARRIBA COMO QUIERES)
# -------------------------------
st.markdown("## Métricas clave")

col1, col2, col3 = st.columns(3)

total = len(df)
media = df["sentiment"].mean()
std = df["sentiment"].std()

with col1:
    st.metric("Total comentarios", total)

with col2:
    st.metric("Media de sentimiento", round(media, 3))

with col3:
    st.metric("Polarización", round(std, 3))

# -------------------------------
# SECCIÓN 1: GRÁFICOS INICIALES
# -------------------------------
st.markdown("## 1. Análisis exploratorio inicial")

col1, col2 = st.columns(2)

# Función segura para cargar imágenes
def mostrar_imagen(nombre):
    ruta = os.path.join(img_path, nombre)
    if os.path.exists(ruta):
        st.image(ruta)
    else:
        st.warning(f"No se encontró {nombre}")

with col1:
    st.markdown("### Twitter")
    mostrar_imagen("sentimiento_twitter.png")

with col2:
    st.markdown("### Reddit")
    mostrar_imagen("sentimiento_reddit.png")

st.markdown("### Comparativa entre plataformas")
mostrar_imagen("comparativa_twitter_reddit.png")

# -------------------------------
# SECCIÓN 2: ANÁLISIS GLOBAL
# -------------------------------
st.markdown("## 2. Análisis global (dataset unificado)")

col1, col2 = st.columns(2)

with col1:
    mostrar_imagen("sentimiento_global.png")

with col2:
    mostrar_imagen("distribucion_polaridad_global.png")

st.markdown("### Boxplot comparativo")
mostrar_imagen("comparativa_sentimiento.png")

# -------------------------------
# SECCIÓN 3: DISTRIBUCIÓN INTERACTIVA
# -------------------------------
st.markdown("## 3. Distribución interactiva")

sent_counts = df["sentiment_label"].value_counts()

st.bar_chart(sent_counts)

# -------------------------------
# SECCIÓN 4: TABLA DE DATOS
# -------------------------------
st.markdown("## 4. Muestra de datos")

st.dataframe(df.sample(100))

# -------------------------------
# SECCIÓN 5: GEPHI (PARA TU HISTORIA)
# -------------------------------
st.markdown("## 5. Grafos de Gephi")

st.info("""
Aquí puedes añadir tus grafos exportados desde Gephi.

Idea de narrativa para tu trabajo:
- Primero: gráficos simples (arriba)
- Luego: grafos → análisis estructural más avanzado

Añade imágenes como:
- grafo_global.png
- grafo_negativo.png
- grafo_positivo.png
""")

# Ejemplo si luego añades imágenes:
mostrar_imagen("grafo_global.png")