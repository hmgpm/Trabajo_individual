# Importo las librerías necesarias
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configuración visual
sns.set_theme(style="whitegrid")

# 1. CARGA DE DATOS
print("--- Cargando Datasets ---")
# Dataset de Tweets (Kaggle)
df_tweets = pd.read_csv("data/raw/Twitter_Final_data.csv", sep=",") 
# Dataset de Reddit (Kaggle)
df_reddit = pd.read_csv("data/raw/reddit_comments_combined.csv", sep=",") 

#Observamos cuales son las columnas para cada dataset
print("Columnas Tweets:", df_tweets.columns)
print("Columnas Reddit:", df_reddit.columns)

# 2. EQUILIBRADO DE MUESTRAS
# Usamos el tamaño del dataset de Reddit para igualar ambos
n_muestras = len(df_reddit)
df_tweets_sample = df_tweets.sample(n=n_muestras, random_state=42)

print(f"Datasets equilibrados a {n_muestras} filas cada uno.")

# 3. LIMPIEZA DE TEXTO
def limpiar_texto(texto):
    if not isinstance(texto, str): 
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto) # URLs
    texto = re.sub(r'\@\w+|\#','', texto) # Menciones/Hashtags
    texto = re.sub(r'[^a-z\s]', '', texto) # Solo letras
    return texto.strip()

print("Limpiando textos...")

df_tweets_sample['clean_text'] = df_tweets_sample['content'].apply(limpiar_texto)
df_reddit['clean_text'] = df_reddit['Comment Body'].apply(limpiar_texto)

# 4. ANÁLISIS DE SENTIMIENTO (VADER)
analyzer = SentimentIntensityAnalyzer()

def obtener_score(texto):
    return analyzer.polarity_scores(texto)['compound']

print("Calculando polaridad con VADER...")

df_tweets_sample['sentiment'] = df_tweets_sample['clean_text'].apply(obtener_score)
df_reddit['sentiment'] = df_reddit['clean_text'].apply(obtener_score)

# 5. CÁLCULO DE MÉTRICAS AVANZADAS (Para tu nota de 10)
# Calculo la polarización (Desviación Estándar)
pol_tweets = df_tweets_sample['sentiment'].std()
pol_reddit = df_reddit['sentiment'].std()

# Calculo la positividad media
mean_tweets = df_tweets_sample['sentiment'].mean()
mean_reddit = df_reddit['sentiment'].mean()

print("\n--- RESULTADOS DEL ANÁLISIS ---")
print(f"Polarización en Twitter (Tweets): {pol_tweets:.4f}")
print(f"Polarización en Reddit (Discusión): {pol_reddit:.4f}")

print(f"Media Twitter: {mean_tweets:.4f}")
print(f"Media Reddit: {mean_reddit:.4f}")

# 6. VISUALIZACIÓN COMPARATIVA
fig, ax = plt.subplots(figsize=(10, 6))

# Creamos un dataframe temporal para graficar comparativamente
df_total = pd.concat([
    pd.DataFrame({'Score': df_tweets_sample['sentiment'], 'Fuente': 'Twitter (Tweets)'}),
    pd.DataFrame({'Score': df_reddit['sentiment'], 'Fuente': 'Reddit (Discusión)'})
])

sns.boxplot(data=df_total, x='Fuente', y='Score', palette="Set2")
plt.title("Comparativa de Sentimiento: Twitter vs Reddit")
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.show()

# 7. EXPORTACIÓN
df_tweets_sample.to_csv("data/clean/data_processed_tweets.csv", index=False)
df_reddit.to_csv("data/clean/data_processed_reddit.csv", index=False)
print("¡Archivos listos para el Dashboard!")