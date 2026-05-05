# Importo las librerías necesarias
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Ruta base del proyecto (sube un nivel desde notebooks/)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuración visual
sns.set_theme(style="whitegrid")

# 1. CARGA DE DATOS
print("--- Cargando Datasets ---")
# Rutas a datos
tweets_path = os.path.join(base_path, "data", "raw", "Twitter_Final_data.csv")
reddit_path = os.path.join(base_path, "data", "raw", "reddit_comments_combined.csv")

# Dataset de Tweets (Kaggle)
df_tweets = pd.read_csv(tweets_path)
# Dataset de Reddit (Kaggle)
df_reddit = pd.read_csv(reddit_path)

#Observamos cuales son las columnas para cada dataset
print("Columnas Tweets:", df_tweets.columns)
print("Columnas Reddit:", df_reddit.columns)

# 2. LAS MUESTRAS VAN A SER DE 5000 FILAS PARA TWEETS
df_tweets_sample =  df_tweets.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Dataset de Tweets con {len(df_tweets_sample)} filas.")

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

def clasificar_sentimiento(valor):
    if valor > 0.05:
        return "positivo"
    elif valor < -0.05:
        return "negativo"
    else:
        return "neutro"

df_tweets_sample['sentiment_label'] = df_tweets_sample['sentiment'].apply(clasificar_sentimiento)
df_reddit['sentiment_label'] = df_reddit['sentiment'].apply(clasificar_sentimiento)

# 5. ANÁLISIS INDIVIDUAL

print("\n--- ANÁLISIS INDIVIDUAL ---")

# Twitter
print("\nTwitter:")
print(df_tweets_sample['sentiment_label'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=df_tweets_sample,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'red','neutro':'yellow','positivo':'green'})
plt.title("Sentimiento en Twitter")
plt.savefig("data/clean/sentimiento_twitter.png")
plt.close()

# Reddit
print("\nReddit:")
print(df_reddit['sentiment_label'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=df_reddit,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'red','neutro':'yellow','positivo':'green'})
plt.title("Sentimiento en Reddit")
plt.savefig("data/clean/sentimiento_reddit.png")
plt.close()

# 6. UNIFICACIÓN DE DATASETS

print("\n--- Unificando datasets ---")

df_tweets_final = df_tweets_sample[['clean_text','sentiment','sentiment_label']].copy()
df_tweets_final['source'] = 'twitter'

df_reddit_final = df_reddit[['clean_text','sentiment','sentiment_label']].copy()
df_reddit_final['source'] = 'reddit'

df_unificado = pd.concat([df_tweets_final, df_reddit_final], ignore_index=True)

print(f"Dataset unificado con {len(df_unificado)} filas.")

# 7. ANÁLISIS SOBRE DATASET CONJUNTO


print("\n--- ANÁLISIS GLOBAL ---")

# Métricas
media_global = df_unificado['sentiment'].mean()
std_global = df_unificado['sentiment'].std()

print(f"Media global: {media_global:.4f}")
print(f"Polarización global: {std_global:.4f}")

# Distribución de sentimiento
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=df_unificado,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'red','neutro':'yellow','positivo':'green'})
plt.title("Distribución de Sentimiento (Global)")

plt.savefig("data/clean/sentimiento_global.png")
plt.close()

# Comparativa por fuente
plt.figure(figsize=(8,5))
sns.countplot(x='source', hue='sentiment_label', data=df_unificado,
              palette={'negativo':'red','neutro':'yellow','positivo':'green'})
plt.title("Comparativa Twitter vs Reddit")

plt.savefig("data/clean/comparativa_twitter_reddit.png")
plt.close()

# Distribución de polaridad
plt.figure(figsize=(8,5))
sns.histplot(df_unificado['sentiment'], bins=30, kde=True)
plt.title("Distribución de Polaridad Global")
plt.savefig("data/clean/distribucion_polaridad_global.png")
plt.close()

# 8. VISUALIZACIÓN COMPARATIVA ORIGINAL

fig, ax = plt.subplots(figsize=(10, 6))

df_total = pd.concat([
    pd.DataFrame({'Score': df_tweets_sample['sentiment'], 'Fuente': 'Twitter'}),
    pd.DataFrame({'Score': df_reddit['sentiment'], 'Fuente': 'Reddit'})
])

sns.boxplot(data=df_total, x='Fuente', y='Score', palette="Set2")
plt.title("Comparativa de Sentimiento: Twitter vs Reddit")
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.savefig("data/clean/comparativa_sentimiento.png")
plt.close()


# 9. EXPORTACIÓN

df_tweets_sample.to_csv("data/clean/data_processed_tweets.csv", index=False)
df_reddit.to_csv("data/clean/data_processed_reddit.csv", index=False)
df_unificado.to_csv("data/clean/data_unificado.csv", index=False)

print("¡Archivos listos para Dashboard y Gephi!")