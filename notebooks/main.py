# Importo las librerías necesarias
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
from textblob import TextBlob

# Configuración visual
plt.style.use('default')

# Cargo el dataset de Kaggle
df_kaggle = pd.read_csv("ai_data_jobs_tweets.csv")

# Visualizo las primeras filas
print(df_kaggle.head())

# Veo estructura general
print(df_kaggle.info())

# Cargo los comentarios del hilo
df_twitter = pd.read_csv("bernie_thread_comments.csv")

# Reviso datos
print(df_twitter.head())


# Selecciono una muestra aleatoria de 2000 comentarios
df_twitter_sample = df_twitter.sample(n=2000, random_state=42)

print(len(df_twitter_sample))


# Defino una función de limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()  # paso a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # elimino URLs
    texto = re.sub(r"@\w+", "", texto)  # elimino menciones
    texto = re.sub(r"#\w+", "", texto)  # elimino hashtags
    texto = re.sub(r"[^a-z\s]", "", texto)  # elimino caracteres especiales
    texto = texto.strip()
    return texto

# Aplico la limpieza al dataset Kaggle
df_kaggle['clean_text'] = df_kaggle['text'].apply(limpiar_texto)

# Aplico la limpieza al dataset de Twitter
df_twitter_sample['clean_text'] = df_twitter_sample['text'].apply(limpiar_texto)


# Función para calcular sentimiento
def obtener_sentimiento(texto):
    return TextBlob(texto).sentiment.polarity

# Aplico al dataset Kaggle
df_kaggle['sentiment'] = df_kaggle['clean_text'].apply(obtener_sentimiento)

# Aplico al dataset Twitter
df_twitter_sample['sentiment'] = df_twitter_sample['clean_text'].apply(obtener_sentimiento)


# Clasifico en positivo, negativo o neutro
def clasificar_sentimiento(valor):
    if valor > 0:
        return "positivo"
    elif valor < 0:
        return "negativo"
    else:
        return "neutro"

df_kaggle['sentiment_label'] = df_kaggle['sentiment'].apply(clasificar_sentimiento)
df_twitter_sample['sentiment_label'] = df_twitter_sample['sentiment'].apply(clasificar_sentimiento)


sns.countplot(x='sentiment_label', data=df_kaggle)
plt.title("Sentimiento en dataset Kaggle")
plt.show()


sns.countplot(x='sentiment_label', data=df_twitter_sample)
plt.title("Sentimiento en hilo de Twitter")
plt.show()


# Comparo proporciones
kaggle_dist = df_kaggle['sentiment_label'].value_counts(normalize=True)
twitter_dist = df_twitter_sample['sentiment_label'].value_counts(normalize=True)

print("Kaggle:\n", kaggle_dist)
print("\nTwitter:\n", twitter_dist)


plt.hist(df_twitter_sample['sentiment'], bins=30)
plt.title("Distribución de polaridad - Twitter")
plt.show()



