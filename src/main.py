# ==============================================================================
# ANÁLISIS DE SENTIMIENTO Y OPINIÓN PÚBLICA SOBRE IA Y EMPLEO
# Trabajo Individual - Modelos Predictivos III
# ==============================================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import itertools
from collections import Counter
import networkx as nx

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_raw = os.path.join(base_path, "data", "raw")
data_clean = os.path.join(base_path, "data", "clean")

# Crear carpetas si no existen
os.makedirs(data_clean, exist_ok=True)

print("="*80)
print("ANÁLISIS DE SENTIMIENTO: IA Y EMPLEO")
print("="*80)

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
print("\n[1/7] Cargando datasets...")

tweets_path = os.path.join(data_raw, "Twitter_Final_data.csv")
reddit_path = os.path.join(data_raw, "reddit_comments_combined.csv")

df_tweets = pd.read_csv(tweets_path)
df_reddit = pd.read_csv(reddit_path)

print(f"   ✓ Tweets cargados: {len(df_tweets)} filas")
print(f"   ✓ Reddit cargados: {len(df_reddit)} filas")

# Muestreo de 5000 tweets para balancear con Reddit
df_tweets_sample = df_tweets.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"   ✓ Muestra de Twitter: {len(df_tweets_sample)} filas")

# ==============================================================================
# 2. LIMPIEZA DE TEXTO
# ==============================================================================
print("\n[2/7] Limpiando textos...")

def limpiar_texto(texto):
    """Limpieza exhaustiva de texto para análisis NLP"""
    if not isinstance(texto, str): 
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)  # URLs
    texto = re.sub(r'\@\w+|\#','', texto)  # Menciones/Hashtags
    texto = re.sub(r'[^a-z\s]', '', texto)  # Solo letras
    texto = re.sub(r'\s+', ' ', texto)  # Espacios múltiples
    return texto.strip()

df_tweets_sample['clean_text'] = df_tweets_sample['content'].apply(limpiar_texto)
df_reddit['clean_text'] = df_reddit['Comment Body'].apply(limpiar_texto)

print("   ✓ Textos limpios generados")

# ==============================================================================
# 3. ANÁLISIS DE SENTIMIENTO CON VADER
# ==============================================================================
print("\n[3/7] Calculando sentimiento con VADER...")

analyzer = SentimentIntensityAnalyzer()

def obtener_score(texto):
    """Calcula el score de sentimiento compuesto de VADER"""
    return analyzer.polarity_scores(texto)['compound']

def clasificar_sentimiento(valor):
    """Clasifica el sentimiento en categorías"""
    if valor > 0.05:
        return "positivo"
    elif valor < -0.05:
        return "negativo"
    else:
        return "neutro"

df_tweets_sample['sentiment'] = df_tweets_sample['clean_text'].apply(obtener_score)
df_reddit['sentiment'] = df_reddit['clean_text'].apply(obtener_score)

df_tweets_sample['sentiment_label'] = df_tweets_sample['sentiment'].apply(clasificar_sentimiento)
df_reddit['sentiment_label'] = df_reddit['sentiment'].apply(clasificar_sentimiento)

print("   ✓ Sentimiento calculado para ambas fuentes")

# ==============================================================================
# 4. ABSA - ASPECT-BASED SENTIMENT ANALYSIS
# ==============================================================================
print("\n[4/7] Ejecutando ABSA (Aspect-Based Sentiment Analysis)...")

# Definición de aspectos clave sobre IA y empleo
ASPECTOS_IA = {
    'Empleo': ['job', 'jobs', 'work', 'employment', 'unemployed', 'unemployment', 'career', 'workforce', 'worker', 'workers', 'hire', 'hiring'],
    'Automatizacion': ['automation', 'automate', 'automated', 'robot', 'robots', 'robotics', 'machine', 'machines', 'replace', 'replacement'],
    'Futuro': ['future', 'tomorrow', 'coming', 'next', 'ahead', 'prospect', 'prediction', 'forecast', 'trend', 'evolution'],
    'Productividad': ['productivity', 'productive', 'efficient', 'efficiency', 'improve', 'improvement', 'optimize', 'optimization', 'performance'],
    'Etica': ['ethics', 'ethical', 'moral', 'fairness', 'fair', 'bias', 'biased', 'discrimination', 'privacy', 'rights', 'responsible'],
    'Economia': ['economy', 'economic', 'money', 'cost', 'profit', 'salary', 'wage', 'income', 'wealth', 'financial', 'growth']
}

def detectar_aspectos(texto):
    """Detecta qué aspectos se mencionan en un texto"""
    aspectos_detectados = []
    texto_lower = texto.lower()
    
    for aspecto, palabras_clave in ASPECTOS_IA.items():
        if any(palabra in texto_lower for palabra in palabras_clave):
            aspectos_detectados.append(aspecto)
    
    return aspectos_detectados if aspectos_detectados else ['General']

def calcular_sentimiento_por_aspecto(df_input):
    """Calcula el sentimiento promedio por aspecto"""
    resultados = []
    
    for index, row in df_input.iterrows():
        aspectos = detectar_aspectos(row['clean_text'])
        for aspecto in aspectos:
            resultados.append({
                'aspecto': aspecto,
                'sentiment': row['sentiment'],
                'sentiment_label': row['sentiment_label'],
                'source': row.get('source', 'unknown')
            })
    
    return pd.DataFrame(resultados)

# Añadir columna de fuente
df_tweets_sample['source'] = 'twitter'
df_reddit['source'] = 'reddit'

# Calcular ABSA para ambas fuentes
df_absa_twitter = calcular_sentimiento_por_aspecto(df_tweets_sample)
df_absa_reddit = calcular_sentimiento_por_aspecto(df_reddit)
df_absa_completo = pd.concat([df_absa_twitter, df_absa_reddit], ignore_index=True)

print(f"   ✓ Análisis de aspectos completado")
print(f"   ✓ Aspectos detectados: {df_absa_completo['aspecto'].nunique()}")

# Guardar análisis ABSA
df_absa_completo.to_csv(os.path.join(data_clean, "absa_resultados.csv"), index=False)

# ==============================================================================
# 5. ANÁLISIS INDIVIDUAL
# ==============================================================================

print("\n--- ANÁLISIS INDIVIDUAL ---")

# Twitter
print("\nTwitter:")
print(df_tweets_sample['sentiment_label'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x='sentiment_label', data=df_tweets_sample,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'#E74C3C','neutro':'#F39C12','positivo':'#27AE60'},
              hue='sentiment_label', legend=False)  # ← Corregido warning
plt.title("Sentimiento en Twitter", fontsize=14, fontweight='bold')
plt.xlabel("Sentimiento", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "sentimiento_twitter.png"), dpi=300, bbox_inches='tight')
plt.close()

# Reddit
print("\nReddit:")
print(df_reddit['sentiment_label'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x='sentiment_label', data=df_reddit,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'#E74C3C','neutro':'#F39C12','positivo':'#27AE60'},
              hue='sentiment_label', legend=False)  # ← Corregido warning
plt.title("Sentimiento en Reddit", fontsize=14, fontweight='bold')
plt.xlabel("Sentimiento", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "sentimiento_reddit.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. UNIFICACIÓN DE DATASETS
# ==============================================================================

print("\n--- Unificando datasets ---")

df_tweets_final = df_tweets_sample[['clean_text','sentiment','sentiment_label']].copy()
df_tweets_final['source'] = 'twitter'

df_reddit_final = df_reddit[['clean_text','sentiment','sentiment_label']].copy()
df_reddit_final['source'] = 'reddit'

df_unificado = pd.concat([df_tweets_final, df_reddit_final], ignore_index=True)

print(f"Dataset unificado con {len(df_unificado)} filas.")

# ==============================================================================
# 7. ABSA - ASPECT-BASED SENTIMENT ANALYSIS
# ==============================================================================
print("\n[4/8] Ejecutando ABSA (Aspect-Based Sentiment Analysis)...")

# Definición de aspectos clave sobre IA y empleo
ASPECTOS_IA = {
    'Empleo': ['job', 'jobs', 'work', 'employment', 'unemployed', 'unemployment', 'career', 'workforce', 'worker', 'workers', 'hire', 'hiring'],
    'Automatizacion': ['automation', 'automate', 'automated', 'robot', 'robots', 'robotics', 'machine', 'machines', 'replace', 'replacement'],
    'Futuro': ['future', 'tomorrow', 'coming', 'next', 'ahead', 'prospect', 'prediction', 'forecast', 'trend', 'evolution'],
    'Productividad': ['productivity', 'productive', 'efficient', 'efficiency', 'improve', 'improvement', 'optimize', 'optimization', 'performance'],
    'Etica': ['ethics', 'ethical', 'moral', 'fairness', 'fair', 'bias', 'biased', 'discrimination', 'privacy', 'rights', 'responsible'],
    'Economia': ['economy', 'economic', 'money', 'cost', 'profit', 'salary', 'wage', 'income', 'wealth', 'financial', 'growth']
}

def detectar_aspectos(texto):
    """Detecta qué aspectos se mencionan en un texto"""
    aspectos_detectados = []
    texto_lower = texto.lower()
    
    for aspecto, palabras_clave in ASPECTOS_IA.items():
        if any(palabra in texto_lower for palabra in palabras_clave):
            aspectos_detectados.append(aspecto)
    
    return aspectos_detectados if aspectos_detectados else ['General']

def calcular_sentimiento_por_aspecto(df_input):
    """Calcula el sentimiento promedio por aspecto"""
    resultados = []
    
    for index, row in df_input.iterrows():
        aspectos = detectar_aspectos(row['clean_text'])
        for aspecto in aspectos:
            resultados.append({
                'aspecto': aspecto,
                'sentiment': row['sentiment'],
                'sentiment_label': row['sentiment_label'],
                'source': row.get('source', 'unknown')
            })
    
    return pd.DataFrame(resultados)

# Calcular ABSA para dataset unificado
df_absa_completo = calcular_sentimiento_por_aspecto(df_unificado)

print(f"   ✓ Análisis de aspectos completado")
print(f"   ✓ Aspectos detectados: {df_absa_completo['aspecto'].nunique()}")

# Guardar análisis ABSA
df_absa_completo.to_csv(os.path.join(data_clean, "absa_resultados.csv"), index=False)

# ==============================================================================
# 8. ANÁLISIS GLOBAL Y VISUALIZACIONES
# ==============================================================================

print("\n--- ANÁLISIS GLOBAL ---")

# Métricas
media_global = df_unificado['sentiment'].mean()
std_global = df_unificado['sentiment'].std()

print(f"Media global: {media_global:.4f}")
print(f"Polarización global: {std_global:.4f}")

# 8.1 Distribución de sentimiento
plt.figure(figsize=(8,5))
sns.countplot(x='sentiment_label', data=df_unificado,
              order=['negativo','neutro','positivo'],
              palette={'negativo':'#E74C3C','neutro':'#F39C12','positivo':'#27AE60'},
              hue='sentiment_label', legend=False)
plt.title("Distribución de Sentimiento (Global)", fontsize=14, fontweight='bold')
plt.xlabel("Sentimiento", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "sentimiento_global.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8.2 Comparativa por fuente
plt.figure(figsize=(10,6))
sns.countplot(x='source', hue='sentiment_label', data=df_unificado,
              palette={'negativo':'#E74C3C','neutro':'#F39C12','positivo':'#27AE60'})
plt.title("Comparativa Twitter vs Reddit", fontsize=14, fontweight='bold')
plt.xlabel("Plataforma", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.legend(title='Sentimiento', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "comparativa_twitter_reddit.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8.3 Distribución de polaridad
plt.figure(figsize=(10,6))
sns.histplot(df_unificado['sentiment'], bins=30, kde=True, color='#3498DB')
plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Neutro (0)')
plt.title("Distribución de Polaridad Global", fontsize=14, fontweight='bold')
plt.xlabel("Score de Sentimiento (VADER)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "distribucion_polaridad_global.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8.4 Visualización comparativa
fig, ax = plt.subplots(figsize=(10, 6))
df_total = pd.concat([
    pd.DataFrame({'Score': df_tweets_sample['sentiment'], 'Fuente': 'Twitter'}),
    pd.DataFrame({'Score': df_reddit['sentiment'], 'Fuente': 'Reddit'})
])
sns.boxplot(data=df_total, x='Fuente', y='Score', palette="Set2", hue='Fuente', legend=False)
plt.title("Comparativa de Sentimiento: Twitter vs Reddit", fontsize=14, fontweight='bold')
plt.xlabel("Plataforma", fontsize=12)
plt.ylabel("Score de Sentimiento", fontsize=12)
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "comparativa_sentimiento.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8.5 Análisis ABSA - Sentimiento por aspecto
plt.figure(figsize=(12, 6))
sentimiento_por_aspecto = df_absa_completo.groupby(['aspecto', 'sentiment_label']).size().unstack(fill_value=0)
sentimiento_por_aspecto.plot(kind='bar', stacked=False, 
                              color={'negativo':'#E74C3C','neutro':'#F39C12','positivo':'#27AE60'},
                              figsize=(12,6))
plt.title("Análisis de Sentimiento por Aspecto (ABSA)", fontsize=14, fontweight='bold')
plt.xlabel("Aspecto", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentimiento', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "absa_por_aspecto.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8.6 Sentimiento promedio por aspecto
plt.figure(figsize=(10, 6))
sentimiento_medio_aspecto = df_absa_completo.groupby('aspecto')['sentiment'].mean().sort_values()
colores = ['#E74C3C' if x < -0.05 else '#27AE60' if x > 0.05 else '#F39C12' for x in sentimiento_medio_aspecto]
sentimiento_medio_aspecto.plot(kind='barh', color=colores, figsize=(10,6))
plt.title("Sentimiento Promedio por Aspecto", fontsize=14, fontweight='bold')
plt.xlabel("Score de Sentimiento (VADER)", fontsize=12)
plt.ylabel("Aspecto", fontsize=12)
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(data_clean, "sentimiento_promedio_aspecto.png"), dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ Visualizaciones guardadas en data/clean/")

# ==============================================================================
# 9. GENERACIÓN DE RED SEMÁNTICA PARA GEPHI (ULTRA-FILTRADA)
# ==============================================================================
print("\n[7/8] Generando red semántica para Gephi...")

# Stopwords expandidas
stop_words = {
    'que', 'este', 'esta', 'con', 'para', 'una', 'los', 'las', 'del', 'por', 
    'ia', 'ai', 'the', 'and', 'are', 'will', 'can', 'this', 'that', 'have',
    'from', 'not', 'but', 'they', 'their', 'been', 'more', 'would', 'could',
    'about', 'like', 'just', 'what', 'when', 'where', 'which', 'who', 'how',
    'also', 'much', 'many', 'some', 'than', 'them', 'into', 'only', 'other',
    'should', 'being', 'make', 'made', 'need', 'want', 'even', 'well', 'back',
    'going', 'think', 'know', 'people', 'time', 'good', 'said', 'work', 'get',
    'use', 'used', 'using', 'way', 'things', 'thing', 'really', 'dont', 'isnt',
    'still', 'different', 'take', 'help', 'every', 'something', 'someone'
}

# SOLO palabras clave específicas de IA y empleo
palabras_clave_ia = {
    # Empleo
    'jobs', 'employment', 'unemployed', 'unemployment', 'workforce', 'worker', 
    'workers', 'career', 'skill', 'skills', 'training', 'education',
    
    # Automatización
    'automation', 'automate', 'automated', 'robot', 'robots', 'robotics',
    'machine', 'machines', 'replace', 'replacement', 'displaced',
    
    # IA/Tecnología
    'artificial', 'intelligence', 'algorithm', 'algorithms', 'technology',
    'technologies', 'software', 'computer', 'computers', 'digital', 'data',
    'learning', 'neural', 'network', 'networks',
    
    # Impacto
    'future', 'impact', 'change', 'changing', 'transform', 'transformation',
    'innovation', 'productivity', 'efficiency', 'economic', 'economy',
    'industry', 'industries', 'business', 'businesses', 'company', 'companies',
    
    # Aspectos humanos
    'human', 'humans', 'people', 'society', 'social', 'ethical', 'ethics'
}

conexiones = []

for _, fila in df_unificado.iterrows():
    texto = str(fila['clean_text'])
    sentimiento = fila['sentiment_label']
    
    # SOLO palabras de la lista de palabras clave
    palabras = [
        w for w in texto.split() 
        if w in palabras_clave_ia
    ]
    
    # Crear parejas SOLO si hay al menos 2 palabras clave
    if len(palabras) >= 2:
        for pareja in itertools.combinations(sorted(set(palabras)), 2):
            conexiones.append((pareja[0], pareja[1], sentimiento))

# Contar frecuencias
contador = Counter(conexiones)

# FILTRO SUPER AGRESIVO: Weight > 25
df_gephi = pd.DataFrame([
    {
        'Source': k[0], 
        'Target': k[1], 
        'Sentiment_Type': k[2],
        'Weight': v
    } 
    for k, v in contador.items() if v > 25
])

# Filtrado adicional: eliminar nodos con muy pocas conexiones
from collections import Counter as Counter2
nodos_source = Counter2(df_gephi['Source'])
nodos_target = Counter2(df_gephi['Target'])
nodos_totales = {k: nodos_source.get(k, 0) + nodos_target.get(k, 0) for k in set(nodos_source) | set(nodos_target)}

# Mantener solo nodos con al menos 3 conexiones
nodos_validos = {k for k, v in nodos_totales.items() if v >= 3}

df_gephi = df_gephi[
    (df_gephi['Source'].isin(nodos_validos)) & 
    (df_gephi['Target'].isin(nodos_validos))
]

# Guardar archivo para Gephi
gephi_path = os.path.join(data_clean, "gephi_global_con_sentimiento.csv")
df_gephi.to_csv(gephi_path, index=False)

nodos_unicos = len(set(df_gephi['Source']).union(set(df_gephi['Target'])))

print(f"   ✓ Red semántica generada: {len(df_gephi)} aristas")
print(f"   ✓ Nodos únicos: {nodos_unicos}")
if nodos_unicos > 0:
    print(f"   ✓ Densidad aproximada: {len(df_gephi) / (nodos_unicos * (nodos_unicos - 1) / 2) * 100:.2f}%")

# Estadísticas de la red
print(f"\n   Estadísticas de la red:")
print(f"   - Aristas totales: {len(df_gephi)}")
if len(df_gephi) > 0:
    print(f"   - Distribución por sentimiento:")
    for sent in df_gephi['Sentiment_Type'].unique():
        count = len(df_gephi[df_gephi['Sentiment_Type'] == sent])
        print(f"     · {sent.capitalize()}: {count} ({count/len(df_gephi)*100:.1f}%)")

# Verificación de calidad
if nodos_unicos > 200:
    print(f"\n   ⚠️ ADVERTENCIA: {nodos_unicos} nodos pueden ser demasiados")
    print(f"   ✓ Sugerencia: Aumentar el filtro de Weight a >30 o >40")
elif nodos_unicos < 30:
    print(f"\n   ⚠️ ADVERTENCIA: {nodos_unicos} nodos puede ser muy poco")
    print(f"   ✓ Sugerencia: Reducir el filtro de Weight a >15 o >20")
else:
    print(f"\n   ✅ Tamaño de red óptimo para visualización")

# ==============================================================================
# 10. EXPORTACIÓN FINAL
# ==============================================================================
print("\n[8/8] Exportando datasets finales...")

df_tweets_sample.to_csv(os.path.join(data_clean, "data_processed_tweets.csv"), index=False)
df_reddit.to_csv(os.path.join(data_clean, "data_processed_reddit.csv"), index=False)
df_unificado.to_csv(os.path.join(data_clean, "data_unificado.csv"), index=False)

print("   ✓ Datasets procesados guardados")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
print("\n" + "="*80)
print("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)
print("\nArchivos generados:")
print(f"  1. data_unificado.csv - Dataset principal ({len(df_unificado)} filas)")
print(f"  2. gephi_global_con_sentimiento.csv - Red para Gephi ({len(df_gephi)} aristas, {nodos_unicos} nodos)")
print(f"  3. absa_resultados.csv - Análisis por aspectos ({len(df_absa_completo)} registros)")
print(f"  4. Visualizaciones (9 gráficos PNG)")
print("\nPróximos pasos:")
print("  → Ejecutar: streamlit run src/dashboard.py")
print("  → Importar gephi_global_con_sentimiento.csv en Gephi")
print("  → Revisar INSTRUCCIONES_GEPHI.md para visualización de red")
print("="*80)