"""
MAIN.PY - Pipeline Principal de Análisis de Sentimiento sobre IA
================================================================

Este script realiza:
1. Carga y unificación de datos de Twitter y Reddit
2. Limpieza y preprocesamiento de texto
3. Análisis de sentimiento con VADER
4. Detección de aspectos (ABSA)
5. Topic Modelling con BERTopic
6. Exportación para Gephi y Dashboard

Autor: Helena Molina
Fecha: Mayo 2026
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Librerías de procesamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Análisis de sentimiento
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Topic Modelling (opcional - requiere C++ build tools)
try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("[WARNING] BERTopic no disponible. Topic modelling sera omitido.")

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

# Obtener ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR = BASE_DIR / "data" / "clean"
IMAGES_DIR = BASE_DIR / "data" / "imagenes"

# Crear directorios si no existen
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Directorio base: {BASE_DIR}")
print(f"📊 Datos raw: {RAW_DATA_DIR}")
print(f"✅ Datos clean: {CLEAN_DATA_DIR}")
print(f"🎨 Imágenes: {IMAGES_DIR}\n")

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

def cargar_datos():
    """
    Carga los datasets de Twitter y Reddit.
    
    Returns:
        tuple: (df_twitter, df_reddit)
    """
    print("=" * 70)
    print("📥 PASO 1: CARGANDO DATOS")
    print("=" * 70)
    
    # Cargar Twitter
    twitter_path = RAW_DATA_DIR / "Twitter_Final_data.csv"
    print(f"\n🐦 Cargando Twitter desde: {twitter_path}")
    df_twitter = pd.read_csv(twitter_path)
    print(f"   ✓ Twitter: {len(df_twitter):,} filas, {len(df_twitter.columns)} columnas")
    print(f"   Columnas: {list(df_twitter.columns)}")
    
    # Cargar Reddit
    reddit_path = RAW_DATA_DIR / "reddit_comments_combined.csv"
    print(f"\n🔴 Cargando Reddit desde: {reddit_path}")
    df_reddit = pd.read_csv(reddit_path)
    print(f"   ✓ Reddit: {len(df_reddit):,} filas, {len(df_reddit.columns)} columnas")
    print(f"   Columnas: {list(df_reddit.columns)}")
    
    return df_twitter, df_reddit


# ============================================================================
# 2. LIMPIEZA DE TEXTO
# ============================================================================

def limpiar_texto(texto):
    """
    Limpia un texto eliminando URLs, menciones, hashtags, emojis,
    caracteres especiales y convierte a minúsculas.
    
    Args:
        texto (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    if not isinstance(texto, str) or pd.isna(texto):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    # Eliminar menciones y hashtags
    texto = re.sub(r'@\w+|#\w+', '', texto)
    
    # Eliminar emojis (rangos Unicode)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    texto = emoji_pattern.sub(r'', texto)
    
    # Eliminar caracteres especiales, pero mantener espacios
    texto = re.sub(r'[^a-z\s]', '', texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


def eliminar_stopwords(texto, idioma='english'):
    """
    Elimina stopwords de un texto.
    
    Args:
        texto (str): Texto limpio
        idioma (str): Idioma de las stopwords
        
    Returns:
        str: Texto sin stopwords
    """
    if not texto:
        return ""
    
    stop_words = set(stopwords.words(idioma))
    
    # Añadir stopwords personalizadas (muy comunes pero poco informativas)
    stop_words_custom = {'ai', 'ia', 'like', 'just', 'get', 'one', 'would', 'could'}
    stop_words.update(stop_words_custom)
    
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if p not in stop_words and len(p) > 2]
    
    return ' '.join(palabras_filtradas)


def preprocesar_datasets(df_twitter, df_reddit):
    """
    Aplica limpieza a ambos datasets y los prepara para unificación.
    
    Args:
        df_twitter: DataFrame de Twitter
        df_reddit: DataFrame de Reddit
        
    Returns:
        tuple: (df_twitter_clean, df_reddit_clean)
    """
    print("\n" + "=" * 70)
    print("🧹 PASO 2: LIMPIEZA Y PREPROCESAMIENTO")
    print("=" * 70)
    
    # Twitter: identificar columna de texto
    # Asumiendo que la columna se llama 'content', 'text', 'tweet', etc.
    posibles_columnas_twitter = ['content', 'text', 'tweet', 'full_text']
    columna_twitter = None
    for col in posibles_columnas_twitter:
        if col in df_twitter.columns:
            columna_twitter = col
            break
    
    if columna_twitter is None:
        raise ValueError(f"No se encontró columna de texto en Twitter. Columnas: {df_twitter.columns}")
    
    print(f"\n🐦 Twitter: usando columna '{columna_twitter}'")
    
    # Reddit: identificar columna de texto
    posibles_columnas_reddit = ['Comment Body', 'body', 'text', 'comment']
    columna_reddit = None
    for col in posibles_columnas_reddit:
        if col in df_reddit.columns:
            columna_reddit = col
            break
    
    if columna_reddit is None:
        raise ValueError(f"No se encontró columna de texto en Reddit. Columnas: {df_reddit.columns}")
    
    print(f"🔴 Reddit: usando columna '{columna_reddit}'")
    
    # Limpieza Twitter
    print("\n   Limpiando textos de Twitter...")
    df_twitter['texto_original'] = df_twitter[columna_twitter]
    df_twitter['texto_limpio'] = df_twitter[columna_twitter].apply(limpiar_texto)
    df_twitter['texto_sin_stopwords'] = df_twitter['texto_limpio'].apply(eliminar_stopwords)
    
    # Limpieza Reddit
    print("   Limpiando textos de Reddit...")
    df_reddit['texto_original'] = df_reddit[columna_reddit]
    df_reddit['texto_limpio'] = df_reddit[columna_reddit].apply(limpiar_texto)
    df_reddit['texto_sin_stopwords'] = df_reddit['texto_limpio'].apply(eliminar_stopwords)
    
    # Eliminar textos vacíos
    df_twitter = df_twitter[df_twitter['texto_limpio'].str.len() > 10].copy()
    df_reddit = df_reddit[df_reddit['texto_limpio'].str.len() > 10].copy()
    
    # Eliminar duplicados
    df_twitter = df_twitter.drop_duplicates(subset=['texto_limpio']).reset_index(drop=True)
    df_reddit = df_reddit.drop_duplicates(subset=['texto_limpio']).reset_index(drop=True)
    
    print(f"\n✅ Twitter después de limpieza: {len(df_twitter):,} registros")
    print(f"✅ Reddit después de limpieza: {len(df_reddit):,} registros")
    
    return df_twitter, df_reddit


# ============================================================================
# 3. ANÁLISIS DE SENTIMIENTO CON VADER
# ============================================================================

def analizar_sentimiento_vader(df, columna_texto='texto_limpio'):
    """
    Aplica análisis de sentimiento con VADER.
    
    Args:
        df: DataFrame con textos
        columna_texto: Nombre de la columna con texto limpio
        
    Returns:
        DataFrame con columnas de sentimiento añadidas
    """
    print("\n" + "=" * 70)
    print("😊 PASO 3: ANÁLISIS DE SENTIMIENTO (VADER)")
    print("=" * 70)
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Calcular scores de sentimiento
    print("\n   Calculando scores de polaridad...")
    df['vader_neg'] = df[columna_texto].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df['vader_neu'] = df[columna_texto].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    df['vader_pos'] = df[columna_texto].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df['vader_compound'] = df[columna_texto].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Clasificar sentimiento
    def clasificar_sentimiento(compound):
        if compound >= 0.05:
            return 'positivo'
        elif compound <= -0.05:
            return 'negativo'
        else:
            return 'neutro'
    
    df['sentimiento'] = df['vader_compound'].apply(clasificar_sentimiento)
    
    # Estadísticas
    print("\n📊 Distribución de sentimiento:")
    print(df['sentimiento'].value_counts())
    print(f"\n📈 Polaridad media: {df['vader_compound'].mean():.3f}")
    print(f"📏 Desviación estándar: {df['vader_compound'].std():.3f}")
    
    return df


# ============================================================================
# 4. ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ============================================================================

def detectar_aspectos(df, columna_texto='texto_limpio'):
    """
    Detecta aspectos relacionados con IA en los textos.
    
    Aspectos considerados:
    - Empleo y trabajo
    - Automatización
    - Productividad
    - Creatividad
    - Educación
    - Salarios
    - Ética
    - Reemplazo laboral
    - Oportunidades
    - Miedo/preocupación
    
    Args:
        df: DataFrame
        columna_texto: Columna con texto limpio
        
    Returns:
        DataFrame con columnas de aspectos
    """
    print("\n" + "=" * 70)
    print("🔍 PASO 4: DETECCIÓN DE ASPECTOS (ABSA)")
    print("=" * 70)
    
    # Definir keywords para cada aspecto
    aspectos = {
        'empleo': ['job', 'work', 'employment', 'career', 'workplace', 'worker', 'trabajador', 'empleo'],
        'automatizacion': ['automat', 'robot', 'machine', 'algorithm'],
        'productividad': ['productiv', 'efficien', 'performance', 'output'],
        'creatividad': ['creativ', 'art', 'design', 'innovation', 'innovat'],
        'educacion': ['educat', 'learn', 'teach', 'school', 'student', 'training'],
        'salarios': ['salary', 'wage', 'income', 'pay', 'salario', 'sueldo'],
        'etica': ['ethic', 'moral', 'bias', 'fair', 'responsible', 'transparency'],
        'reemplazo': ['replac', 'displace', 'eliminate', 'threat', 'lose', 'lost'],
        'oportunidades': ['opportunit', 'benefit', 'advantage', 'potential', 'future'],
        'miedo': ['fear', 'worry', 'concern', 'anxiety', 'scary', 'danger', 'risk']
    }
    
    print("\n   Detectando aspectos en textos...")
    
    # Detectar presencia de cada aspecto
    for aspecto, keywords in aspectos.items():
        pattern = '|'.join(keywords)
        df[f'aspecto_{aspecto}'] = df[columna_texto].str.contains(pattern, case=False, na=False)
    
    # Contar aspectos por texto
    columnas_aspectos = [f'aspecto_{asp}' for asp in aspectos.keys()]
    df['num_aspectos'] = df[columnas_aspectos].sum(axis=1)
    
    # Estadísticas
    print("\n📊 Frecuencia de aspectos detectados:")
    for aspecto in aspectos.keys():
        count = df[f'aspecto_{aspecto}'].sum()
        pct = (count / len(df)) * 100
        print(f"   • {aspecto.capitalize():20s}: {count:6,} ({pct:5.2f}%)")
    
    print(f"\n   Textos con al menos un aspecto: {(df['num_aspectos'] > 0).sum():,}")
    
    return df


# ============================================================================
# 5. TOPIC MODELLING CON BERTOPIC
# ============================================================================

def aplicar_topic_modelling(df, columna_texto='texto_sin_stopwords', n_topics=10):
    """
    Aplica BERTopic para identificar temas principales.
    
    Args:
        df: DataFrame
        columna_texto: Columna con texto procesado
        n_topics: Número aproximado de tópicos a extraer
        
    Returns:
        tuple: (df con columnas de tópicos, modelo BERTopic o None)
    """
    if not BERTOPIC_AVAILABLE:
        print("\n⚠️  BERTopic no disponible. Omitiendo topic modelling.")
        df['topic'] = -1  # Asignar tópico por defecto
        return df, None
    
    print("\n" + "=" * 70)
    print("📚 PASO 5: TOPIC MODELLING (BERTOPIC)")
    print("=" * 70)
    
    # Preparar textos
    textos = df[columna_texto].tolist()
    
    print(f"\n   Analizando {len(textos):,} textos...")
    print("   Esto puede tardar varios minutos...")
    
    # Configurar BERTopic
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5
    )
    
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics,
        language='english',
        calculate_probabilities=False,
        verbose=False
    )
    
    # Entrenar modelo
    topics, probabilities = topic_model.fit_transform(textos)
    
    # Añadir tópicos al DataFrame
    df['topic'] = topics
    
    # Obtener información de tópicos
    topic_info = topic_model.get_topic_info()
    
    print("\n✅ Tópicos identificados:")
    print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))
    
    # Guardar visualizaciones
    print("\n   Guardando visualizaciones de tópicos...")
    
    try:
        # Visualización de tópicos
        fig = topic_model.visualize_topics()
        fig.write_html(str(IMAGES_DIR / "topics_visualization.html"))
        print("   ✓ topics_visualization.html")
    except Exception as e:
        print(f"   ⚠ No se pudo generar visualización de tópicos: {e}")
    
    try:
        # Barchart de tópicos
        fig = topic_model.visualize_barchart(top_n_topics=10)
        fig.write_html(str(IMAGES_DIR / "topics_barchart.html"))
        print("   ✓ topics_barchart.html")
    except Exception as e:
        print(f"   ⚠ No se pudo generar barchart: {e}")
    
    return df, topic_model


# ============================================================================
# 6. UNIFICACIÓN DE DATASETS
# ============================================================================

def unificar_datasets(df_twitter, df_reddit):
    """
    Unifica los datasets de Twitter y Reddit en uno solo.
    
    Args:
        df_twitter: DataFrame de Twitter procesado
        df_reddit: DataFrame de Reddit procesado
        
    Returns:
        DataFrame unificado
    """
    print("\n" + "=" * 70)
    print("🔗 PASO 6: UNIFICACIÓN DE DATASETS")
    print("=" * 70)
    
    # Seleccionar columnas relevantes
    columnas_comunes = [
        'texto_original', 'texto_limpio', 'texto_sin_stopwords',
        'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
        'sentimiento', 'num_aspectos', 'topic'
    ]
    
    # Añadir columnas de aspectos
    columnas_aspectos = [col for col in df_twitter.columns if col.startswith('aspecto_')]
    columnas_comunes.extend(columnas_aspectos)
    
    # Crear copias con identificador de fuente
    df_twitter_final = df_twitter[columnas_comunes].copy()
    df_twitter_final['fuente'] = 'twitter'
    
    df_reddit_final = df_reddit[columnas_comunes].copy()
    df_reddit_final['fuente'] = 'reddit'
    
    # Unificar
    df_unificado = pd.concat([df_twitter_final, df_reddit_final], ignore_index=True)
    
    print(f"\n✅ Dataset unificado:")
    print(f"   • Twitter: {len(df_twitter_final):,} registros")
    print(f"   • Reddit:  {len(df_reddit_final):,} registros")
    print(f"   • TOTAL:   {len(df_unificado):,} registros")
    
    return df_unificado


# ============================================================================
# 7. EXPORTACIÓN PARA GEPHI
# ============================================================================

def exportar_para_gephi(df, columna_texto='texto_sin_stopwords'):
    """
    Crea un archivo de red semántica para Gephi.
    
    Genera un grafo de co-ocurrencias de palabras donde:
    - Nodos = palabras clave
    - Aristas = co-ocurrencia en el mismo texto
    - Peso = frecuencia de co-ocurrencia
    - Atributo = sentimiento predominante
    
    Args:
        df: DataFrame unificado
        columna_texto: Columna con texto procesado
        
    Returns:
        None (guarda archivo CSV)
    """
    print("\n" + "=" * 70)
    print("🕸️ PASO 7: PREPARACIÓN PARA GEPHI")
    print("=" * 70)
    
    from collections import Counter
    import itertools
    
    print("\n   Generando red de co-ocurrencias...")
    
    # Palabras a excluir (demasiado genéricas)
    stop_words_extra = {'ai', 'ia', 'artificial', 'intelligence', 'will', 'can', 'may'}
    
    conexiones = []
    
    for idx, row in df.iterrows():
        texto = str(row[columna_texto])
        sentimiento = row['sentimiento']
        
        # Extraer palabras significativas (longitud > 3)
        palabras = [
            w for w in texto.split()
            if len(w) > 3 and w not in stop_words_extra
        ]
        
        # Eliminar duplicados en el mismo texto
        palabras_unicas = sorted(set(palabras))
        
        # Crear parejas de palabras (co-ocurrencias)
        for pareja in itertools.combinations(palabras_unicas, 2):
            conexiones.append((pareja[0], pareja[1], sentimiento))
    
    # Contar frecuencias
    contador = Counter(conexiones)
    
    # Crear DataFrame para Gephi
    datos_gephi = []
    for (palabra1, palabra2, sentimiento), peso in contador.items():
        if peso > 2:  # Solo conexiones con más de 2 ocurrencias
            datos_gephi.append({
                'Source': palabra1,
                'Target': palabra2,
                'Weight': peso,
                'Sentiment': sentimiento
            })
    
    df_gephi = pd.DataFrame(datos_gephi)
    
    # Guardar
    output_path = CLEAN_DATA_DIR / "gephi_global_con_sentimiento.csv"
    df_gephi.to_csv(output_path, index=False)
    
    print(f"\n✅ Archivo Gephi guardado: {output_path}")
    print(f"   • {len(df_gephi):,} conexiones (aristas)")
    print(f"   • {len(set(df_gephi['Source']) | set(df_gephi['Target'])):,} palabras únicas (nodos)")
    
    # Mostrar top conexiones
    top_conexiones = df_gephi.nlargest(10, 'Weight')[['Source', 'Target', 'Weight', 'Sentiment']]
    print("\n📊 Top 10 conexiones más frecuentes:")
    print(top_conexiones.to_string(index=False))


# ============================================================================
# 8. VISUALIZACIONES PRINCIPALES
# ============================================================================

def generar_visualizaciones(df):
    """
    Genera gráficos principales para el análisis.
    
    Args:
        df: DataFrame unificado
    """
    print("\n" + "=" * 70)
    print("📊 PASO 8: GENERANDO VISUALIZACIONES")
    print("=" * 70)
    
    sns.set_theme(style="whitegrid", palette="husl")
    
    # 1. Distribución de sentimiento global
    print("\n   Generando gráfico de sentimiento global...")
    plt.figure(figsize=(10, 6))
    order = ['negativo', 'neutro', 'positivo']
    colors = {'negativo': '#e74c3c', 'neutro': '#f39c12', 'positivo': '#27ae60'}
    
    sns.countplot(data=df, x='sentimiento', order=order, palette=colors)
    plt.title('Distribución de Sentimiento sobre IA', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimiento', fontsize=12)
    plt.ylabel('Número de publicaciones', fontsize=12)
    plt.savefig(IMAGES_DIR / 'sentimiento_global.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ sentimiento_global.png")
    
    # 2. Comparativa Twitter vs Reddit
    print("   Generando comparativa por fuente...")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='fuente', hue='sentimiento', hue_order=order, palette=colors)
    plt.title('Sentimiento por Fuente: Twitter vs Reddit', fontsize=16, fontweight='bold')
    plt.xlabel('Fuente', fontsize=12)
    plt.ylabel('Número de publicaciones', fontsize=12)
    plt.legend(title='Sentimiento')
    plt.savefig(IMAGES_DIR / 'comparativa_fuentes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ comparativa_fuentes.png")
    
    # 3. Distribución de polaridad
    print("   Generando distribución de polaridad...")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='vader_compound', bins=50, kde=True, color='#3498db')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutro (0)')
    plt.title('Distribución de Polaridad (VADER Compound)', fontsize=16, fontweight='bold')
    plt.xlabel('Score de Polaridad', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.savefig(IMAGES_DIR / 'distribucion_polaridad.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ distribucion_polaridad.png")
    
    # 4. Boxplot comparativo
    print("   Generando boxplot comparativo...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fuente', y='vader_compound', palette='Set2')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Distribución de Polaridad por Fuente', fontsize=16, fontweight='bold')
    plt.xlabel('Fuente', fontsize=12)
    plt.ylabel('Score VADER Compound', fontsize=12)
    plt.savefig(IMAGES_DIR / 'boxplot_fuentes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ boxplot_fuentes.png")
    
    # 5. Aspectos más mencionados
    print("   Generando gráfico de aspectos...")
    columnas_aspectos = [col for col in df.columns if col.startswith('aspecto_')]
    aspectos_count = df[columnas_aspectos].sum().sort_values(ascending=False)
    aspectos_count.index = [col.replace('aspecto_', '').capitalize() for col in aspectos_count.index]
    
    plt.figure(figsize=(12, 6))
    aspectos_count.plot(kind='barh', color='#9b59b6')
    plt.title('Aspectos más Mencionados sobre IA', fontsize=16, fontweight='bold')
    plt.xlabel('Número de menciones', fontsize=12)
    plt.ylabel('Aspecto', fontsize=12)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'aspectos_frecuencia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ aspectos_frecuencia.png")
    
    # 6. Sentimiento por aspecto
    print("   Generando sentimiento por aspecto...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(columnas_aspectos):
        aspecto_nombre = col.replace('aspecto_', '').capitalize()
        df_aspecto = df[df[col] == True]
        
        if len(df_aspecto) > 0:
            sentimiento_counts = df_aspecto['sentimiento'].value_counts()
            axes[idx].pie(
                sentimiento_counts,
                labels=sentimiento_counts.index,
                autopct='%1.1f%%',
                colors=[colors.get(s, '#95a5a6') for s in sentimiento_counts.index]
            )
            axes[idx].set_title(f'{aspecto_nombre}\n(n={len(df_aspecto)})', fontsize=10)
    
    plt.suptitle('Distribución de Sentimiento por Aspecto', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'sentimiento_por_aspecto.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ sentimiento_por_aspecto.png")
    
    print("\n Todas las visualizaciones generadas correctamente")


# ============================================================================
# 9. EXPORTACIÓN DE DATOS FINALES
# ============================================================================

def exportar_datos(df_unificado, df_twitter, df_reddit):
    """
    Exporta los datasets procesados.
    
    Args:
        df_unificado: Dataset completo
        df_twitter: Dataset de Twitter
        df_reddit: Dataset de Reddit
    """
    print("\n" + "=" * 70)
    print(" PASO 9: EXPORTANDO DATOS PROCESADOS")
    print("=" * 70)
    
    # Exportar dataset unificado
    path_unificado = CLEAN_DATA_DIR / "datos_unificados.csv"
    df_unificado.to_csv(path_unificado, index=False)
    print(f"\n✓ {path_unificado}")
    
    # Exportar Twitter
    path_twitter = CLEAN_DATA_DIR / "twitter_procesado.csv"
    df_twitter.to_csv(path_twitter, index=False)
    print(f"✓ {path_twitter}")
    
    # Exportar Reddit
    path_reddit = CLEAN_DATA_DIR / "reddit_procesado.csv"
    df_reddit.to_csv(path_reddit, index=False)
    print(f"✓ {path_reddit}")
    
    print("\n Exportación completada")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """
    Pipeline principal que ejecuta todo el análisis.
    """
    print("\n" + "=" * 70)
    print(" INICIANDO ANÁLISIS DE SENTIMIENTO SOBRE IA")
    print("=" * 70)
    
    try:
        # 1. Cargar datos
        df_twitter, df_reddit = cargar_datos()
        
        # 2. Limpiar textos
        df_twitter, df_reddit = preprocesar_datasets(df_twitter, df_reddit)
        
        # 3. Análisis de sentimiento
        df_twitter = analizar_sentimiento_vader(df_twitter)
        df_reddit = analizar_sentimiento_vader(df_reddit)
        
        # 4. Detección de aspectos
        df_twitter = detectar_aspectos(df_twitter)
        df_reddit = detectar_aspectos(df_reddit)
        
        # 5. Topic modelling
        if BERTOPIC_AVAILABLE:
            df_twitter, twitter_topic_model = aplicar_topic_modelling(df_twitter, n_topics=8)
            df_reddit, reddit_topic_model = aplicar_topic_modelling(df_reddit, n_topics=8)
        else:
            print("\n⚠️  Topic modelling omitido (BERTopic no disponible)")
            df_twitter['topic'] = -1
            df_reddit['topic'] = -1
            twitter_topic_model = None
            reddit_topic_model = None
        
        # 6. Unificar datasets
        df_unificado = unificar_datasets(df_twitter, df_reddit)
        
        # 7. Exportar para Gephi
        exportar_para_gephi(df_unificado)
        
        # 8. Generar visualizaciones
        generar_visualizaciones(df_unificado)
        
        # 9. Exportar datos finales
        exportar_datos(df_unificado, df_twitter, df_reddit)
        
        print("\n" + "=" * 70)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"\n📁 Los archivos están listos en:")
        print(f"   • Datos limpios: {CLEAN_DATA_DIR}")
        print(f"   • Visualizaciones: {IMAGES_DIR}")
        print("\n🎯 Próximos pasos:")
        print("   1. Importar 'gephi_global_con_sentimiento.csv' en Gephi")
        print("   2. Ejecutar 'streamlit run notebooks/dashboard.py'")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()