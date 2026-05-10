# ==============================================================================
# DASHBOARD INTERACTIVO - ANÁLISIS DE SENTIMIENTO: IA Y EMPLEO
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Análisis IA y Empleo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# RUTAS Y CARGA DE DATOS
# ==============================================================================
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data", "clean")

@st.cache_data
def cargar_datos():
    """Carga todos los datasets necesarios"""
    df = pd.read_csv(os.path.join(data_path, "data_unificado.csv"))
    df_absa = pd.read_csv(os.path.join(data_path, "absa_resultados.csv"))
    return df, df_absa

df, df_absa = cargar_datos()

# ==============================================================================
# FUNCIÓN PARA CARGAR IMÁGENES
# ==============================================================================
def mostrar_imagen(nombre, caption=None):
    """Carga y muestra una imagen de forma segura"""
    ruta = os.path.join(data_path, nombre)
    if os.path.exists(ruta):
        st.image(ruta, caption=caption, use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen: {nombre}")

# ==============================================================================
# ESTILOS PERSONALIZADOS
# ==============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
st.markdown('<h1 class="main-title">Análisis de Sentimiento sobre IA y Empleo</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Comparativa Twitter vs Reddit + Análisis ABSA</p>', unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# SECCIÓN 1: MÉTRICAS CLAVE (KPIs)
# ==============================================================================
st.markdown('<h2 class="section-header">1. Métricas Clave del Estudio</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

total = len(df)
media = df["sentiment"].mean()
std = df["sentiment"].std()

# Calcular porcentajes por sentimiento
pct_positivo = (df['sentiment_label'] == 'positivo').sum() / total * 100
pct_negativo = (df['sentiment_label'] == 'negativo').sum() / total * 100
pct_neutro = (df['sentiment_label'] == 'neutro').sum() / total * 100

with col1:
    st.metric("Total de Comentarios", f"{total:,}", 
              help="Suma de tweets y posts de Reddit analizados")

with col2:
    st.metric("Sentimiento Medio", f"{media:.3f}", 
              delta=f"{'Positivo' if media > 0 else 'Negativo'}",
              help="Score promedio VADER (-1 a +1)")

with col3:
    st.metric("Polarización", f"{std:.3f}",
              help="Desviación estándar - Mayor valor = Mayor polarización")

with col4:
    st.metric("Sentimiento Dominante", 
              "Positivo" if pct_positivo > pct_negativo else "Negativo",
              delta=f"{max(pct_positivo, pct_negativo):.1f}%",
              help="Categoría con mayor frecuencia")

st.markdown("---")

# ==============================================================================
# SECCIÓN 2: ANÁLISIS POR PLATAFORMA
# ==============================================================================
st.markdown('<h2 class="section-header">2. Análisis por Plataforma</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Twitter", "Reddit", "Comparativa"])

with tab1:
    st.markdown("### Distribución de Sentimiento en Twitter")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mostrar_imagen("sentimiento_twitter.png")
    
    with col2:
        df_twitter = df[df['source'] == 'twitter']
        st.markdown("#### Estadísticas Twitter")
        st.write(f"**Total:** {len(df_twitter):,}")
        st.write(f"**Positivos:** {(df_twitter['sentiment_label']=='positivo').sum()} ({(df_twitter['sentiment_label']=='positivo').sum()/len(df_twitter)*100:.1f}%)")
        st.write(f"**Negativos:** {(df_twitter['sentiment_label']=='negativo').sum()} ({(df_twitter['sentiment_label']=='negativo').sum()/len(df_twitter)*100:.1f}%)")
        st.write(f"**Neutros:** {(df_twitter['sentiment_label']=='neutro').sum()} ({(df_twitter['sentiment_label']=='neutro').sum()/len(df_twitter)*100:.1f}%)")
        st.write(f"**Sentimiento medio:** {df_twitter['sentiment'].mean():.3f}")

with tab2:
    st.markdown("### Distribución de Sentimiento en Reddit")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mostrar_imagen("sentimiento_reddit.png")
    
    with col2:
        df_reddit = df[df['source'] == 'reddit']
        st.markdown("#### Estadísticas Reddit")
        st.write(f"**Total:** {len(df_reddit):,}")
        st.write(f"**Positivos:** {(df_reddit['sentiment_label']=='positivo').sum()} ({(df_reddit['sentiment_label']=='positivo').sum()/len(df_reddit)*100:.1f}%)")
        st.write(f"**Negativos:** {(df_reddit['sentiment_label']=='negativo').sum()} ({(df_reddit['sentiment_label']=='negativo').sum()/len(df_reddit)*100:.1f}%)")
        st.write(f"**Neutros:** {(df_reddit['sentiment_label']=='neutro').sum()} ({(df_reddit['sentiment_label']=='neutro').sum()/len(df_reddit)*100:.1f}%)")
        st.write(f"**Sentimiento medio:** {df_reddit['sentiment'].mean():.3f}")

with tab3:
    st.markdown("### Comparativa entre Plataformas")
    col1, col2 = st.columns(2)
    
    with col1:
        mostrar_imagen("comparativa_twitter_reddit.png", caption="Distribución de sentimientos por plataforma")
    
    with col2:
        mostrar_imagen("comparativa_sentimiento.png", caption="Boxplot comparativo de scores")

st.markdown("---")

# ==============================================================================
# SECCIÓN 3: ANÁLISIS GLOBAL
# ==============================================================================
st.markdown('<h2 class="section-header">3. Análisis Global (Dataset Unificado)</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Distribución Global de Sentimiento")
    mostrar_imagen("sentimiento_global.png")

with col2:
    st.markdown("### Distribución de Polaridad")
    mostrar_imagen("distribucion_polaridad_global.png")

# Interpretación
st.markdown("#### Interpretación de Resultados")
st.info(f"""
**Hallazgos principales:**
- El sentimiento medio global es de **{media:.3f}**, indicando una tendencia {'positiva' if media > 0 else 'negativa'}.
- La polarización (desviación estándar) es de **{std:.3f}**, mostrando {'alta' if std > 0.4 else 'moderada' if std > 0.3 else 'baja'} divergencia de opiniones.
- {'Twitter muestra más positividad que Reddit' if df[df['source']=='twitter']['sentiment'].mean() > df[df['source']=='reddit']['sentiment'].mean() else 'Reddit muestra más positividad que Twitter'}.
""")

st.markdown("---")

# ==============================================================================
# SECCIÓN 4: ANÁLISIS ABSA (ASPECT-BASED SENTIMENT ANALYSIS)
# ==============================================================================
st.markdown('<h2 class="section-header">4. Análisis por Aspectos (ABSA)</h2>', unsafe_allow_html=True)

st.markdown("""
El **Aspect-Based Sentiment Analysis (ABSA)** identifica aspectos específicos mencionados en los textos 
y analiza el sentimiento asociado a cada uno. Los aspectos detectados incluyen:
- **Empleo**: Referencias a trabajo, empleos, desempleo
- **Automatización**: Robots, automatización, reemplazo
- **Futuro**: Predicciones, tendencias, evolución
- **Productividad**: Eficiencia, mejoras, optimización
- **Ética**: Consideraciones morales, sesgos, privacidad
- **Economía**: Aspectos financieros, salarios, crecimiento
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Frecuencia de Sentimientos por Aspecto")
    mostrar_imagen("absa_por_aspecto.png")

with col2:
    st.markdown("### Sentimiento Promedio por Aspecto")
    mostrar_imagen("sentimiento_promedio_aspecto.png")

# Tabla de aspectos
st.markdown("### Tabla Detallada de Aspectos")
aspectos_stats = df_absa.groupby('aspecto').agg({
    'sentiment': ['count', 'mean', 'std']
}).round(3)
aspectos_stats.columns = ['Frecuencia', 'Sentimiento Medio', 'Desviación']
aspectos_stats = aspectos_stats.sort_values('Frecuencia', ascending=False)
st.dataframe(aspectos_stats, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECCIÓN 5: EXPLORADOR DE DATOS INTERACTIVO
# ==============================================================================
st.markdown('<h2 class="section-header">5. Explorador de Datos Interactivo</h2>', unsafe_allow_html=True)

# Filtros
col1, col2, col3 = st.columns(3)

with col1:
    filtro_fuente = st.multiselect(
        "Filtrar por plataforma:",
        options=df['source'].unique(),
        default=df['source'].unique()
    )

with col2:
    filtro_sentimiento = st.multiselect(
        "Filtrar por sentimiento:",
        options=df['sentiment_label'].unique(),
        default=df['sentiment_label'].unique()
    )

with col3:
    num_registros = st.slider(
        "Número de registros a mostrar:",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

# Aplicar filtros
df_filtrado = df[
    (df['source'].isin(filtro_fuente)) &
    (df['sentiment_label'].isin(filtro_sentimiento))
]

st.markdown(f"**Mostrando {min(num_registros, len(df_filtrado))} de {len(df_filtrado)} registros filtrados**")
st.dataframe(
    df_filtrado[['clean_text', 'sentiment', 'sentiment_label', 'source']].head(num_registros),
    use_container_width=True
)

# Botón de descarga
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar datos filtrados (CSV)",
    data=csv,
    file_name="datos_filtrados.csv",
    mime="text/csv"
)

st.markdown("---")

# ==============================================================================
# SECCIÓN 6: ANÁLISIS DE REDES SEMÁNTICAS (GEPHI)
# ==============================================================================
st.markdown('<h2 class="section-header">6. Análisis de Redes Semánticas</h2>', unsafe_allow_html=True)

st.markdown("""
### Visualización de Redes de Co-ocurrencia

Para complementar el análisis de sentimiento tradicional, se construyeron grafos semánticos 
mediante **Gephi**. 

En estas redes:
- Cada **nodo** representa una palabra o concepto relevante.
- Cada **arista** representa una co-ocurrencia entre palabras dentro de un mismo comentario.
- El tamaño de los nodos refleja la importancia o frecuencia del término.
- La estructura de la red permite detectar comunidades, conceptos centrales y relaciones semánticas.

Este enfoque permite pasar de un análisis basado únicamente en porcentajes y métricas 
a una representación estructural de cómo se relacionan las opiniones públicas sobre IA y empleo.
""")

st.markdown("---")

# ==============================================================================
# GRAFO GLOBAL
# ==============================================================================
st.markdown("## Grafo Global de Conversaciones")

col1, col2 = st.columns([2,1])

with col1:
    mostrar_imagen(
        "grafo_global.png",
        caption="Red global de relaciones semánticas sobre IA y empleo"
    )

with col2:
    st.markdown("""
### Interpretación del Grafo Global

El grafo global muestra los conceptos más importantes dentro de las conversaciones analizadas.

#### Hallazgos principales:
- **artificial intelligence**
- **machine learning**
- **data science**
- **big data**

son los nodos más centrales de la red.

Esto indica que las discusiones sobre IA y empleo giran principalmente en torno a:
- automatización,
- ciencia de datos,
- aprendizaje automático,
- y oportunidades laborales tecnológicas.

La alta densidad de conexiones refleja una conversación muy interrelacionada entre conceptos.
""")

st.markdown("---")

# ==============================================================================
# GRAFOS POR SENTIMIENTO
# ==============================================================================
st.markdown("## Redes Semánticas por Tipo de Sentimiento")

st.markdown("""
Para comprender mejor la estructura de las opiniones, la red fue dividida según el sentimiento 
detectado en los comentarios: positivo, neutro y negativo.
""")

# ---------------- POSITIVO ----------------
st.markdown("### Sentimiento Positivo")

col1, col2 = st.columns([1.5,1])

with col1:
    mostrar_imagen(
        "positivo.png",
        caption="Red semántica de comentarios positivos"
    )

with col2:
    st.success("""
#### Interpretación

La red positiva está dominada por conceptos relacionados con:
- oportunidades laborales,
- aprendizaje tecnológico,
- crecimiento profesional,
- y herramientas de IA.

Palabras como:
- machine learning
- data science
- artificial intelligence
- python

aparecen altamente conectadas, indicando que gran parte del discurso positivo 
asocia la IA con innovación y nuevas oportunidades de empleo.
""")

st.markdown("---")

# ---------------- NEUTRO ----------------
st.markdown("### Sentimiento Neutro")

col1, col2 = st.columns([1.5,1])

with col1:
    mostrar_imagen(
        "neutro.png",
        caption="Red semántica de comentarios neutros"
    )

with col2:
    st.info("""
#### Interpretación

La red neutra presenta conexiones más simples y menos densas.

Los usuarios tienden a:
- describir tecnologías,
- mencionar herramientas,
- o compartir información objetiva.

Los conceptos principales siguen siendo:
- data science,
- machine learning,
- big data,
- artificial intelligence.

Sin embargo, el tono es principalmente descriptivo y no emocional.
""")

st.markdown("---")

# ---------------- NEGATIVO ----------------
st.markdown("### Sentimiento Negativo")

col1, col2 = st.columns([1.5,1])

with col1:
    mostrar_imagen(
        "negativo.png",
        caption="Red semántica de comentarios negativos"
    )

with col2:
    st.error("""
#### Interpretación

La red negativa muestra una estructura más fragmentada y polarizada.

Las conexiones reflejan preocupaciones relacionadas con:
- sustitución laboral,
- automatización,
- incertidumbre profesional,
- y cambios en el mercado de trabajo.

Aunque los mismos conceptos tecnológicos siguen siendo centrales,
las relaciones entre nodos reflejan una percepción más crítica sobre 
el impacto de la IA en el empleo.
""")

st.markdown("---")

# ==============================================================================
# CONCLUSIÓN DEL ANÁLISIS DE RED
# ==============================================================================
st.markdown("## Conclusiones del Análisis de Redes")

st.markdown(f"""
El análisis mediante grafos permitió identificar no solo el sentimiento predominante, 
sino también la estructura interna de las conversaciones.

### Principales conclusiones:
- Los conceptos tecnológicos forman el núcleo de todas las discusiones.
- Las redes positivas son más densas y conectadas.
- Las redes negativas muestran mayor fragmentación y polarización.
- El análisis estructural complementa los resultados estadísticos obtenidos previamente.

Este enfoque demuestra cómo el análisis de redes semánticas puede enriquecer 
el estudio de opinión pública sobre inteligencia artificial y empleo.
""")

st.markdown("---")

# ==============================================================================
# SECCIÓN 7: CONCLUSIONES Y METODOLOGÍA
# ==============================================================================
st.markdown('<h2 class="section-header">7. Metodología y Conclusiones</h2>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Metodología", "Conclusiones"])

with tab1:
    st.markdown("""
    ### Metodología Aplicada
    
    #### 1. Recopilación de Datos
    - **Twitter**: Muestra de 5,000 tweets sobre IA y empleo
    - **Reddit**: Posts y comentarios de subreddits relacionados
    - **Periodo**: Datos recientes hasta 2024
    
    #### 2. Preprocesamiento
    - Limpieza de URLs, menciones y caracteres especiales
    - Normalización de texto (minúsculas, espacios)
    - Eliminación de duplicados y spam
    
    #### 3. Análisis de Sentimiento
    - **VADER**: Sentiment Intensity Analyzer para textos de redes sociales
    - **Clasificación**: Positivo (>0.05), Negativo (<-0.05), Neutro (entre -0.05 y 0.05)
    
    #### 4. ABSA (Aspect-Based Sentiment Analysis)
    - Detección de 6 aspectos clave: Empleo, Automatización, Futuro, Productividad, Ética, Economía
    - Análisis de sentimiento específico por cada aspecto
    
    #### 5. Análisis de Redes
    - Construcción de red de co-ocurrencia de palabras
    - Cálculo de métricas de centralidad
    - Detección de comunidades temáticas
    - Visualización en Gephi
    """)

with tab2:
    st.markdown(f"""
    ### Conclusiones Principales
    
    #### Hallazgos del Análisis de Sentimiento
    
    1. **Sentimiento General**: El análisis revela un sentimiento {'mayoritariamente positivo' if media > 0.1 else 'mayoritariamente negativo' if media < -0.1 else 'neutral/dividido'} 
    hacia la IA en el contexto laboral (score medio: {media:.3f}).
    
    2. **Diferencias entre Plataformas**: 
        - Twitter tiende a mostrar opiniones {'más positivas' if df[df['source']=='twitter']['sentiment'].mean() > df[df['source']=='reddit']['sentiment'].mean() else 'más negativas'} que Reddit
        - Reddit presenta discusiones {'más polarizadas' if df[df['source']=='reddit']['sentiment'].std() > df[df['source']=='twitter']['sentiment'].std() else 'más homogéneas'}
    
    3. **Aspectos Más Discutidos**:
        - Los aspectos de **{df_absa['aspecto'].value_counts().index[0]}** y **{df_absa['aspecto'].value_counts().index[1]}** son los más mencionados
        - El sentimiento varía significativamente según el aspecto analizado
    
    4. **Polarización**: Con una desviación estándar de {std:.3f}, se observa {'alta' if std > 0.4 else 'moderada' if std > 0.3 else 'baja'} 
    divergencia en las opiniones, indicando un debate {'muy activo' if std > 0.4 else 'moderadamente activo' if std > 0.3 else 'relativamente consensuado'}.
    
    #### Implicaciones
    
    - La opinión pública sobre IA y empleo está {'fuertemente dividida' if std > 0.4 else 'en proceso de formación'}
    - Es necesario abordar las preocupaciones sobre {df_absa.groupby('aspecto')['sentiment'].mean().idxmin()} (aspecto más negativo)
    - Los discursos positivos se centran principalmente en {df_absa.groupby('aspecto')['sentiment'].mean().idxmax()}
    """)

st.markdown("---")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0; border-top: 1px solid #ddd;'>
    <p><strong>Trabajo Individual - Modelos Predictivos III</strong></p>
    <p>Análisis de Sentimiento y Opinión Pública sobre IA y Empleo</p>
    <p style='font-size: 0.9rem;'>Metodología: VADER + ABSA + Análisis de Redes | Fuentes: Twitter & Reddit</p>
</div>
""", unsafe_allow_html=True)