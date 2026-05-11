"""

DASHBOARD.PY - Panel Interactivo de Análisis de Sentimiento sobre IA

======================================================================

"""

import streamlit as st

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from pathlib import Path

# ============================================================================

# CONFIGURACIÓN

# ============================================================================

st.set_page_config(

    page_title="Análisis de Sentimiento IA",

    page_icon="🤖",

    layout="wide",

    initial_sidebar_state="expanded"

)

st.markdown("""

<style>

    .main-title {

        font-size: 2.5rem;

        font-weight: 700;

        color: #1f77b4;

        text-align: center;

        margin-bottom: 1rem;

    }

    .subtitle {

        font-size: 1.2rem;

        color: #555;

        text-align: center;

        margin-bottom: 2rem;

    }

</style>

""", unsafe_allow_html=True)

# ============================================================================

# RUTAS

# ============================================================================

def get_project_root():

    current_dir = Path(__file__).resolve().parent

    for _ in range(3):

        if (current_dir / "data").exists():

            return current_dir

        current_dir = current_dir.parent

    return Path(__file__).resolve().parent

@st.cache_data

def cargar_datos():

    root_dir = get_project_root()

    data_path = root_dir / "data" / "clean" / "datos_unificados.csv"

    if not data_path.exists():

        st.error("No se encontró datos_unificados.csv")

        st.stop()

    return pd.read_csv(data_path)

def obtener_ruta_imagenes():

    root_dir = get_project_root()

    return root_dir / "data" / "imagenes"

# ============================================================================

# DATOS

# ============================================================================

df = cargar_datos()

img_dir = obtener_ruta_imagenes()

# ============================================================================

# HEADER

# ============================================================================

st.markdown('<h1 class="main-title">Análisis de Sentimiento y Opinión Pública sobre IA</h1>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Twitter + Reddit | VADER + ABSA + BERTopic + Gephi</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================

# INTRODUCCIÓN

# ============================================================================

st.header("1. Descripción del Proyecto y Tratamiento de Datos")

st.markdown("""

Este proyecto analiza la conversación pública sobre Inteligencia Artificial y empleo utilizando publicaciones de Twitter y Reddit.

El objetivo principal es identificar:

- La percepción social hacia la IA.

- Los principales temas de conversación.

- Los niveles de polarización.

- Las diferencias entre plataformas.

- Los conceptos más conectados dentro del debate.

### Dataset utilizado

Se combinaron datasets procedentes de:

- Twitter/X

- Reddit

El dataset final contiene miles de publicaciones relacionadas con:

- Machine Learning

- Inteligencia Artificial

- Empleo y automatización

- Productividad

- Educación tecnológica

- Ética y riesgos de la IA

### Preprocesamiento aplicado

Los textos fueron limpiados mediante:

- Eliminación de URLs.

- Eliminación de emojis y caracteres especiales.

- Conversión a minúsculas.

- Eliminación de stopwords.

- Tokenización.

- Eliminación de duplicados.

### Técnicas aplicadas

- VADER Sentiment Analysis

- Aspect-Based Sentiment Analysis (ABSA)

- BERTopic

- Redes semánticas con Gephi

- PMI (Pointwise Mutual Information)

- Centralidad y comunidades

""")

st.markdown("---")

# ============================================================================

# KPIS

# ============================================================================

st.header("2. Métricas Clave")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric("Total publicaciones", f"{len(df):,}")

with col2:

    st.metric("Sentimiento medio", f"{df['vader_compound'].mean():.3f}")

with col3:

    st.metric("Polarización", f"{df['vader_compound'].std():.3f}")

with col4:

    pct_pos = (df['sentimiento'] == 'positivo').sum() / len(df) * 100

    st.metric("% Positivo", f"{pct_pos:.1f}%")

st.markdown("---")

# ============================================================================

# DISTRIBUCIÓN SENTIMIENTO

# ============================================================================

st.header("3. Distribución de Sentimiento")

col1, col2 = st.columns(2)

colors_map = {

    'positivo': '#27ae60',

    'neutro': '#f39c12',

    'negativo': '#e74c3c'

}

with col1:

    sentimiento_counts = df['sentimiento'].value_counts()

    fig = go.Figure(data=[

        go.Bar(

            x=sentimiento_counts.index,

            y=sentimiento_counts.values,

            marker_color=[colors_map[s] for s in sentimiento_counts.index]

        )

    ])

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig2 = px.histogram(

        df,

        x='vader_compound',

        nbins=50,

        color_discrete_sequence=['#3498db']

    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""

### Resultados

El análisis revela una conversación predominantemente positiva hacia la IA, aunque existe una polarización moderada.

""")

st.markdown("---")

# ============================================================================

# COMPARATIVA

# ============================================================================

st.header("4. Comparativa Twitter vs Reddit")

col1, col2 = st.columns(2)

with col1:

    df_fuente = df.groupby(['fuente', 'sentimiento']).size().reset_index(name='count')

    fig_fuente = px.bar(

        df_fuente,

        x='fuente',

        y='count',

        color='sentimiento',

        barmode='group',

        color_discrete_map=colors_map

    )

    st.plotly_chart(fig_fuente, use_container_width=True)

with col2:

    fig_box = px.box(

        df,

        x='fuente',

        y='vader_compound',

        color='fuente'

    )

    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("""

### Resultados

Twitter presenta conversaciones más rápidas y generalmente más optimistas.

Reddit contiene debates más largos y polarizados, especialmente en temas relacionados con automatización y riesgos futuros.

""")

st.markdown("---")

# ============================================================================

# GRAFOS GEPHI

# ============================================================================

st.header("5. Redes Semánticas y Pilares Conversacionales")

st.markdown("""

Los grafos fueron construidos mediante redes de co-ocurrencia.

Cada nodo representa una palabra relevante.

Cada arista representa la relación semántica entre conceptos.

Las aristas fueron coloreadas según el sentimiento medio calculado con VADER:

- Verde → positivo

- Amarillo → neutro

- Rojo → negativo

El tamaño de los nodos depende de su centralidad y frecuencia.

""")

# ============================================================================

# GRAFO GLOBAL

# ============================================================================

st.subheader("5.1 Grafo Global")

ruta_global = img_dir / "gephi_global.png"

if ruta_global.exists():

    st.image(str(ruta_global), use_container_width=True)

st.markdown("""

### Interpretación del grafo global

El grafo global concentra los conceptos más importantes del debate sobre IA.

Los nodos más centrales corresponden a:

- datascience

- machinelearning

- artificialintelligence

- bigdata

Esto indica que gran parte de la conversación se centra en:

- aplicaciones técnicas de IA,

- desarrollo profesional,

- automatización,

- herramientas de machine learning.

Las conexiones densas muestran una fuerte asociación entre ciencia de datos, aprendizaje automático y productividad.

""")

st.markdown("---")

# ============================================================================

# PILARES

# ============================================================================

pilares = {

    'gephi_empleo_automatizacion.png': {

        'titulo': 'Empleo y Automatización',

        'texto': 'Este pilar representa el debate sobre reemplazo laboral, automatización y futuro del empleo.'

    },

    'gephi_machine_learning_datascience.png': {

        'titulo': 'Machine Learning y Data Science',

        'texto': 'Este grafo concentra discusiones técnicas relacionadas con Python, TensorFlow y modelos de IA.'

    },

    'gephi_educacion_habilidades.png': {

        'titulo': 'Educación y Habilidades',

        'texto': 'Las conversaciones se enfocan en aprendizaje, cursos online, capacitación y habilidades tecnológicas.'

    },

    'gephi_negocio_productividad.png': {

        'titulo': 'Negocio y Productividad',

        'texto': 'Este pilar muestra el uso empresarial de la IA para eficiencia y productividad.'

    },

    'gephi_etica_sociedad.png': {

        'titulo': 'Ética y Sociedad',

        'texto': 'Incluye preocupaciones sobre riesgos, regulación, privacidad y sesgos algorítmicos.'

    }

}

for archivo, contenido in pilares.items():

    st.subheader(contenido['titulo'])

    ruta_img = img_dir / archivo

    if ruta_img.exists():

        st.image(str(ruta_img), use_container_width=True)

    st.markdown(contenido['texto'])

    st.markdown("""

    #### Resultados observados

    - Los nodos más conectados representan conceptos centrales del debate.

    - Las conexiones rojas muestran asociaciones negativas.

    - Las conexiones verdes indican percepción positiva.

    - Las conexiones neutras reflejan conversación técnica o descriptiva.

    """)

    st.markdown("---")

# ============================================================================

# ASPECTOS

# ============================================================================

st.header("6. Análisis por Aspectos")

columnas_aspectos = [col for col in df.columns if col.startswith('aspecto_')]

aspectos_data = []

for col in columnas_aspectos:

    aspecto = col.replace('aspecto_', '')

    df_sub = df[df[col] == True]

    aspectos_data.append({

        'Aspecto': aspecto,

        'Menciones': len(df_sub),

        'Sentimiento': df_sub['vader_compound'].mean()

    })

df_aspectos = pd.DataFrame(aspectos_data)

col1, col2 = st.columns(2)

with col1:

    fig_a = px.bar(

        df_aspectos,

        y='Aspecto',

        x='Menciones',

        orientation='h',

        color='Menciones'

    )

    st.plotly_chart(fig_a, use_container_width=True)

with col2:

    fig_b = px.bar(

        df_aspectos,

        y='Aspecto',

        x='Sentimiento',

        orientation='h',

        color='Sentimiento',

        color_continuous_scale='RdYlGn'

    )

    st.plotly_chart(fig_b, use_container_width=True)

st.markdown("""

### Conclusiones del análisis por aspectos

Los temas relacionados con productividad y oportunidades suelen mostrar sentimientos positivos.

Los aspectos relacionados con reemplazo laboral y riesgos presentan mayor negatividad.

La conversación pública refleja tanto entusiasmo tecnológico como preocupación social.

""")

st.markdown("---")

# ============================================================================

# TOPIC MODELLING

# ============================================================================

st.header("7. Topic Modelling")

if 'topic' in df.columns:

    topic_counts = df['topic'].value_counts().sort_index()

    fig_topics = px.bar(

        x=topic_counts.index,

        y=topic_counts.values,

        color=topic_counts.values,

        color_continuous_scale='Viridis'

    )

    st.plotly_chart(fig_topics, use_container_width=True)

st.markdown("""

### Resultados del Topic Modelling

BERTopic permitió identificar automáticamente grupos temáticos dentro del dataset.

Los tópicos más frecuentes están relacionados con:

- Machine Learning

- Programación

- Empleo

- Productividad

- Educación tecnológica

""")

st.markdown("---")

# ============================================================================

# CONCLUSIONES

# ============================================================================

st.header("8. Conclusiones Finales")

st.markdown(f"""

## Resultados principales

- El sentimiento medio del dataset es {df['vader_compound'].mean():.3f}.

- Existe una polarización de {df['vader_compound'].std():.3f}.

- Twitter presenta conversaciones más optimistas.

- Reddit contiene mayor profundidad argumentativa.

- Los temas más importantes giran alrededor de Machine Learning, automatización y empleo.

## Conclusión general

La opinión pública sobre IA es predominantemente positiva, aunque existe una preocupación significativa respecto al impacto laboral y ético.

La IA se percibe simultáneamente como:

- una oportunidad de crecimiento,

- una herramienta de productividad,

- y un posible riesgo de automatización laboral.

El análisis de redes semánticas demuestra que los pilares conversacionales están altamente conectados entre sí y permiten visualizar cómo se estructura el discurso social sobre inteligencia artificial.

""")

st.markdown("---")

st.markdown(

    "<p style='text-align:center;color:gray;'>Trabajo Final - Modelos Predictivos III | Helena Molina | 2026</p>",

    unsafe_allow_html=True

)