"""
DASHBOARD.PY - Panel Interactivo de Análisis de Sentimiento sobre IA
======================================================================

Dashboard profesional construido con Streamlit para visualizar:
- Métricas clave (KPIs)
- Distribución de sentimientos
- Análisis por aspectos
- Topic Modelling
- Comparativa Twitter vs Reddit
- Insights estratégicos
- Grafos semánticos generados en Gephi

Autor: Helena Molina
Fecha: Mayo 2026
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="Análisis de Sentimiento IA",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Título del sidebar en blanco */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] .stMarkdown h2 {
    color: white !important;
}

/* Labels de filtros en blanco */
[data-testid="stSidebar"] label {
    color: white !important;
}

/* Texto general sidebar */
[data-testid="stSidebar"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #2563eb;
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #475569;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.insight-box {
    background-color: white;
    border-left: 5px solid #2563eb;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
}

h1, h2, h3 {
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES
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
        st.error(f"No se encontró el archivo: {data_path}")
        st.stop()

    return pd.read_csv(data_path)


def obtener_ruta_imagenes():
    root_dir = get_project_root()
    return root_dir / "data" / "imagenes"


# ============================================================================
# CARGA DE DATOS
# ============================================================================

df = cargar_datos()
img_dir = obtener_ruta_imagenes()

# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    '<h1 class="main-title">Análisis de Sentimiento y Opinión Pública sobre Inteligencia Artificial</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Twitter/X + Reddit | Redes semánticas + VADER Sentiment Analysis + Gephi</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================================
# INTRODUCCIÓN METODOLÓGICA
# ============================================================================

st.header("1. Contexto y Metodología")

st.markdown("""
Este estudio analiza conversaciones públicas sobre **Inteligencia Artificial** extraídas de **Twitter/X** y **Reddit**.  
Los datasets fueron sometidos a un proceso de **limpieza textual**, normalización lingüística y análisis semántico utilizando **VADER Sentiment Analysis** y técnicas de **redes de co-ocurrencia**.

Posteriormente, las conversaciones fueron agrupadas en diferentes **pilares temáticos** relacionados con empleo, automatización, ética, sociedad y aprendizaje automático, permitiendo estudiar tanto la estructura conversacional como el sentimiento asociado a cada tema.

Las redes semánticas fueron generadas mediante **NetworkX** y visualizadas en **Gephi**, limitando el número de nodos para garantizar interpretabilidad visual y explicabilidad analítica.
""")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Filtros")
st.sidebar.markdown(
    "<p style='color:white; font-size:14px;'>"
    "Los filtros se reflejan en los apartados 2 y 3."
    "</p>",
    unsafe_allow_html=True
)

fuentes_disponibles = ['Todos'] + list(df['fuente'].unique())

fuente_seleccionada = st.sidebar.selectbox(
    "Fuente",
    fuentes_disponibles
)

sentimientos_disponibles = ['Todos'] + list(df['sentimiento'].unique())

sentimiento_seleccionado = st.sidebar.selectbox(
    "Sentimiento",
    sentimientos_disponibles
)

df_filtrado = df.copy()

if fuente_seleccionada != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['fuente'] == fuente_seleccionada]

if sentimiento_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['sentimiento'] == sentimiento_seleccionado]

# ============================================================================
# KPIS
# ============================================================================

st.header("2. Métricas Clave")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Publicaciones",
        f"{len(df_filtrado):,}"
    )

with col2:

    sentimiento_medio = df_filtrado['vader_compound'].mean()

    escala_5 = ((sentimiento_medio + 1) / 2) * 5

    st.metric(
        "Sentimiento Medio",
        f"{escala_5:.1f}/5"
    )

with col3:

    polarizacion = df_filtrado['vader_compound'].std()

    nivel_polarizacion = (
        "Baja" if polarizacion < 0.25 else
        "Moderada" if polarizacion < 0.50 else
        "Alta"
    )

    st.metric(
        "Polarización",
        nivel_polarizacion
    )

with col4:

    pct_positivo = (
        (df_filtrado['sentimiento'] == 'positivo').sum()
        / len(df_filtrado)
    ) * 100

    st.metric(
        "% Positivo",
        f"{pct_positivo:.1f}%"
    )

st.markdown("---")

# ============================================================================
# SENTIMIENTO
# ============================================================================

st.header("3. Distribución de Sentimiento")

col1, col2 = st.columns(2)

colors_map = {
    'positivo': '#22c55e',
    'neutro': '#f59e0b',
    'negativo': '#ef4444'
}

with col1:

    sentimiento_counts = (
        df_filtrado['sentimiento']
        .value_counts(normalize=True)
        .reset_index()
    )

    sentimiento_counts.columns = ['sentimiento', 'porcentaje']

    fig_sent = px.bar(
        sentimiento_counts,
        x='sentimiento',
        y='porcentaje',
        color='sentimiento',
        color_discrete_map=colors_map,
        text=sentimiento_counts['porcentaje'].apply(lambda x: f"{x:.1%}")
    )

    fig_sent.update_layout(
        title="Distribución Relativa de Sentimientos",
        yaxis_title="% publicaciones",
        xaxis_title=""
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    st.info("""
La comparación entre Twitter y Reddit debe interpretarse considerando el desequilibrio muestral entre plataformas.

Twitter presenta un volumen mucho mayor de publicaciones debido a la selección para el trabajo, mientras que Reddit contiene menos publicaciones anque generalmente son más extensas y argumentativas.
""")

with col2:

    fig_polar = px.histogram(
        df_filtrado,
        x='vader_compound',
        nbins=50,
        title="Distribución del Score VADER Compound"
    )

    fig_polar.add_vline(
        x=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(fig_polar, use_container_width=True)

    st.markdown("""
**Interpretación del gráfico:**  
El score `VADER Compound` resume el sentimiento global de cada publicación en una escala entre **-1 (muy negativo)** y **+1 (muy positivo)**.

La concentración alrededor del cero indica que gran parte de las conversaciones mantienen un tono relativamente neutral o informativo. Sin embargo, las colas hacia ambos extremos reflejan polarización en temas como automatización, ética y reemplazo laboral.
""")

st.markdown("---")

# ============================================================================
# COMPARATIVA FUENTES
# ============================================================================

st.header("4. Comparativa Twitter vs Reddit")

col1, col2 = st.columns(2)

with col1:

    df_fuente_sent = (
        df.groupby(['fuente', 'sentimiento'])
        .size()
        .reset_index(name='count')
    )

    fig_fuente = px.bar(
        df_fuente_sent,
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

    fig_box.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
**Interpretación del boxplot:**  
Reddit presenta una mayor dispersión emocional, indicando conversaciones más polarizadas y argumentativas. Twitter concentra más publicaciones neutrales e informativas.

Esto sugiere que Reddit favorece debates más profundos, mientras que Twitter funciona principalmente como canal de difusión rápida de noticias y tendencias tecnológicas.
""")

st.markdown("---")

# ============================================================================
# GRAFOS
# ============================================================================

st.header("5. Redes Semánticas y Grafos Conversacionales")

st.markdown("""
No todos los aspectos detectados fueron representados mediante grafos independientes.

Aspectos como **salarios**, **educación** o **creatividad** presentaban una densidad conversacional insuficiente para generar redes interpretables sin introducir ruido visual.

Otros temas fueron agrupados debido a su fuerte solapamiento semántico:

- Empleo + Automatización
- Ética + Sociedad
- Machine Learning + Tecnología

Esto permite obtener comunidades conversacionales más coherentes y metodológicamente interpretables.
""")

# ============================================================================
# GRAFO GLOBAL
# ============================================================================

st.subheader("Grafo Global")

global_img = img_dir / "gephi_global.png"

if global_img.exists():
    st.image(str(global_img), use_container_width=True)

st.markdown("""
Los nodos más grandes, como python, engineer, learning o free, indican los términos con mayor número de conexiones dentro de la red, por lo que actúan como los temas centrales de discusión.

La elevada concentración de conexiones alrededor de estos nodos muestra que gran parte de la conversación está orientada hacia el aprendizaje técnico, el desarrollo profesional y las herramientas asociadas a la IA y funciones gratuitas de esta. Además, la predominancia de aristas verdes refleja que el sentimiento medio asociado a estas relaciones es mayoritariamente positivo, especialmente cuando se habla de programación, aprendizaje automático y oportunidades laborales.

En conjunto, el grafo evidencia que la percepción global de la IA en redes sociales se encuentra más vinculada a innovación y crecimiento profesional que a discursos negativos o de rechazo.""")

# ============================================================================
# EMPLEO + AUTOMATIZACIÓN
# ============================================================================

st.subheader("Empleo y Automatización")

empleo_img = img_dir / "gephi_empleo_automatizacion.png"

if empleo_img.exists():
    st.image(str(empleo_img), use_container_width=True)

st.markdown("""
Los nodos más importantes, como work, human, career, engineering o tech, reflejan que una parte significativa de la conversación se centra en cómo la IA está transformando el mercado laboral.

La presencia de ciudades y regiones como California, San Francisco, London o USA indica que muchos debates sobre automatización están vinculados a hubs tecnológicos concretos donde se concentra el desarrollo de IA y las oportunidades laborales del sector.

Aunque predominan las conexiones positivas, aparecen algunas aristas en tonos más rojizos alrededor de términos como tech o years que deriban de replace, lo que refleja cierta preocupación sobre la sustitución de empleo humano por sistemas automatizados. Aun así, el grafo sugiere que la conversación general está más orientada hacia adaptación profesional y nuevas oportunidades laborales que hacia una visión completamente negativa de la automatización.
""")

# ============================================================================
# ÉTICA Y SOCIEDAD
# ============================================================================

st.subheader("Ética y Sociedad")

etica_img = img_dir / "gephi_etica_sociedad.png"

if etica_img.exists():
    st.image(str(etica_img), use_container_width=True)

st.markdown("""
Este grafo representa la dimensión ética y social de la inteligencia artificial. A diferencia de otros grafos más técnicos, aquí la red aparece más dispersa, lo que sugiere que el debate ético está menos centralizado y presenta opiniones más variadas.

Los términos machinedriven, work y still muestran que las principales preocupaciones sociales giran alrededor del reemplazo de empleo por máquinas. También se muestra preocupación por el sesgo algorítmico y el impacto de la IA en las personas. 

Aunque existen algunas conexiones negativas, especialmente alrededor de términos asociados a riesgo y contratación laboral, la red mantiene una mezcla equilibrada entre preocupación y confianza tecnológica, mostrando que el debate social sobre IA sigue en desarrollo y todavía no existe un consenso claro.""")

st.markdown("---")

# ============================================================================
# TOPIC MODELLING
# ============================================================================

st.header("6. Topic Modelling")

if 'topic' in df.columns:

    topic_counts = df['topic'].value_counts().sort_index()

    fig_topics = px.bar(
        x=topic_counts.index,
        y=topic_counts.values,
        color=topic_counts.values,
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig_topics, use_container_width=True)

st.markdown("---")

# ============================================================================
# CONCLUSIÓN GLOBAL
# ============================================================================

st.header("7. Conclusiones Globales")

st.markdown("""
El análisis evidencia que la conversación pública sobre Inteligencia Artificial se encuentra dominada por perspectivas relacionadas con productividad, automatización y oportunidades laborales, especialmente en sectores tecnológicos y científicos.

Aunque el sentimiento global se mantiene moderadamente positivo, los grafos semánticos muestran focos relevantes de preocupación vinculados con ética, privacidad, sesgos algorítmicos y reemplazo humano.

Reddit presenta conversaciones más polarizadas y reflexivas, mientras que Twitter concentra difusión rápida de noticias y tendencias relacionadas con IA.

Las redes semánticas permiten observar que la percepción pública de la IA no se organiza de manera aislada, sino mediante comunidades conversacionales altamente conectadas donde empleo, automatización, aprendizaje automático y riesgos sociales aparecen estrechamente relacionados.
""")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    """
<p style='text-align:center; color:gray;'>
Trabajo Final | Modelos Predictivos III | Helena Molina | 2026
</p>
""",
    unsafe_allow_html=True
)