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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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

def mostrar_topicos_con_palabras(df_topicos, titulo):
    topic_counts = df_topicos['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Count', y='Topic', orientation='h', color='Count',
                 color_continuous_scale='Viridis', title=titulo)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Palabras representativas por tópico")
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    for topic_id in sorted(df_topicos['topic'].unique()):
        mask = df_topicos['topic'] == topic_id
        if mask.sum() >= 5:
            docs = df_topicos.loc[mask, 'texto_limpio'].fillna("").tolist()
            if docs:
                tfidf = vectorizer.fit_transform(docs)
                words = vectorizer.get_feature_names_out()
                scores = tfidf.mean(axis=0).A1
                top_idx = scores.argsort()[-5:][::-1]
                top_words = [words[i] for i in top_idx]
                st.write(f"**Topic {topic_id}** (n={mask.sum()}): {', '.join(top_words)}")

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
fuente_seleccionada = st.sidebar.selectbox("Fuente", fuentes_disponibles)

sentimientos_disponibles = ['Todos'] + list(df['sentimiento'].unique())
sentimiento_seleccionado = st.sidebar.selectbox("Sentimiento", sentimientos_disponibles)

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
    st.metric("Publicaciones", f"{len(df_filtrado):,}")
with col2:
    sentimiento_medio = df_filtrado['vader_compound'].mean()
    escala_5 = ((sentimiento_medio + 1) / 2) * 5
    st.metric("Sentimiento Medio", f"{escala_5:.1f}/5")
with col3:
    polarizacion = df_filtrado['vader_compound'].std()
    nivel_polarizacion = ("Baja" if polarizacion < 0.25 else "Moderada" if polarizacion < 0.50 else "Alta")
    st.metric("Polarización", nivel_polarizacion)
with col4:
    pct_positivo = ((df_filtrado['sentimiento'] == 'positivo').sum() / len(df_filtrado)) * 100
    st.metric("% Positivo", f"{pct_positivo:.1f}%")

st.markdown("---")

# ============================================================================
# SENTIMIENTO
# ============================================================================

st.header("3. Distribución de Sentimiento")

col1, col2 = st.columns(2)
colors_map = {'positivo': '#22c55e', 'neutro': '#f59e0b', 'negativo': '#ef4444'}

with col1:
    sentimiento_counts = df_filtrado['sentimiento'].value_counts(normalize=True).reset_index()
    sentimiento_counts.columns = ['sentimiento', 'porcentaje']
    fig_sent = px.bar(
        sentimiento_counts, x='sentimiento', y='porcentaje', color='sentimiento',
        color_discrete_map=colors_map, text=sentimiento_counts['porcentaje'].apply(lambda x: f"{x:.1%}")
    )
    fig_sent.update_layout(title="Distribución Relativa de Sentimientos", yaxis_title="% publicaciones", xaxis_title="")
    st.plotly_chart(fig_sent, use_container_width=True)
    st.info("La comparación entre Twitter y Reddit debe interpretarse considerando el desequilibrio muestral entre plataformas. Twitter presenta un volumen mucho mayor de publicaciones, mientras que Reddit contiene menos publicaciones pero generalmente son más extensas y argumentativas.")

with col2:
    fig_polar = px.histogram(df_filtrado, x='vader_compound', nbins=50, title="Distribución del Score VADER Compound")
    fig_polar.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_polar, use_container_width=True)
    st.markdown("""
    **Interpretación del gráfico:** El score `VADER Compound` resume el sentimiento global en una escala entre **-1 (muy negativo)** y **+1 (muy positivo)**. La concentración alrededor del cero indica un tono informativo/neutral. Sin embargo, las colas reflejan polarización en debates éticos y de reemplazo laboral.
    """)

st.markdown("---")

# ============================================================================
# COMPARATIVA FUENTES
# ============================================================================

st.header("4. Comparativa Twitter vs Reddit")

col1, col2 = st.columns(2)
with col1:
    df_fuente_sent = df.groupby(['fuente', 'sentimiento']).size().reset_index(name='count')
    fig_fuente = px.bar(df_fuente_sent, x='fuente', y='count', color='sentimiento', barmode='group', color_discrete_map=colors_map)
    st.plotly_chart(fig_fuente, use_container_width=True)
with col2:
    fig_box = px.box(df, x='fuente', y='vader_compound', color='fuente')
    fig_box.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("""
    **Interpretación:** Reddit presenta mayor dispersión emocional, indicando debates más argumentativos. Twitter concentra contenido neutral e informativo, operando como canal de difusión rápida.
    """)

st.markdown("---")

# ============================================================================
# GRAFOS Y PILARES DINÁMICOS
# ============================================================================

st.header("5. Análisis por Pilares Temáticos (Redes Semánticas)")

st.markdown("""
Las redes semánticas evidencian que los aspectos detectados no operan de forma aislada. Se han agrupado por proximidad conversacional (ej. Empleo + Automatización) para obtener comunidades interpretables.
""")

# Diccionario centralizado de pilares y sus insights basados en la metadata de Gephi
pilares = {
    "Visión Global": {
        "img": "gephi_global.png",
        "insight": """
        **Conclusión Estratégica:** El grafo muestra una fuerte conexión entre términos como `bot`, `trending` y `article, indicando que gran parte del volumen global proviene de cuentas automatizadas de difusión de noticias y tendencias tecnológicas. 
        
        Además, el nodo `python` destaca por su alta conectividad e intermediación (betweenness), sirviendo de puente principal entre el desarrollo algorítmico y otras áreas de discusión técnica. El sentimiento general es mayormente positivo pero fuertemente impulsado por contenido automatizado y educativo.
        """
    },
    "Empleo y Automatización": {
        "img": "gephi_empleo_automatizacion.png",
        "insight": """
        **Conclusión Estratégica:** A diferencia del pánico al reemplazo laboral, la red revela que la conversación está orientada hacia la transición profesional. Destaca una fuerte interrelación entre `hiring`, `engineer` y referencias de hubs geográficos como `united` (probablemente referenciando United States).
        
        Las conexiones más sólidas vinculan la automatización no con la destrucción, sino con la necesidad de aprendizaje constante (`learning`, `aiml`). La conversación percibe la IA como un reestructurador del mercado, más que como su ejecutor.
        """
    },
    "Ética y Sociedad": {
        "img": "gephi_etica_sociedad.png",
        "insight": """
        **Conclusión Estratégica:** Este pilar presenta la estructura más crítica y polarizada. Es el único grafo donde nodos clave como `risk` muestran una puntuación de sentimiento marcadamente negativa (-0.05), y términos como `bias` aparecen con un sentimiento neutral pero de alta fricción conversacional.
        
        Se evidencia una preocupación tangible por la intersección entre la seguridad técnica (`security`) y la dimensión humana (`human`). La conversación aquí no asume un consenso optimista, reflejando cautela regulatoria y social.
        """
    },
    "Educación y Habilidades": {
        "img": "gephi_educacion_habilidades.png",
        "insight": """
        **Conclusión Estratégica:** Refleja la altísima movilización del ecosistema formativo alrededor de la inteligencia artificial. Nodos como `free` presentan sentimientos fuertemente positivos (0.63), demostrando la enorme valoración de los recursos educativos abiertos y democratizados.
        
        La tríada conformada por `python`, `data` y `machine`  conforma el núcleo inamovible de las competencias técnicas demandadas. El aprendizaje de la IA se percibe de forma casi unánime como una oportunidad inminente de empoderamiento personal.
        """
    },
    "Machine Learning & Data Science": {
        "img": "gephi_machine_learning_datascience.png",
        "insight": """
        **Conclusión Estratégica:** Ecosistema altamente técnico centrado en la implementación práctica. Los nodos dominantes giran alrededor del ecosistema de OpenAI (`chatgpt`, `gpt`, `openai`), consolidándolo como el estándar de facto en las discusiones de vanguardia de ciencia de datos.
        
        Se trata de una red que mantiene un tono informativo y altamente optimista, evidenciando cómo los modelos de lenguaje están redefiniendo el análisis clásico (`data`, `python`).
        """
    },
    "Negocio & Productividad": {
        "img": "gephi_negocio_productividad.png",
        "insight": """
        **Conclusión Estratégica:** Enfoque orientado a la rentabilidad y eficiencia. La inteligencia artificial se ha desmarcado de la ciencia ficción para volverse una herramienta corporativa estándar. Nodos relacionales como `help` alcanzan una positividad muy alta (0.61).
        
        Toda la estructura orbita en torno a la eficiencia operacional (`business`, `team`, `analytics`). La conversación demuestra que los usuarios perciben la IA como el habilitador principal para potenciar las estrategias de marketing y análisis corporativo.
        """
    },
    "Oportunidades": {
        "img": "gephi_oportunidades.png",
        "insight": """
        **Conclusión Estratégica:** El discurso está gobernado por el entusiasmo tecnológico y el progreso continuo. La red se organiza a través de verbos de acción positiva y adjetivos prospectivos: palabras como `improve` (0.62) o `better` (0.53) registran las intensidades emocionales más positivas del estudio.
        
        Las discusiones convergen en torno a la exploración de capacidades futuras (`future`, `potential`), proyectando a la IA como la palanca definitiva para optimizar el trabajo intelectual y creativo.
        """
    }
}

# Selector interactivo
opcion_pilar = st.selectbox("Selecciona el Pilar Analítico a visualizar:", list(pilares.keys()))

st.markdown(f"### {opcion_pilar}")

pilar_data = pilares[opcion_pilar]
img_path = img_dir / pilar_data["img"]

col_img, col_text = st.columns([1.5, 1])

with col_img:
    if img_path.exists():
        st.image(str(img_path), use_container_width=True, caption=f"Grafo de Co-ocurrencia Semántica: {opcion_pilar}")
    else:
        st.warning(f"No se encontró la imagen: {pilar_data['img']} en la carpeta de imágenes.")

with col_text:
    st.markdown(pilar_data["insight"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# 6. TOPIC MODELLING (BERT + KMeans)
# ============================================================================

st.header("6. Topic Modelling")

if 'topic_precalc' in df_filtrado.columns and df_filtrado['topic_precalc'].notna().all() and (df_filtrado['topic_precalc'] != -1).any():
    df_temp = df_filtrado.copy()
    df_temp['topic'] = df_temp['topic_precalc']
    mostrar_topicos_con_palabras(df_temp, "Distribución de Documentos por Tópico (Precalculado con BERT)")
else:
    st.warning("No se encontraron tópicos precalculados. Calculando con TF-IDF + KMeans (rápido)...")
    textos = df_filtrado['texto_limpio'].fillna("").tolist()
    if len(textos) >= 20:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(textos)
        n_clusters = min(5, len(set(textos)) - 1, max(2, len(textos) // 50 + 1))
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            df_temp = df_filtrado.copy()
            df_temp['topic'] = clusters
            mostrar_topicos_con_palabras(df_temp, "Distribución de Documentos por Tópico (TF-IDF + KMeans)")
        else:
            st.info("No hay suficiente variedad de textos para generar tópicos significativos.")
    else:
        st.info("Pocos datos para topic modelling.")

st.markdown("---")

# ============================================================================
# CONCLUSIÓN GLOBAL
# ============================================================================

st.header("7. Conclusiones Globales")

st.markdown("""
El análisis evidencia que la conversación pública sobre Inteligencia Artificial se encuentra dominada por una retórica de **productividad, adaptación profesional y expansión de capacidades**. 

A diferencia de la especulación recurrente sobre el reemplazo masivo, los datos semánticos demuestran que los usuarios perciben la IA como una transición técnica obligatoria: la adopción de herramientas (especialmente del ecosistema de lenguajes como Python y modelos generativos como GPT) es vista como una ventaja competitiva clave.

Sin embargo, persisten focos críticos consolidados —concentrados en mayor medida en plataformas como Reddit— que exigen resolver los debates sobre **privacidad, sesgos (`bias`) algorítmicos y riesgos (`risk`) éticos** antes de alcanzar una asimilación corporativa y social madura.
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