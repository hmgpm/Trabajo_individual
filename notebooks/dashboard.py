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

# CONFIGURACIÓN INICIAL
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

# CSS PERSONALIZADO
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


# FUNCIONES PRINCIPALES

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

# CARGA DE DATOS

df = cargar_datos()
img_dir = obtener_ruta_imagenes()

# HEADER
st.markdown(
    '<h1 class="main-title">Análisis de Sentimiento y Opinión Pública sobre Inteligencia Artificial</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Twitter/X + Reddit | Redes semánticas + VADER Sentiment Analysis + Gephi</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# INTRODUCCIÓN METODOLÓGICA


st.header("1. Contexto y Metodología")
st.markdown("""
Este estudio analiza conversaciones públicas sobre **Inteligencia Artificial** extraídas de **Twitter/X** y **Reddit**.  
Los datasets fueron sometidos a un proceso de **limpieza textual**, normalización lingüística y análisis semántico utilizando **VADER Sentiment Analysis** y técnicas de **redes de co-ocurrencia**.

Posteriormente, las conversaciones fueron agrupadas en diferentes **pilares temáticos** relacionados con empleo, automatización, ética, sociedad y aprendizaje automático, permitiendo estudiar tanto la estructura conversacional como el sentimiento asociado a cada tema.

Las redes semánticas fueron generadas mediante **NetworkX** y visualizadas en **Gephi**, limitando el número de nodos para garantizar interpretabilidad visual y explicabilidad analítica.
""")
st.markdown("---")

# SIDEBAR

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

# KPIS

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

# SENTIMIENTO

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
    st.info("La comparación entre Twitter y Reddit debe interpretarse considerando el desequilibrio muestral entre plataformas. Twitter presenta un volumen mucho mayor de publicaciones, ya que la mustra seleccionada es mayor. Por otro lado, Reddit tiene una menor muestra, pero generalmente las conversaciones son más extensas y argumentativas.")

with col2:
    fig_polar = px.histogram(df_filtrado, x='vader_compound', nbins=50, title="Distribución del Score VADER Compound")
    fig_polar.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_polar, use_container_width=True)
    st.markdown("""
    **Interpretación del gráfico:** El score `VADER Compound` resume el sentimiento global en una escala entre **-1 (muy negativo)** y **+1 (muy positivo)**. La concentración alrededor del cero indica un tono informativo/neutral. Sin embargo, las colas reflejan polarización en debates éticos y de reemplazo laboral.
    """)

st.markdown("---")

# COMPARATIVA FUENTES

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
    **Conclusiones:** Reddit presenta mayor dispersión emocional, indicando debates más argumentativos. Twitter concentra contenido neutral e informativo, operando como canal de difusión.
    """)

st.markdown("---")

# 
# SECCIÓN 5: ANÁLISIS POR ASPECTOS (ABSA)

st.header("5. Análisis por Aspectos (ABSA)")

st.markdown("""
A continuación se muestra la relación entre los **aspectos temáticos** detectados en las conversaciones y el **sentimiento** asociado a cada uno.  
Estos aspectos han sido definidos mediante listas de palabras clave (empleo, automatización, ética, productividad, etc.).
""")

# Obtener columnas de aspectos (todas las que empiezan por 'aspecto_')
aspect_cols = [col for col in df_filtrado.columns if col.startswith('aspecto_')]

if len(aspect_cols) == 0:
    st.warning("No se encontraron columnas de aspectos en los datos. Ejecuta primero 'main.py' con la detección de aspectos.")
else:
    # Preparamos los datos para los gráficos
    data_counts = []   # para barras apiladas (conteos)
    data_pct = []      # para porcentajes
    
    for col in aspect_cols:
        aspecto = col.replace('aspecto_', '').capitalize()
        mask = df_filtrado[col] == True
        total = mask.sum()
        if total > 0:
            for sent in ['positivo', 'neutro', 'negativo']:
                count = (df_filtrado[mask]['sentimiento'] == sent).sum()
                if count > 0:
                    data_counts.append({
                        'Aspecto': aspecto,
                        'Sentimiento': sent.capitalize(),
                        'Conteo': count,
                        'Porcentaje': (count / total) * 100
                    })
    
    df_absa = pd.DataFrame(data_counts)
    
    if df_absa.empty:
        st.info("No hay suficientes menciones de aspectos para mostrar gráficos.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Conteo de publicaciones por aspecto")
            # Gráfico de barras apiladas (Plotly)
            pivot_counts = df_absa.pivot(index='Aspecto', columns='Sentimiento', values='Conteo').fillna(0)
            # Ordenar por total descendente
            pivot_counts['Total'] = pivot_counts.sum(axis=1)
            pivot_counts = pivot_counts.sort_values('Total', ascending=False).drop('Total', axis=1)
            
            fig_counts = go.Figure()
            for sent in ['Positivo', 'Neutro', 'Negativo']:
                if sent in pivot_counts.columns:
                    fig_counts.add_trace(go.Bar(
                        name=sent,
                        x=pivot_counts.index,
                        y=pivot_counts[sent],
                        marker_color={'Positivo': '#27ae60', 'Neutro': '#f39c12', 'Negativo': '#e74c3c'}[sent]
                    ))
            fig_counts.update_layout(
                barmode='stack',
                title="Número de publicaciones por aspecto y sentimiento",
                xaxis_title="Aspecto",
                yaxis_title="Conteo",
                legend_title="Sentimiento",
                height=500
            )
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col2:
            st.subheader(" Distribución porcentual por aspecto")
            # Gráfico de barras horizontales apiladas (porcentajes)
            pivot_pct = df_absa.pivot(index='Aspecto', columns='Sentimiento', values='Porcentaje').fillna(0)
            # Reordenar según el mismo orden que el gráfico de conteos
            pivot_pct = pivot_pct.reindex(pivot_counts.index)
            
            fig_pct = go.Figure()
            for sent in ['Positivo', 'Neutro', 'Negativo']:
                if sent in pivot_pct.columns:
                    fig_pct.add_trace(go.Bar(
                        name=sent,
                        y=pivot_pct.index,
                        x=pivot_pct[sent],
                        orientation='h',
                        marker_color={'Positivo': '#27ae60', 'Neutro': '#f39c12', 'Negativo': '#e74c3c'}[sent]
                    ))
            fig_pct.update_layout(
                barmode='stack',
                title="Porcentaje de sentimiento por aspecto",
                xaxis_title="Porcentaje (%)",
                yaxis_title="Aspecto",
                legend_title="Sentimiento",
                height=500
            )
            st.plotly_chart(fig_pct, use_container_width=True)
        
        # Tabla resumen (opcional)
        with st.expander(" Ver tabla resumen de conteos y porcentajes"):
            st.dataframe(df_absa.pivot(index='Aspecto', columns='Sentimiento', values=['Conteo', 'Porcentaje']).round(1))

st.markdown("---")

# GRAFOS Y PILARES DINÁMICOS

st.header("6. Análisis por Pilares Temáticos (Redes Semánticas)")

st.markdown("""
No todos los aspectos detectados fueron representados mediante grafos independientes.

Aspectos como **salarios**, **educación** o **creatividad** presentaban una densidad conversacional insuficiente para generar redes interpretables sin introducir ruido visual.

Otros temas fueron agrupados debido a su fuerte solapamiento semántico:

- Empleo + Automatización
- Ética + Sociedad
- Machine Learning + Tecnología

Esto permite obtener comunidades conversacionales más coherentes e interpretables.""")

# Diccionario centralizado de pilares y sus insights basados en Gephi
pilares = {
    "Visión Global": {
        "img": "gephi_global.png",
        "insight": """
        **Insights:** El grafo global muestra que la conversación sobre inteligencia artificial está dominada por términos técnicos y profesionales, donde los nodos más grandes y conectados son “python”, “free” y “engineering”. Esto indica que gran parte del debate se centra en programación, acceso abierto a herramientas y desarrollo tecnológico.
También aparecen nodos como “bot”, “trending” y “amp”, que probablemente estén asociados a cuentas automatizadas o sistemas de difusión masiva de contenido. Esto sugiere que parte de la conversación puede estar influenciada por bots que amplifican determinados temas o noticias relacionadas con IA.
Aun así, el análisis de sentimiento de la red es mayoritariamente positivo, lo que refleja una percepción optimista hacia el avance tecnológico y las oportunidades asociadas a la inteligencia artificial.
"""
    },
    "Empleo y Automatización": {
        "img": "gephi_empleo_automatizacion.png",
        "insight": """
        **Insights:** En este pilar destacan nodos como “hiring”, “machine”, “learning” y “machinedriven”, que representan el núcleo real de la conversación sobre empleo y automatización. La red muestra que la IA está muy relacionada con nuevas oportunidades laborales, perfiles técnicos y procesos automatizados de trabajo.
También aparecen términos como “elbasheer”, “granola” y “ramsey”, que probablemente provienen de bots o cuentas automatizadas que generan contenido repetitivo dentro del dataset. Aunque tienen conexiones fuertes, no representan conceptos relevantes del debate sobre IA.
El sentimiento global del grafo es positivo, indicando que la automatización y el desarrollo de sistemas inteligentes se perciben más como una oportunidad de crecimiento y evolución profesional que como una amenaza directa para el empleo.
"""
    },
    "Ética y Sociedad": {
        "img": "gephi_etica_sociedad.png",
        "insight": """
        **Insights:** El grafo de ética y sociedad presenta una conversación principalmente positiva, especialmente alrededor de nodos como “chatgpt”, “data”, “human” y “security”, que concentran gran parte de las conexiones relevantes. Esto refleja que la discusión ética sobre IA está muy vinculada a la seguridad, el uso de datos y la interacción entre humanos y sistemas inteligentes.
Sin embargo, existen ramas con un sentimiento más crítico. El nodo “risk” presenta un sentimiento claramente negativo, asociado a preocupaciones sobre riesgos tecnológicos y posibles impactos sociales. Por otro lado, “bias” mantiene un sentimiento más neutro, lo que indica que el sesgo algorítmico se debate desde una perspectiva más analítica y menos emocional.
En conjunto, la red refleja que la percepción ética de la IA es relativamente optimista, aunque persisten preocupaciones concretas relacionadas con riesgos y responsabilidad tecnológica.
"""
    },
    "Educación y Habilidades": {
        "img": "gephi_educacion_habilidades.png",
        "insight": """
        **Insights:** Este grafo muestra una percepción claramente positiva de la IA en el ámbito educativo y del desarrollo de habilidades. Las conexiones más fuertes aparecen entre los nodos “learning”, “data”, “machine” y “learn”, lo que indica que el aprendizaje técnico y la formación en ciencia de datos son temas centrales dentro de la conversación.
La red refleja cómo muchos usuarios asocian la inteligencia artificial con oportunidades de aprendizaje, adquisición de nuevas competencias y mejora profesional. Además, la fuerte conexión entre términos educativos y tecnológicos demuestra que el interés por la IA está muy ligado al crecimiento académico y laboral.
"""
    },
    "Machine Learning & Data Science": {
        "img": "gephi_machine_learning_datascience.png",
        "insight": """
        **Insights:** El pilar de Machine Learning y Data Science presenta un sentimiento global positivo y está centrado principalmente en nodos como “chatgpt”, “data” y “gpt”, que son los conceptos más conectados y relevantes de la red. Esto confirma que los modelos generativos y el análisis de datos dominan gran parte de la conversación técnica sobre inteligencia artificial.
Al igual que en otros grafos, vuelven a aparecer términos como “digested” y “elbasheer”, lo que sugiere la presencia de bots o sistemas automáticos de difusión de contenido dentro del dataset. Aunque generan conexiones importantes por frecuencia, no representan conceptos técnicos clave del análisis.
En general, la red muestra una comunidad muy enfocada en innovación tecnológica, herramientas de IA generativa y aplicaciones prácticas relacionadas con Data Science y Machine Learning.
"""
    },
    "Negocio & Productividad": {
        "img": "gephi_negocio_productividad.png",
        "insight": """
        **Insights:** El grafo de negocio y productividad presenta un sentimiento claramente positivo, lo que indica que la inteligencia artificial se percibe principalmente como una herramienta para mejorar procesos y generar valor dentro de las organizaciones. Los nodos más conectados son “business”, “data” y “team”, reflejando que la conversación gira alrededor del uso estratégico de los datos y del trabajo colaborativo apoyado por IA.
La red muestra que muchas publicaciones relacionan la inteligencia artificial con eficiencia empresarial, toma de decisiones y optimización de equipos de trabajo. En general, la percepción dominante es que la IA puede actuar como un apoyo para aumentar la productividad y facilitar la innovación dentro del entorno corporativo.
"""
    },
    "Oportunidades": {
        "img": "gephi_oportunidades.png",
        "insight": """
        **Insights:** El pilar de oportunidades también presenta un sentimiento mayoritariamente positivo. Las palabras con más conexiones dentro de la red son “new”, “data”, “future” y “chatgpt”, lo que refleja una conversación muy orientada hacia innovación, crecimiento tecnológico y posibilidades futuras de la inteligencia artificial.
La presencia de términos relacionados con futuro y nuevas tecnologías muestra que muchos usuarios ven la IA como una oportunidad de transformación tanto profesional como empresarial. Además, la importancia de “chatgpt” dentro de la red confirma el fuerte impacto que los modelos generativos están teniendo en la percepción pública de la inteligencia artificial y sus aplicaciones futuras.
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


# 7. TOPIC MODELLING (BERT + KMeans)

def generar_insight_topic(topic_id, palabras, n_docs):
    """
    Genera insights dinámicos según las palabras principales del tópico.
    """

    palabras_texto = " ".join(palabras).lower()

    # Detectar temática automáticamente
    if any(p in palabras_texto for p in ['chatgpt', 'gpt', 'llm', 'openai']):
        tema = "IA generativa y modelos de lenguaje"
        insight = (
            "Este tópico está relacionado con herramientas de IA generativa y modelos "
            "de lenguaje. La presencia de términos como ChatGPT y GPT refleja el fuerte "
            "impacto de los modelos generativos dentro de la conversación online."
        )

    elif any(p in palabras_texto for p in ['hiring', 'job', 'career', 'scientist']):
        tema = "Empleo y mercado laboral"
        insight = (
            "El tópico refleja conversaciones relacionadas con empleo, contratación y "
            "perfiles profesionales vinculados a Data Science e inteligencia artificial."
        )

    elif any(p in palabras_texto for p in ['learning', 'machine', 'python', 'model']):
        tema = "Machine Learning y aprendizaje técnico"
        insight = (
            "Las palabras principales muestran un enfoque técnico orientado a Machine Learning, "
            "programación y desarrollo de modelos de inteligencia artificial."
        )

    elif any(p in palabras_texto for p in ['ethics', 'bias', 'risk', 'privacy', 'security']):
        tema = "Ética y riesgos de la IA"
        insight = (
            "El tópico agrupa preocupaciones relacionadas con ética, privacidad, sesgos "
            "algorítmicos y riesgos asociados al uso de inteligencia artificial."
        )

    elif any(p in palabras_texto for p in ['business', 'team', 'productivity', 'strategy']):
        tema = "Negocio y productividad"
        insight = (
            "La conversación se centra en aplicaciones empresariales de la IA, especialmente "
            "en productividad, estrategia y optimización de procesos."
        )

    elif any(p in palabras_texto for p in ['bot', 'rss', 'amp', 'trending']):
        tema = "Automatización y difusión de contenido"
        insight = (
            "La presencia de términos relacionados con bots y difusión automática sugiere "
            "que parte del contenido puede provenir de cuentas automatizadas o agregadores."
        )

    else:
        tema = "Conversación general sobre IA"
        insight = (
            "El tópico representa una conversación general sobre inteligencia artificial "
            "sin una temática claramente especializada."
        )

    return {
        "tema": tema,
        "insight": insight,
        "n_docs": n_docs
    }
st.header("7. Topic Modelling")

if 'topic_precalc' in df_filtrado.columns and df_filtrado['topic_precalc'].notna().all() and (df_filtrado['topic_precalc'] != -1).any():
    df_temp = df_filtrado.copy()
    df_temp['topic'] = df_temp['topic_precalc']
    mostrar_topicos_con_palabras(df_temp, "Distribución de Documentos por Tópico (Precalculado con BERT)")
    
    # INSIGHTS DINÁMICOS

    st.subheader("Insights de los Tópicos")

    for topic_id in sorted(df_temp['topic'].unique()):

        docs_topic = df_temp[df_temp['topic'] == topic_id]

        # Obtener palabras frecuentes
        texto_topic = " ".join(docs_topic['texto_limpio'].astype(str))

        palabras = (
            pd.Series(texto_topic.split())
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        resultado = generar_insight_topic(
            topic_id,
            palabras,
            len(docs_topic)
        )

        st.markdown(f"""
    ### Topic {topic_id} — {resultado['tema']}

    **Palabras clave:** {", ".join(palabras)}

    {resultado['insight']}

    - Número de documentos: **{resultado['n_docs']:,}**
    """)
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
            
            # INSIGHTS DINÁMICOS

            st.subheader("Insights de los Tópicos")

            for topic_id in sorted(df_temp['topic'].unique()):

                docs_topic = df_temp[df_temp['topic'] == topic_id]

                # Obtener palabras frecuentes
                texto_topic = " ".join(docs_topic['texto_limpio'].astype(str))

                palabras = (
                    pd.Series(texto_topic.split())
                    .value_counts()
                    .head(5)
                    .index
                    .tolist()
                )

                resultado = generar_insight_topic(
                    topic_id,
                    palabras,
                    len(docs_topic)
                )

                st.markdown(f"""
            ### Topic {topic_id} — {resultado['tema']}

            **Palabras clave:** {", ".join(palabras)}

            {resultado['insight']}

            - Número de documentos: **{resultado['n_docs']:,}**
            """)
        else:
            st.info("No hay suficiente variedad de textos para generar tópicos significativos.")
    else:
        st.info("Pocos datos para topic modelling.")

st.markdown("---")

# CONCLUSIÓN GLOBAL

st.header("8. Conclusiones Globales")

st.markdown("""
El análisis evidencia que la conversación pública sobre Inteligencia Artificial se encuentra dominada por una retórica de **productividad, adaptación profesional y expansión de capacidades**. 

A diferencia de la especulación recurrente sobre el reemplazo masivo, los datos semánticos demuestran que los usuarios perciben la IA como una transición técnica obligatoria: la adopción de herramientas (especialmente del ecosistema de lenguajes como Python y modelos generativos como GPT) es vista como una ventaja competitiva clave.

Sin embargo, persisten focos críticos consolidados que exigen resolver los debates sobre **sesgos (`bias`) algorítmicos y riesgos (`risk`) éticos** antes de alcanzar una asimilación corporativa y social avanzada.
""")

st.markdown("---")

# FOOTER

st.markdown(
    """
<p style='text-align:center; color:gray;'>
Trabajo Final | Modelos Predictivos III | Helena Molina | 2026
</p>
""",
    unsafe_allow_html=True
)