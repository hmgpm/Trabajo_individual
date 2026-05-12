#  Análisis de Sentimiento y Opinión Pública: IA y Empleo

**Trabajo Individual - Modelos Predictivos III - Helena Molina**

Análisis exhaustivo de la percepción pública sobre Inteligencia Artificial y su impacto en el empleo, utilizando datos de Twitter y Reddit.

---

##  Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Tecnologías](#-tecnologías)

---

##  Descripción

Este proyecto aplica técnicas avanzadas de **Procesamiento de Lenguaje Natural (NLP)** y **Análisis de Redes Sociales** para:

1. **Analizar el sentimiento** de miles de publicaciones sobre IA y empleo
2. **Identificar aspectos clave** mediante ABSA (Aspect-Based Sentiment Analysis)
3. **Visualizar redes semánticas** de conceptos relacionados
4. **Comparar plataformas** (Twitter vs Reddit) en términos de polarización y opinión

###  Objetivos

- Cuantificar la percepción pública sobre IA en el contexto laboral
- Identificar temas de preocupación y oportunidad
- Detectar comunidades de opinión y actores influyentes
- Proporcionar insights accionables sobre el debate público

---

## 📁 Estructura del Proyecto

Trabajo_individual/
│
├── data/
│ ├── raw/ # Datos originales (no versionados por tamaño)
│ ├── clean/ # Datos procesados (CSV, ignorados por git)
│ ├── gephi/ # Archivos de Gephi (versionados)
│ │ └── Proyecto_IA.gephi # Proyecto completo de Gephi
│ └── imagenes/
│ │  ├── gephi_global.png
│ │  ├── gephi_empleo_automatizacion.png
│ │  ├── gephi_etica_sociedad.png
│ │  ├── gephi_educacion_habilidades.png
│ │  ├── gephi_machine_learning_datascience.png
│ │  ├── gephi_negocio_productividad.png
│ │  └── gephi_oportunidades.png
│
├── src/
│ └── main.py # Pipeline principal (genera datos, gráficos ABSA)
│ 
├── notebooks/
│ ├── dashboard.py # Dashboard interactivo (con ABSA)
│ └── separacion_gephi.py # Genera archivos .gexf para Gephi
│
├──  .gitignore
├── README.md
└── requirements.txt # Dependencias
---


---

##  Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/hmgpm/Trabajo_individual.git
cd Trabajo_individual

### 2. Crear Entorno Virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Recursos de NLTK

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

##  Uso

### 1. Ejecutar el Análisis Completo

```bash
python src/main.py
```

Este script:

    Carga y limpia los datos de Twitter y Reddit

    Calcula sentimiento con VADER

    Detecta 10 aspectos (empleo, automatización, ética, productividad, etc.)

    Genera visualizaciones estáticas, incluyendo dos gráficos de ABSA:

        sentimiento_por_aspecto.png (porcentajes apilados)

        absa_por_aspecto.png (conteos apilados)

    Exporta red de co-ocurrencias para Gephi (gephi_global_con_sentimiento.csv)

    Si BERT está disponible, realiza topic modelling y guarda tópicos precalculados.


**Tiempo estimado**: 5-10 minutos

### 2. Generar grafos radiales por pilar

```bash
python notebooks/separacion_gephi.py
```

Crea archivos .gexf para cada pilar temático. Luego impórtalos en Gephi y exporta como PNG con los nombres esperados (ver sección de Grafos). Los PNG ya incluidos en data/gephi/ se muestran en el dashboard.

### 3. Lanzar el dashboard interactivo

```bash
streamlit run notebooks/dashboard.py
```

Abre http://localhost:8501 en tu navegador.

El dashboard incluye:

    KPIs dinámicos (publicaciones, sentimiento medio, polarización, % positivo)

    Distribución de sentimiento (barras + histograma VADER)

    Comparativa Twitter vs Reddit

    Análisis por Aspectos (ABSA) con gráficos interactivos de barras apiladas (conteos y porcentajes)

    Grafos semánticos de Gephi con insights personalizados

    Topic modelling (palabras clave e insights automáticos)

    Conclusiones globales
---

##  Metodología

### Fase 1: Recopilación y Preprocesamiento

- **Fuentes**: Twitter (5,000 tweets) + Reddit (posts completos)
-  
- **Limpieza de texto**: eliminación de URLs, menciones, emojis, caracteres especiales; lematización y eliminación de stopwords.


### Fase 2: Análisis de Sentimiento

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Especializado en textos de redes sociales
- Score compuesto: -1 (muy negativo) a +1 (muy positivo)
- Clasificación: Positivo (>0.05), Negativo (<-0.05), Neutro

#### ABSA (Aspect-Based Sentiment Analysis)
10 aspectos predefinidos mediante coincidencia de keywords. Los gráficos generados (estáticos e interactivos) permiten explorar la relación entre cada aspecto y el sentimiento.

Aspectos detectados:
- **Salarios**
- **Miedo**
- **Reemplazo**
- **Oportunidades**
- **Empleo**:
- **Automatización**
- **Creatividad**
- **Productividad**
- **Ética**
- **Educación**

### Fase 3: Análisis de Redes Sociales

Grafos de co-ocurrencia de palabras (global) y radiales por pilar. Las aristas se colorean según el sentimiento promedio.

- **Tipo de red**: Co-ocurrencia de palabras
- **Pilar Central**: Pilar de la conversación establecido.
- **Nodos**: Conceptos clave (palabras frecuentes)
- **Aristas**: Palabras que aparecen juntas
- **Peso**: Frecuencia de co-ocurrencia
- **Color**: Sentimiento asociado

**Métricas calculadas**:
- Centralidad de grado
- Betweenness centrality
- Closeness centrality
- Modularidad (comunidades)

Realizado también un Topic modelling: embeddings BERT + KMeans (si está disponible) o TF-IDF + KMeans como respaldo.

### Fase 4: Visualización

- **Dashboard Streamlit**: Exploración interactiva
- **Gephi**: Visualización de red semántica
- **Gráficos estáticos**: distribución de los comentarios simplificada

---

##  Resultados

### Hallazgos Principales

1. **Sentimiento General**: Calculado sobre 30,000+ comentarios
2. **Polarización**: Medida mediante desviación estándar
3. **Diferencias entre Plataformas**: Twitter vs Reddit
4. **Aspectos Más Discutidos**: Ranking de temas
5. **Comunidades Detectadas**: Grupos de opinión

### Archivos Generados

- data/clean/datos_unificados.csv

- data/clean/twitter_procesado.csv

- data/clean/reddit_procesado.csv

- data/clean/gephi_global_con_sentimiento.csv

### Grafos semánticos incluidos en el repositorio (versionados)

Los siguientes PNG se encuentran en data/imagenes/ y se muestran en el dashboard:

- Visión Global - gephi_global.png
- Empleo y Automatización - gephi_empleo_automatizacion.png
- Ética y Sociedad - gephi_etica_sociedad.png
- Machine Learning & Data Science - gephi_machine_learning_datascience.png
- Educación y Habilidades - gephi_educacion_habilidades.png
- Negocio y Productividad - gephi_negocio_productividad.png
- Oportunidades - gephi_oportunidades.png	
---

## 🛠️ Tecnologías

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.9+ |
| **NLP** | VADER, NLTK, sentence-transformers (BERT) |
| **Data Processing** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn, Plotly, Streamlit |
| **Dashboard** | Streamlit |
| **Network Analysis** | NetworkX, Gephi |
| **Topic Modelling** | scikit-learn, BERT, KMeans |




---

##  Autor

**Helena Molina**  
Modelos Predictivos III : Opinión Pública y Sentimiento


---

## Referencias
Dataset Tweets: https://www.kaggle.com/datasets/hossamelshabory97/ai-and-data-jobs-tweets
Dataset Reddit: https://www.kaggle.com/datasets/adrianaajaafar/ai-and-job-security-reddit-discussion 
---
**Última actualización**: Mayo 2026