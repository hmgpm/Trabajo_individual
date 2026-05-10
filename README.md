# 🤖 Análisis de Sentimiento y Opinión Pública: IA y Empleo

**Trabajo Individual - Modelos Predictivos III**

Análisis exhaustivo de la percepción pública sobre Inteligencia Artificial y su impacto en el empleo, utilizando datos de Twitter y Reddit.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Tecnologías](#-tecnologías)

---

## 📖 Descripción

Este proyecto aplica técnicas avanzadas de **Procesamiento de Lenguaje Natural (NLP)** y **Análisis de Redes Sociales** para:

1. **Analizar el sentimiento** de miles de publicaciones sobre IA y empleo
2. **Identificar aspectos clave** mediante ABSA (Aspect-Based Sentiment Analysis)
3. **Visualizar redes semánticas** de conceptos relacionados
4. **Comparar plataformas** (Twitter vs Reddit) en términos de polarización y opinión

### 🎯 Objetivos

- Cuantificar la percepción pública sobre IA en el contexto laboral
- Identificar temas de preocupación y oportunidad
- Detectar comunidades de opinión y actores influyentes
- Proporcionar insights accionables sobre el debate público

---

## 📁 Estructura del Proyecto
Trabajo_individual/
│
├── data/
│   ├── raw/                          # Datos originales
│   │   ├── Twitter_Final_data.csv
│   │   └── reddit_comments_combined.csv
│   │
│   └── clean/                        # Datos procesados
│       ├── data_unificado.csv
│       ├── gephi_global_con_sentimiento.csv
│       ├── absa_resultados.csv
│       └── *.png                     # Visualizaciones
│
├── src/
│   ├── main.py                       # Script principal de análisis
│   └── dashboard.py                  # Dashboard interactivo
│
├── notebooks/
│   └── 1_EDA_y_Preprocesamiento.ipynb
│
├── requirements.txt                  # Dependencias
├── README.md                         # Este archivo
├── INSTRUCCIONES_GEPHI.md           # Guía para análisis de red
└── .gitignore
---

## 🚀 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/trabajo-ia-empleo.git
cd trabajo-ia-empleo
```

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
nltk.download('vader_lexicon')
nltk.download('punkt')
```

---

## 💻 Uso

### 1. Ejecutar el Análisis Completo

```bash
python src/main.py
```

Este script:
- Carga y limpia los datos de Twitter y Reddit
- Calcula sentimiento con VADER
- Ejecuta ABSA para detectar aspectos
- Genera visualizaciones
- Crea archivo para Gephi

**Tiempo estimado**: 5-10 minutos

### 2. Visualizar Dashboard Interactivo

```bash
streamlit run src/dashboard.py
```

El dashboard se abrirá en tu navegador en `http://localhost:8501`

### 3. Análisis de Red en Gephi

Ver `INSTRUCCIONES_GEPHI.md` para el tutorial completo.

---

## 🔬 Metodología

### Fase 1: Recopilación y Preprocesamiento

- **Fuentes**: Twitter (5,000 tweets) + Reddit (posts completos)
- **Limpieza**: 
  - Eliminación de URLs, menciones, hashtags
  - Normalización de texto
  - Filtrado de spam y duplicados

### Fase 2: Análisis de Sentimiento

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Especializado en textos de redes sociales
- Score compuesto: -1 (muy negativo) a +1 (muy positivo)
- Clasificación: Positivo (>0.05), Negativo (<-0.05), Neutro

#### ABSA (Aspect-Based Sentiment Analysis)
Aspectos detectados:
- **Empleo**: trabajo, desempleo, carrera
- **Automatización**: robots, reemplazo, máquinas
- **Futuro**: predicciones, tendencias
- **Productividad**: eficiencia, mejoras
- **Ética**: sesgos, privacidad, responsabilidad
- **Economía**: salarios, crecimiento, costes

### Fase 3: Análisis de Redes Sociales

- **Tipo de red**: Co-ocurrencia de palabras
- **Nodos**: Conceptos clave (palabras frecuentes)
- **Aristas**: Palabras que aparecen juntas
- **Peso**: Frecuencia de co-ocurrencia
- **Color**: Sentimiento asociado

**Métricas calculadas**:
- Centralidad de grado
- Betweenness centrality
- Closeness centrality
- Modularidad (comunidades)

### Fase 4: Visualización

- **Dashboard Streamlit**: Exploración interactiva
- **Gephi**: Visualización de red semántica
- **Gráficos estáticos**: PNG de alta resolución

---

## 📊 Resultados

### Hallazgos Principales

_(Estos se generan automáticamente al ejecutar `main.py`)_

1. **Sentimiento General**: Calculado sobre 10,000+ comentarios
2. **Polarización**: Medida mediante desviación estándar
3. **Diferencias entre Plataformas**: Twitter vs Reddit
4. **Aspectos Más Discutidos**: Ranking de temas
5. **Comunidades Detectadas**: Grupos de opinión

### Archivos Generados

- `data_unificado.csv`: Dataset completo procesado
- `absa_resultados.csv`: Sentimiento por aspecto
- `gephi_global_con_sentimiento.csv`: Red para Gephi
- 9 visualizaciones PNG

---

## 🛠️ Tecnologías

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.9+ |
| **NLP** | VADER, NLTK |
| **Data Processing** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Network Analysis** | NetworkX, Gephi |
| **Notebooks** | Jupyter |




---

## 👨‍💻 Autor

**Helena Molina**  
Modelos Predictivos III : Opinión Pública y Sentimiento


---

**Última actualización**: Mayo 2026