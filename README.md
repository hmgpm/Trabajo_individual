# sentimiento-individual
Trabajo individual opinión pública y sentimiento- IA
# AI Sentiment Analysis: Impact of AI on Jobs

## Descripción
Este proyecto analiza la polarización en redes sociales sobre el impacto de la inteligencia artificial en el empleo.

## Fuentes de datos
- Dataset de Kaggle: AI and Data Jobs Tweets (2023)
- Hilo de Twitter/X de Bernie Sanders

## Metodología
1. Extracción de datos (snscrape)
2. Preprocesamiento de texto
3. Análisis de sentimiento (TextBlob)
4. Análisis exploratorio (EDA)
5. Visualización en dashboard

## Cómo ejecutar

### 1. Instalar dependencias
pip install -r requirements.txt

### 2. Extraer datos
python src/scrape_twitter.py

### 3. Ejecutar análisis
python main.py

### 4. Lanzar dashboard
streamlit run dashboard/app.py