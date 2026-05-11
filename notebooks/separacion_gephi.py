"""
separacion_gephi.py - Construcción de redes radiales por pilar conversacional
======================================================================
Genera un grafo radial por cada pilar temático (empleo, automatización, ética, etc.)
donde el nodo central es el pilar y los nodos satélite son palabras clave
con aristas coloreadas por sentimiento promedio.
"""

import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from pathlib import Path
import os

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = DATA_DIR
IMAGES_DIR = BASE_DIR / "data" / "imagenes"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEFINICIÓN DE PILARES CON PALABRAS CLAVE (como en el EDA de Ryanair)
# ============================================================================
SEMILLAS = {
    'empleo_automatizacion': {
        'keywords': ['job', 'employment', 'career', 'hiring', 'worker', 'automation',
                     'replace', 'replacement', 'layoff', 'workplace', 'futureofwork',
                     'robot', 'machine', 'automat'],
        'nombre_legible': 'Empleo y Automatización'
    },
    'machine_learning_datascience': {
        'keywords': ['machinelearning', 'datascience', 'python', 'tensorflow',
                     'deeplearning', 'algorithm', 'model', 'analysis', 'coding',
                     'developer', 'software', 'gpt', 'openai', 'neural'],
        'nombre_legible': 'Machine Learning y Data Science'
    },
    'educacion_habilidades': {
        'keywords': ['learning', 'student', 'education', 'training', 'skill',
                     'course', 'bootcamp', 'certificate', 'university', 'learn',
                     'teach', 'study'],
        'nombre_legible': 'Educación y Habilidades'
    },
    'negocio_productividad': {
        'keywords': ['business', 'productivity', 'efficiency', 'marketing',
                     'company', 'team', 'workflow', 'startup', 'enterprise',
                     'profit', 'strategy'],
        'nombre_legible': 'Negocio y Productividad'
    },
    'etica_sociedad': {
        'keywords': ['ethics', 'bias', 'privacy', 'society', 'human',
                     'regulation', 'security', 'trust', 'risk', 'danger',
                     'fairness', 'transparency'],
        'nombre_legible': 'Ética y Sociedad'
    },
    'oportunidades': {
        'keywords': ['opportunity', 'opportunities', 'benefit', 'advantage',
                     'potential', 'innovation', 'growth', 'improve', 'future',
                     'new', 'better'],
        'nombre_legible': 'Oportunidades'
    }
}

# Stopwords adicionales para limpiar las palabras satélite
STOPWORDS_RADIAL = {
    'ai', 'ia', 'artificial', 'intelligence', 'use', 'make', 'get', 'would', 'could',
    'like', 'just', 'one', 'also', 'really', 'even', 'way', 'see', 'say', 'need',
    'well', 'going', 'want', 'time', 'much', 'thing', 'people', 'think', 'know'
}

def construir_red_radial(df, pilar_id, config, top_n_palabras=200, min_frecuencia=3, max_palabras_por_pilar=20):
    """
    Construye un grafo radial para un pilar específico.
    
    Args:
        df: DataFrame con textos limpios y sentimiento.
        pilar_id: Identificador del pilar (ej. 'empleo_automatizacion').
        config: Diccionario con 'keywords' y 'nombre_legible'.
        top_n_palabras: Número de palabras más frecuentes a considerar.
        min_frecuencia: Frecuencia mínima de co-ocurrencia para crear arista.
        max_palabras_por_pilar: Número máximo de palabras satélite a incluir.
    
    Returns:
        networkx.Graph: Grafo radial con el pilar como nodo central.
    """
    # 1. Filtrar textos que contengan al menos una keyword del pilar
    pattern = '|'.join(config['keywords'])
    df_pilar = df[df['texto_limpio'].str.contains(pattern, case=False, na=False)].copy()
    
    if len(df_pilar) < 50:
        print(f"  ⚠️ Pilar '{pilar_id}' con solo {len(df_pilar)} documentos. No se genera grafo.")
        return nx.Graph()
    
    # 2. Obtener palabras más frecuentes en todo el dataset (para filtrar ruido)
    todas_palabras = ' '.join(df['texto_sin_stopwords'].fillna('')).split()
    counter_global = Counter(todas_palabras)
    palabras_populares = {w for w, _ in counter_global.most_common(top_n_palabras)}
    
    # 3. Diccionario para acumular sentimientos por par (pilar, palabra)
    conexiones = defaultdict(list)
    
    # 4. Recorrer los textos del pilar para llenar conexiones
    for _, row in df_pilar.iterrows():
        tokens = set(row['texto_sin_stopwords'].split())
        sentimiento = row['vader_compound']
        
        for palabra in tokens:
            # La palabra debe ser popular o estar en las keywords del pilar
            if palabra in palabras_populares or palabra in config['keywords']:
                if len(palabra) > 2 and palabra not in STOPWORDS_RADIAL:
                    # Evitar que la palabra sea igual al nombre del pilar
                    if palabra != pilar_id:
                        conexiones[(pilar_id, palabra)].append(sentimiento)
    
    # 5. Convertir conexiones en lista de tuplas (palabra, frecuencia, lista_sent)
    lista_conexiones = []
    for (_, palabra), lista_sent in conexiones.items():
        if len(lista_sent) >= min_frecuencia:
            lista_conexiones.append((palabra, len(lista_sent), lista_sent))
    
    # 6. Ordenar por frecuencia descendente y quedarse con las primeras max_palabras_por_pilar
    lista_conexiones.sort(key=lambda x: x[1], reverse=True)
    conexiones_top = lista_conexiones[:max_palabras_por_pilar]
    
    # 7. Construir el grafo radial
    G = nx.Graph()
    nombre_central = config['nombre_legible']
    G.add_node(nombre_central, tipo='central', label=nombre_central)
    
    for palabra, freq, lista_sent in conexiones_top:
        sent_promedio = sum(lista_sent) / len(lista_sent)
        # Clasificar sentimiento para colorear la arista
        if sent_promedio >= 0.05:
            sent_label = 'positivo'
        elif sent_promedio <= -0.05:
            sent_label = 'negativo'
        else:
            sent_label = 'neutro'
        
        # Añadir nodo palabra
        if not G.has_node(palabra):
            G.add_node(palabra, tipo='palabra', label=palabra)
        # Añadir arista
        G.add_edge(nombre_central, palabra,
                   weight=freq,
                   sentiment_score=sent_promedio,
                   sentiment_label=sent_label)
    
    # 8. Eliminar nodos aislados (por si acaso)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # 9. Calcular grado ponderado para los nodos (útil para tamaño en Gephi)
    if G.number_of_nodes() > 0:
        weighted_deg = dict(G.degree(weight='weight'))
        nx.set_node_attributes(G, weighted_deg, 'weighted_degree')
    
    return G

def main():
    print("🚀 Construyendo redes radiales por pilar conversacional...")
    
    # Cargar datos unificados
    input_file = DATA_DIR / "datos_unificados.csv"
    if not input_file.exists():
        print(f"❌ Error: {input_file} no encontrado. Ejecuta main.py primero.")
        return
    
    df = pd.read_csv(input_file)
    
    # Verificar columnas necesarias
    if 'texto_sin_stopwords' not in df.columns:
        print("❌ Columna 'texto_sin_stopwords' no encontrada. Asegúrate de ejecutar main.py con lematización.")
        return
    
    # Construir un grafo por cada pilar
    for pilar_id, config in SEMILLAS.items():
        print(f"\n--- Procesando pilar: {pilar_id} ---")
        G = construir_red_radial(df, pilar_id, config, top_n_palabras=200, min_frecuencia=3)
        
        if G.number_of_nodes() > 1:
            # Exportar a GEXF para Gephi
            out_file = OUTPUT_DIR / f"gephi_{pilar_id}.gexf"
            nx.write_gexf(G, out_file)
            print(f"  ✅ Exportado: {out_file} | Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
        else:
            print(f"  ⚠️ Grafo vacío para '{pilar_id}'")
    
    # También generar un grafo global con todos los pilares como centros? Opcional.
    # Por ahora solo generamos los pilares individuales.
    print("\n✨ Proceso completado. Archivos .gexf listos para importar en Gephi.")

if __name__ == "__main__":
    main()