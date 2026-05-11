"""

SEPARACION_GEPHI.PY

CORRECCIÓN IMPORTANTE:

- Todas las aristas ahora tienen sentimiento medible.

- Todos los nodos provienen exclusivamente de los datasets originales.

- Se eliminan conexiones sin evidencia textual real.

- Cada edge utiliza el promedio REAL de vader_compound de las publicaciones donde coocurren ambos términos.

- No se generan nodos artificiales ni inferidos.

 - VIP Node Network Generator (Dashboard Optimized)

---------------------------------------------------------------------

Estrategia Top-Down para reducir el grafo a ~50 nodos.

1. Identifica los N nodos más frecuentes globalmente.

2. Construye la red EXCLUSIVAMENTE entre estos nodos VIP.

3. Aplica PMI y Top-N Edges para limpiar las conexiones entre ellos.

"""

import pandas as pd

import networkx as nx

import itertools

import math

from ast import literal_eval

from pathlib import Path

from collections import Counter, defaultdict

# Configuración de rutas

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "clean"

OUTPUT_DIR = DATA_DIR

# 1. Custom Stopwords (Crucial para que los 50 nodos no sean basura)

CUSTOM_STOPWORDS = {

    'ai', 'artificial', 'intelligence', 'chatgpt', 'data', 'elbasheer', 

    'people', 'thing', 'use', 'make', 'granola', 'think', 'know', 'like', 

    'one', 'would', 'could', 'get', 'time', 'much', 'really', 'even',

    'way', 'also', 'want', 'see', 'say', 'need', 'well', 'going', 'aiml', 'identified', 'digested', 'ramsey', 'elbasheer', 

    'via', 'granola', 'news', 'feed', 'original', 'source'

}

# Quitamos palabras genéricas y algunas que no aportan información como granola

def clean_and_ensure_tokens(df):

    """Asegura la existencia de tokens y filtra stopwords customizadas."""

    if 'tokens' in df.columns:

        if len(df) > 0 and isinstance(df['tokens'].iloc[0], str):

            df['tokens'] = df['tokens'].apply(literal_eval)

    else:

        text_column = next((c for c in ['texto_sin_stopwords', 'texto_limpio', 'clean_text', 'texto_original'] if c in df.columns), None)

        if text_column is None:

            raise ValueError("No se encontró columna de texto válida.")

        df['tokens'] = df[text_column].fillna('').astype(str).str.split()

    

    # Filtrar palabras genéricas y de un solo caracter

    df['tokens'] = df['tokens'].apply(

        lambda x: [w.lower() for w in x if w.lower() not in CUSTOM_STOPWORDS and len(w) > 2]

    )

    return df

def calculate_ppmi(cooc_freq, freq_a, freq_b, total_pairs):

    """Calcula el Pointwise Mutual Information Positivo."""

    prob_cooc = cooc_freq / total_pairs

    prob_a = freq_a / total_pairs

    prob_b = freq_b / total_pairs

    pmi = math.log2(prob_cooc / (prob_a * prob_b))

    return max(0, pmi)

def detectar_pilares_conversacionales(df):

    """

    Divide automáticamente el dataset en pilares temáticos.

    """

    pilares = {

        'empleo_automatizacion': [

            'job', 'jobs', 'employment', 'career', 'hiring', 'worker',

            'automation', 'replace', 'replacement', 'layoff', 'salary',

            'workplace', 'futureofwork'

        ],

        'machine_learning_datascience': [

            'machinelearning', 'datascience', 'python', 'tensorflow',

            'deeplearning', 'algorithm', 'model', 'analysis', 'coding',

            'developer', 'software'

        ],

        'educacion_habilidades': [

            'learning', 'student', 'education', 'training', 'skill',

            'course', 'udemy', 'bootcamp', 'certificate', 'university'

        ],

        'negocio_productividad': [

            'business', 'productivity', 'efficiency', 'marketing',

            'company', 'team', 'workflow', 'startup', 'enterprise'

        ],

        'etica_sociedad': [

            'ethics', 'bias', 'privacy', 'society', 'human',

            'regulation', 'security', 'trust', 'risk', 'danger'

        ]

    }

    datasets_pilares = {}

    for nombre_pilar, keywords in pilares.items():

        pattern = '|'.join(keywords)

        df_pilar = df[

            df['texto_limpio'].str.contains(pattern, case=False, na=False)

        ].copy()

        datasets_pilares[nombre_pilar] = df_pilar

    return datasets_pilares

def build_semantic_network(df, max_nodes=40, max_edges_per_node=4, min_cooc=3):

    """Construye un grafo cerrado limitando estrictamente el número máximo de nodos."""

    G = nx.Graph()

    df = clean_and_ensure_tokens(df.copy())

    

    print(f"  > Analizando {len(df)} documentos para extraer el Top {max_nodes}...")

    

    # PASO 1: Calcular frecuencias globales por documento (para evitar que un texto largo sesgue)

    global_counts = Counter()

    for tokens in df['tokens']:

        global_counts.update(set(tokens))

        

    if not global_counts:

        return G

        

    # PASO 2: Extraer los VIP Nodes (Top N)

    vip_nodes = {term for term, freq in global_counts.most_common(max_nodes)}

    print(f"  > Nodos VIP seleccionados (ejemplo): {list(vip_nodes)[:5]}...")

    

    # PASO 3: Conteo de co-ocurrencias SOLO entre nodos VIP

    edge_counts = Counter()

    edge_sentiments = defaultdict(list)

    total_pairs = 0

    

    for _, row in df.iterrows():

        tokens = row['tokens']

        sentimiento = row['vader_compound'] if 'vader_compound' in row else 0

        # Filtramos los tokens del documento para que solo queden los VIP

        filtered_tokens = sorted([t for t in set(tokens) if t in vip_nodes])

        

        if len(filtered_tokens) >= 2:

            pairs = list(itertools.combinations(filtered_tokens, 2))

            edge_counts.update(pairs)

            total_pairs += len(pairs)

            for pair in pairs:

                edge_sentiments[pair].append(sentimiento)

            

    if total_pairs == 0:

        return G

    # PASO 4: Filtrado PMI entre los VIP

    scored_edges = []

    for (node_a, node_b), cooc in edge_counts.items():

        if cooc >= min_cooc:

            pmi = calculate_ppmi(cooc, global_counts[node_a], global_counts[node_b], total_pairs)

            sentimiento_medio = sum(edge_sentiments[(node_a, node_b)]) / len(edge_sentiments[(node_a, node_b)])

            if sentimiento_medio >= 0.05:

                sentimiento_label = 'positivo'

            elif sentimiento_medio <= -0.05:

                sentimiento_label = 'negativo'

            else:

                sentimiento_label = 'neutro'

            scored_edges.append((

                node_a,

                node_b,

                cooc,

                pmi,

                sentimiento_medio,

                sentimiento_label

            ))

                

    for u, v, cooc, pmi, sent_score, sent_label in scored_edges:

        G.add_edge(

            u,

            v,

            weight=cooc,

            pmi=pmi,

            sentiment_score=sent_score,

            sentiment_label=sent_label

        )

        

    # PASO 5: Poda estricta de Ego-Network (Para evitar que todos los 50 se conecten entre sí)

    edges_to_keep = set()

    for node in G.nodes():

        neighbors = sorted(G[node].items(), key=lambda x: x[1]['pmi'], reverse=True)

        top_neighbors = neighbors[:max_edges_per_node]

        for neighbor, _ in top_neighbors:

            u, v = sorted([node, neighbor])

            edges_to_keep.add((u, v))

            

    # Reconstruir el grafo definitivo

    G_final = nx.Graph()

    for u, v in edges_to_keep:

        G_final.add_edge(

            u,

            v,

            weight=G[u][v]['weight'],

            pmi=G[u][v]['pmi'],

            sentiment_score=G[u][v]['sentiment_score'],

            sentiment_label=G[u][v]['sentiment_label']

        )

    # Limpiar huérfanos

    G_final.remove_nodes_from(list(nx.isolates(G_final)))

    # Calcular Métricas

    if len(G_final) > 0:

        nx.set_node_attributes(G_final, dict(G_final.degree(weight='weight')), 'weighted_degree')

        nx.set_node_attributes(G_final, nx.betweenness_centrality(G_final), 'betweenness')

    return G_final

def export_network_files(G, filename_base):

    if len(G) == 0:

        print(f"  ⚠️ Grafo {filename_base} vacío. Saltando.")

        return

    gexf_path = OUTPUT_DIR / f"{filename_base}.gexf"

    nx.write_gexf(G, gexf_path)

    print(f"  ✅ {filename_base}: {G.number_of_nodes()} nodos | {G.number_of_edges()} aristas")

def main():

    print("🚀 Iniciando Construcción de Grafos VIP (Dashboard Mode)...")

    input_file = DATA_DIR / "datos_unificados.csv"

    

    if not input_file.exists():

        print(f"❌ Error: {input_file} no encontrado.")

        return

        

    df = pd.read_csv(input_file)

    # --- 1. Grafo Global ---

    print("\n--- 1. Grafo Global (Top 40) ---")

    G_global = build_semantic_network(

        df,

        max_nodes=40,

        max_edges_per_node=4,

        min_cooc=5

    )

    export_network_files(G_global, "gephi_global")

    # --- 2. Pilares Conversacionales ---

    print("\n--- 2. Generando Pilares Conversacionales ---")

    pilares = detectar_pilares_conversacionales(df)

    for nombre_pilar, df_pilar in pilares.items():

        print(f"\n--- Procesando Pilar: {nombre_pilar.upper()} ---")

        if not df_pilar.empty:

            G_pilar = build_semantic_network(

                df_pilar,

                max_nodes=40,

                max_edges_per_node=4,

                min_cooc=3

            )

            export_network_files(G_pilar, f"gephi_{nombre_pilar}")

    print("\n✨ PROCESO COMPLETADO. Listos para importar a Gephi.")

if __name__ == "__main__":

    main()