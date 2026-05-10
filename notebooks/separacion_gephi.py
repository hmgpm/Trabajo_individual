import pandas as pd
import itertools
from collections import Counter
import os

# Configuración de rutas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(base_path, "data", "clean", "data_unificado.csv")
output_dir = os.path.join(base_path, "data", "clean")

def generar_grafo_unificado(df_input, min_weight=15, top_n_edges=2000):
    """
    Genera un CSV de aristas para Gephi sin sesgos y con limpieza mejorada.
    """
    stop_words = {
        'este', 'esta', 'estos', 'estas', 'para', 'como', 'pero', 'todo', 
        'esto', 'cuando', 'hacer', 'tiene', 'sobre', 'tambien', 'solo', 
        'porque', 'desde', 'hasta', 'todos', 'bien', 'nada', 'algo', 
        'aqui', 'esos', 'esas', 'entre', 'donde', 'hacia', 'muy', 'mismo',
        'poco', 'antes', 'despues', 'otro', 'otra', 'otros', 'otras', 'ia', 'ai',
        'pueden', 'puede', 'seria', 'sido', 'toda', 'parte', 'cada', 'claro'
    }
    
    conexiones = []
    df_filtered = df_input.dropna(subset=['clean_text']).copy()
    
    print(f"Analizando {len(df_filtered)} textos para el grafo...")

    for _, fila in df_filtered.iterrows():
        sentimiento = fila['sentiment_label']
        palabras = [w for w in str(fila['clean_text']).split() if len(w) > 3 and w not in stop_words]
        
        # Mantenemos el orden original y evitamos el sesgo alfabético al tomar la muestra
        palabras_unicas = list(dict.fromkeys(palabras))[:12]
        
        if len(palabras_unicas) < 2:
            continue
            
        # Generamos combinaciones de palabras (aristas)
        for p1, p2 in itertools.combinations(palabras_unicas, 2):
            # Normalizamos el orden para que (A,B) sea igual a (B,A)
            nodos = sorted([p1, p2])
            conexiones.append((nodos[0], nodos[1], sentimiento))
    
    # Contamos la frecuencia de cada arista
    contador = Counter(conexiones)
    
    # Filtramos por peso
    df_gephi = pd.DataFrame([
        {'Source': k[0], 'Target': k[1], 'Sentiment_Type': k[2], 'Weight': v} 
        for k, v in contador.items() if v >= min_weight
    ])
    
    if df_gephi.empty:
        print(f"No hay conexiones con peso >= {min_weight}.")
        return

    # Ordenar por importancia y limitar para evitar saturación en Gephi
    df_gephi = df_gephi.sort_values(by='Weight', ascending=False).head(top_n_edges)
    
    save_path = os.path.join(output_dir, "gephi_global_con_sentimiento.csv")
    df_gephi.to_csv(save_path, index=False)
    
    print(f"--- Grafo Generado ---")
    print(f"Aristas: {len(df_gephi)} | Archivo: {save_path}")

if __name__ == "__main__":
    try:
        if os.path.exists(input_path):
            df = pd.read_csv(input_path)
            generar_grafo_unificado(df)
        else:
            print(f"Error: No se encuentra {input_path}")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")