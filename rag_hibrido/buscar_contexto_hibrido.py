"""
ETAPA C - BÚSQUEDA HÍBRIDA DE CONTEXTO EN CHROMADB

versión MEJORADA del buscar_contexto.py original.

MEJORAS IMPLEMENTADAS:
Combina Embeddings (similitud semántica) + BM25 (relevancia léxica)
Recupera chunks más relevantes usando dos señales simultáneamente
Mantiene compatibilidad con toda la pipeline existente
Parámetros ajustables (pesos, top-k)

ETAPAS DEL RETRIEVAL HÍBRIDO:
1. Usuario escribe pregunta
2. Se convierte a embedding (semántica)
3. Se tokeniza y busca en BM25 (léxica)
4. Se combinan scores: 50% embedding + 50% BM25
5. Se devuelven los TOP-K chunks más relevantes

Sin OpenAI. Sin API key. 100% local.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import math

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# BLOQUE 1: CONFIGURACIÓN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

NOMBRE_COLECCION = "documentos_obesidad"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# PARÁMETROS DE CONTROL DEL RETRIEVAL HÍBRIDO
EMBEDDING_WEIGHT = 0.5  # 50% embeddings
BM25_WEIGHT = 0.5       # 50% BM25
# Ajustar estos valores para cambiar el equilibrio:
# - 0.7 / 0.3 → más semántico
# - 0.3 / 0.7 → más léxico


# =========================================================
# BLOQUE 2: IMPLEMENTACIÓN DE BM25
# =========================================================

class BM25:
    """Implementación simple de BM25 para retrieval léxico."""
    
    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        """
        Inicializa BM25 con los documentos.
        
        Args:
            documentos: Lista de textos a indexar
            k1: Parámetro de saturación de término (típicamente 1.5)
            b: Parámetro de normalización de longitud (típicamente 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documentos = documentos
        
        # Tokenizar documentos
        self.tokenized_docs = [self._tokenize(doc) for doc in documentos]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0
        
        # Calcular IDF
        self.idf = self._compute_idf()
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza y normaliza el texto."""
        text = text.lower()
        # Remover puntuación básica
        for char in ".,;:!?()[]{}\"'—–":
            text = text.replace(char, " ")
        # Dividir en palabras
        tokens = text.split()
        # Filtrar palabras muy cortas (stopwords básicos)
        return [t for t in tokens if len(t) > 2]
    
    def _compute_idf(self) -> dict[str, float]:
        """Calcula IDF (Inverse Document Frequency) para todos los términos."""
        idf = {}
        doc_count = len(self.tokenized_docs)
        
        # Contar documentos que contienen cada término
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                idf[term] = idf.get(term, 0) + 1
        
        # Convertir a IDF: log((N - df + 0.5) / (df + 0.5))
        for term in idf:
            idf[term] = math.log((doc_count - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
        
        return idf
    
    def score(self, query: str) -> list[float]:
        """
        Calcula BM25 score para todos los documentos.
        
        Returns:
            Lista de scores (uno por documento)
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            
            # Frecuencia de términos en el documento
            term_freq = Counter(doc_tokens)
            
            for term in query_tokens:
                if term not in term_freq:
                    continue
                
                freq = term_freq[term]
                idf = self.idf.get(term, 0)
                
                # Fórmula BM25
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                
                score += numerator / denominator
            
            scores.append(score)
        
        return scores


# =========================================================
# BLOQUE 3: CARGAR RECURSOS
# =========================================================

def cargar_recursos():
    """Carga el modelo de embeddings y la colección de ChromaDB."""

    if not BASE_VECTORIAL_DIR.exists():
        raise FileNotFoundError(
            f"No existe la base vectorial: {BASE_VECTORIAL_DIR}"
        )

    print(f"Cargando modelo: {MODELO_EMBEDDINGS}")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    
    cliente = chromadb.PersistentClient(path=str(BASE_VECTORIAL_DIR))
    coleccion = cliente.get_collection(name=NOMBRE_COLECCION)

    if coleccion.count() == 0:
        raise ValueError("La colección existe, pero no contiene registros.")

    return modelo, coleccion


# =========================================================
# BLOQUE 4: RETRIEVAL VECTORIAL (EMBEDDINGS)
# =========================================================

def crear_embedding_pregunta(pregunta: str, modelo) -> list[list[float]]:
    """Convierte la pregunta en un vector normalizado."""
    return modelo.encode(
        [pregunta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


def buscar_por_embeddings(pregunta: str, modelo, coleccion, top_k: int = TOP_K) -> dict:
    """
    Busca chunks por similitud semántica (embeddings).
    
    Returns:
        Diccionario con IDs y scores de similitud
    """
    embedding = crear_embedding_pregunta(pregunta, modelo)

    resultados = coleccion.query(
        query_embeddings=embedding,
        n_results=min(top_k, coleccion.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Convertir distancia a similitud (1 - distancia para cosine)
    scores_embedding = {}
    for id_chunk, distancia in zip(resultados["ids"][0], resultados["distances"][0]):
        # Distancia cosine va de 0 (igual) a 2 (opuesto)
        # Convertimos a similitud: 1 - (distancia / 2)
        similitud = 1 - (float(distancia) / 2)
        scores_embedding[id_chunk] = similitud

    return scores_embedding


# =========================================================
# BLOQUE 5: RETRIEVAL LÉXICO (BM25)
# =========================================================

def buscar_por_bm25(pregunta: str, coleccion, top_k: int = TOP_K) -> dict:
    """
    Busca chunks por relevancia de palabras clave (BM25).
    
    Returns:
        Diccionario con IDs y scores BM25
    """
    # Obtener todos los documentos de la colección
    todos = coleccion.get(include=["documents", "metadatas"])
    
    documentos = todos["documents"]
    ids = todos["ids"]
    
    # Crear índice BM25
    bm25 = BM25(documentos)
    
    # Calcular scores para todos los documentos
    scores_bm25_raw = bm25.score(pregunta)
    
    # Normalizar scores a [0, 1]
    max_score = max(scores_bm25_raw) if scores_bm25_raw else 1
    if max_score == 0:
        max_score = 1
    
    scores_bm25 = {
        id_chunk: score / max_score
        for id_chunk, score in zip(ids, scores_bm25_raw)
    }
    
    return scores_bm25


# =========================================================
# BLOQUE 6: RETRIEVAL HÍBRIDO (COMBINADO)
# =========================================================

def buscar_chunks_hibrido(
    pregunta: str,
    modelo,
    coleccion,
    top_k: int = TOP_K,
    embedding_weight: float = EMBEDDING_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[dict]:
    """
    Búsqueda HÍBRIDA que combina embeddings + BM25.
    
    Pasos:
    1. Busca por embeddings (semántica)
    2. Busca por BM25 (palabras clave)
    3. Combina scores: embedding_weight * emb_score + bm25_weight * bm25_score
    4. Retorna TOP-K documentos con mejor score híbrido
    """
    
    print(f"\n🔍 Búsqueda híbrida (E:{embedding_weight:.1%} + B:{bm25_weight:.1%})")
    
    # Paso 1: Retrieval por embeddings
    print("  Buscando por embeddings (semántica)...", end=" ", flush=True)
    scores_embedding = buscar_por_embeddings(pregunta, modelo, coleccion, top_k=coleccion.count())
    print(f"✓ ({len(scores_embedding)} resultados)")
    
    # Paso 2: Retrieval por BM25
    print("  Buscando por BM25 (palabras clave)...", end=" ", flush=True)
    scores_bm25 = buscar_por_bm25(pregunta, coleccion, top_k=coleccion.count())
    print(f"✓ ({len(scores_bm25)} resultados)")
    
    # Paso 3: Combinar scores
    print("   Combinando scores...", end=" ", flush=True)
    todos_ids = set(scores_embedding.keys()) | set(scores_bm25.keys())
    
    hybrid_scores = {}
    for id_chunk in todos_ids:
        emb_score = scores_embedding.get(id_chunk, 0.0)
        bm25_score = scores_bm25.get(id_chunk, 0.0)
        
        hybrid_score = (embedding_weight * emb_score + bm25_weight * bm25_score)
        hybrid_scores[id_chunk] = {
            "hybrid": hybrid_score,
            "embedding": emb_score,
            "bm25": bm25_score,
        }
    print("✓")
    
    # Paso 4: Ordenar y obtener TOP-K
    print(f"  Ordenando y recuperando TOP-{top_k}...", end=" ", flush=True)
    top_ids = sorted(
        hybrid_scores.keys(),
        key=lambda x: hybrid_scores[x]["hybrid"],
        reverse=True
    )[:top_k]
    
    # Obtener información detallada
    resultados = []
    
    datos_chromadb = coleccion.get(
        ids=top_ids,
        include=["documents", "metadatas"]
    )
    
    # Mapear IDs a índices
    id_a_indice = {id_chunk: idx for idx, id_chunk in enumerate(datos_chromadb["ids"])}
    
    for posicion, id_chunk in enumerate(top_ids, start=1):
        idx = id_a_indice[id_chunk]
        
        resultados.append({
            "posicion": posicion,
            "id": id_chunk,
            "texto": datos_chromadb["documents"][idx],
            "metadata": datos_chromadb["metadatas"][idx] or {},
            "score_hybrid": hybrid_scores[id_chunk]["hybrid"],
            "score_embedding": hybrid_scores[id_chunk]["embedding"],
            "score_bm25": hybrid_scores[id_chunk]["bm25"],
        })
    
    print("✓")
    return resultados


# =========================================================
# BLOQUE 7: MOSTRAR RESULTADOS
# =========================================================

def mostrar_resultados(resultados: list[dict]) -> None:
    """Muestra los chunks recuperados con sus scores detallados."""

    print("\n" + "=" * 80)
    print("CHUNKS RECUPERADOS (HÍBRIDO)")
    print("=" * 80)

    for resultado in resultados:
        metadata = resultado["metadata"]

        print(f"\n RESULTADO #{resultado['posicion']}")
        print(f"   ID              : {resultado['id']}")
        print(f"   Documento       : {metadata.get('archivo', 'Sin información')}")
        print(f"   Página          : {metadata.get('pagina', 'Sin información')}")
        print(f"   Chunk en página : {metadata.get('chunk_en_pagina', 'Sin información')}")
        print(f"\n    SCORES:")
        print(f"      Híbrido      : {resultado['score_hybrid']:.4f} ")
        print(f"      Embedding    : {resultado['score_embedding']:.4f} (semántica)")
        print(f"      BM25         : {resultado['score_bm25']:.4f} (léxica)")
        print(f"\n    Texto: {resultado['texto'][:150]}...")


# =========================================================
# BLOQUE 8: CONSTRUIR CONTEXTO
# =========================================================

def construir_contexto(resultados: list[dict]) -> str:
    """Une los chunks recuperados en un solo contexto con fuentes."""

    partes = []

    for resultado in resultados:
        metadata = resultado["metadata"]

        fuente = (
            f"[Fuente: {metadata.get('archivo', 'Desconocida')}, "
            f"página {metadata.get('pagina', 'desconocida')}]"
        )

        partes.append(f"{fuente}\n{resultado['texto']}")

    return "\n\n".join(partes)


# =========================================================
# BLOQUE 9: PROGRAMA PRINCIPAL
# =========================================================

def main() -> int:
    """Permite realizar búsquedas hasta que el usuario escriba salir."""

    print("=" * 80)
    print(" ETAPA C - BÚSQUEDA HÍBRIDA DE CONTEXTO")
    print("=" * 80)
    print("\n VERSIÓN MEJORADA CON RETRIEVAL HÍBRIDO")
    print(f"   • Embeddings: {EMBEDDING_WEIGHT:.0%}")
    print(f"   • BM25: {BM25_WEIGHT:.0%}")
    print(f"   • TOP-K: {TOP_K}")

    try:
        print(f"\nCargando modelo: {MODELO_EMBEDDINGS}")

        modelo, coleccion = cargar_recursos()

        print(f"✓ Colección cargada       : {NOMBRE_COLECCION}")
        print(f"✓ Registros disponibles   : {coleccion.count()}")
        print(f"✓ Modelo cargado          : {MODELO_EMBEDDINGS}")

    except Exception as error:
        print(f" [ERROR] {type(error).__name__}: {error}")
        return 1

    print("\n" + "=" * 80)
    print(" LISTO PARA BÚSQUEDAS")
    print("=" * 80)

    while True:
        pregunta = input("\n Escribe una pregunta o 'salir': ").strip()

        if pregunta.lower() == "salir":
            print(" Programa finalizado.")
            break

        if not pregunta:
            print(" La pregunta no puede estar vacía.")
            continue

        try:
            # Búsqueda híbrida
            resultados = buscar_chunks_hibrido(
                pregunta,
                modelo,
                coleccion,
                top_k=TOP_K,
                embedding_weight=EMBEDDING_WEIGHT,
                bm25_weight=BM25_WEIGHT,
            )

            # Mostrar resultados
            mostrar_resultados(resultados)

            # Construir contexto
            contexto = construir_contexto(resultados)

            print("\n" + "=" * 80)
            print(" CONTEXTO FINAL (para pasar al LLM)")
            print("=" * 80)
            print(contexto)

        except Exception as error:
            print(f" [ERROR] {type(error).__name__}: {error}")

    return 0


# =========================================================
# BLOQUE 10: PUNTO DE ENTRADA
# =========================================================

if __name__ == "__main__":
    sys.exit(main())
