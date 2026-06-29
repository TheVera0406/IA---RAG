"""
ETAPA C - BÚSQUEDA DE CONTEXTO EN CHROMADB

Este programa:
1. Carga el modelo de embeddings usado para procesar los documentos.
2. Abre la colección almacenada en ChromaDB.
3. Recibe una pregunta del usuario.
4. Convierte la pregunta en un embedding.
5. Recupera los candidatos, filtra por distancia y conserva el top de contexto.
6. Muestra los resultados y construye el contexto final.

Todavía NO utiliza la API key ni llama a OpenAI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Importamos la configuración centralizada
from config import (
    BASE_VECTORIAL_DIR,
    NOMBRE_COLECCION,
    MODELO_EMBEDDINGS,
    TOP_K_CANDIDATOS,
    MAX_CHUNKS_CONTEXTO,
    UMBRAL_DISTANCIA
)


# =========================================================
# BLOQUE 1: CARGAR MODELO Y CHROMADB
# =========================================================

def cargar_recursos():
    """Carga el modelo de embeddings y la colección de ChromaDB."""

    if not BASE_VECTORIAL_DIR.exists():
        raise FileNotFoundError(
            f"No existe la base vectorial: {BASE_VECTORIAL_DIR}"
        )

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    cliente = chromadb.PersistentClient(path=str(BASE_VECTORIAL_DIR))
    coleccion = cliente.get_collection(name=NOMBRE_COLECCION)

    if coleccion.count() == 0:
        raise ValueError("La colección existe, pero no contiene registros.")

    return modelo, coleccion


# =========================================================
# BLOQUE 2: CREAR EMBEDDING DE LA PREGUNTA
# =========================================================

def crear_embedding_pregunta(pregunta: str, modelo) -> list[list[float]]:
    """Convierte la pregunta en un vector normalizado."""
    
    # CORRECCIÓN CRÍTICA: E5 requiere el prefijo 'query: ' para consultas.
    texto_consulta = f"query: {pregunta}"

    return modelo.encode(
        [texto_consulta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


# =========================================================
# BLOQUE 3: BUSCAR CHUNKS EN CHROMADB
# =========================================================

def buscar_chunks(pregunta: str, modelo, coleccion, top_k: int = TOP_K_CANDIDATOS) -> list[dict]:
    """Busca candidatos, filtra por distancia y devuelve el contexto definitivo."""

    embedding = crear_embedding_pregunta(pregunta, modelo)

    resultados = coleccion.query(
        query_embeddings=embedding,
        n_results=min(top_k, coleccion.count()),
        include=["documents", "metadatas", "distances"],
    )

    encontrados = []

    datos = zip(
        resultados["ids"][0],
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    )

    # 1. Aplicamos el umbral (si está definido)
    for posicion, (id_chunk, texto, metadata, distancia) in enumerate(datos, start=1):
        if UMBRAL_DISTANCIA is None or float(distancia) <= UMBRAL_DISTANCIA:
            encontrados.append({
                "posicion": len(encontrados) + 1,
                "id": id_chunk,
                "texto": texto,
                "metadata": metadata or {},
                "distancia": float(distancia),
            })
            
    # 2. Control de contexto vacío
    if not encontrados:
        raise ValueError("No se encontraron fragmentos suficientemente relevantes (todos superaron el umbral de distancia).")

    # 3. Limitamos a los mejores fragmentos según MAX_CHUNKS_CONTEXTO
    return encontrados[:MAX_CHUNKS_CONTEXTO]


# =========================================================
# BLOQUE 4: MOSTRAR LOS RESULTADOS
# =========================================================

def mostrar_resultados(resultados: list[dict]) -> None:
    """Muestra los chunks recuperados junto con sus fuentes."""

    print("\n" + "=" * 70)
    print("CHUNKS RECUPERADOS Y ACEPTADOS")
    print("=" * 70)

    for resultado in resultados:
        metadata = resultado["metadata"]

        print(f"\nRESULTADO {resultado['posicion']}")
        print(f"ID        : {resultado['id']}")
        print(f"Documento : {metadata.get('archivo', 'Sin información')}")
        print(f"Página    : {metadata.get('pagina', 'Sin información')}")
        print(f"Chunk pág.: {metadata.get('chunk_en_pagina', 'Sin información')}")
        print(f"Distancia : {resultado['distancia']:.6f}")
        print(f"Texto     : {resultado['texto']}")


# =========================================================
# BLOQUE 5: CONSTRUIR EL CONTEXTO
# =========================================================

def construir_contexto(resultados: list[dict]) -> str:
    """Une los chunks recuperados en un solo contexto con fuentes explícitas."""

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
# BLOQUE 6: PROGRAMA PRINCIPAL
# =========================================================

def main() -> int:
    """Permite realizar búsquedas hasta que el usuario escriba salir."""

    print("=" * 70)
    print("ETAPA C - BÚSQUEDA DE CONTEXTO")
    print("=" * 70)

    try:
        print(f"Cargando modelo: {MODELO_EMBEDDINGS}")

        modelo, coleccion = cargar_recursos()

        print(f"Colección cargada       : {NOMBRE_COLECCION}")
        print(f"Registros disponibles   : {coleccion.count()}")
        print(f"Buscando candidatos     : {TOP_K_CANDIDATOS}")
        print(f"Chunks a conservar      : {MAX_CHUNKS_CONTEXTO}")
        print(f"Umbral de distancia     : {UMBRAL_DISTANCIA}")

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    while True:
        pregunta = input("\nEscribe una pregunta o 'salir': ").strip()

        if pregunta.lower() == "salir":
            print("Programa finalizado.")
            break

        if not pregunta:
            print("[AVISO] La pregunta no puede estar vacía.")
            continue

        try:
            resultados = buscar_chunks(pregunta, modelo, coleccion)

            mostrar_resultados(resultados)

            contexto = construir_contexto(resultados)

            print("\n" + "=" * 70)
            print("CONTEXTO FINAL")
            print("=" * 70)
            print(contexto)

        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")

    return 0


# =========================================================
# BLOQUE 7: PUNTO DE ENTRADA
# =========================================================

if __name__ == "__main__":
    sys.exit(main())