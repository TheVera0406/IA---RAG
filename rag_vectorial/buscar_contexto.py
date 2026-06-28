
"""
ETAPA C - BÚSQUEDA DE CONTEXTO EN CHROMADB

Este programa:
1. Carga el modelo de embeddings usado para procesar los documentos.
2. Abre la colección almacenada en ChromaDB.
3. Recibe una pregunta del usuario.
4. Convierte la pregunta en un embedding.
5. Recupera los 5 chunks más cercanos.
6. Muestra los resultados y construye el contexto final.

Todavía NO utiliza la API key ni llama a OpenAI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# BLOQUE 1: CONFIGURACIÓN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

NOMBRE_COLECCION = "documentos_obesidad"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
TOP_K = 8
UMBRAL_DISTANCIA = 1.2 # Si la distancia es mayor a esto, lo descartamos por irrelevante


# =========================================================
# BLOQUE 2: CARGAR MODELO Y CHROMADB
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
# BLOQUE 3: CREAR EMBEDDING DE LA PREGUNTA
# =========================================================

def crear_embedding_pregunta(pregunta: str, modelo) -> list[list[float]]:
    """Convierte la pregunta en un vector normalizado."""

    return modelo.encode(
        [pregunta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


# =========================================================
# BLOQUE 4: BUSCAR CHUNKS EN CHROMADB
# =========================================================

def buscar_chunks(pregunta: str, modelo, coleccion, top_k: int = TOP_K) -> list[dict]:
    """Busca en ChromaDB los chunks más cercanos a la pregunta."""

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

    for posicion, (id_chunk, texto, metadata, distancia) in enumerate(datos, start=1):
    # Solo agregamos el chunk si cumple con el umbral de similitud
        if float(distancia) <= UMBRAL_DISTANCIA:
            encontrados.append({
                "posicion": len(encontrados) + 1,
                "id": id_chunk,
                "texto": texto,
                "metadata": metadata or {},
                "distancia": float(distancia),
            })

    return encontrados


# =========================================================
# BLOQUE 5: MOSTRAR LOS RESULTADOS
# =========================================================

def mostrar_resultados(resultados: list[dict]) -> None:
    """Muestra los chunks recuperados junto con sus fuentes."""

    print("\n" + "=" * 70)
    print("CHUNKS RECUPERADOS")
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
# BLOQUE 6: CONSTRUIR EL CONTEXTO
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
# BLOQUE 7: PROGRAMA PRINCIPAL
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
        print(f"Resultados por búsqueda : {TOP_K}")

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
# BLOQUE 8: PUNTO DE ENTRADA
# =========================================================

if __name__ == "__main__":
    sys.exit(main())
