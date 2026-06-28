"""
ETAPA B.4 - GUARDAR LOS CHUNKS Y EMBEDDINGS EN CHROMADB

Lee:
    salidas/chunks.jsonl
    salidas/embeddings.npy

Genera:
    base_vectorial/
    salidas/info_chromadb.json

Todavía NO usa la API key ni llama a OpenAI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm


# BLOQUE 1: CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent
SALIDAS_DIR = BASE_DIR / "salidas"
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

ARCHIVO_CHUNKS = SALIDAS_DIR / "chunks.jsonl"
ARCHIVO_EMBEDDINGS = SALIDAS_DIR / "embeddings.npy"
ARCHIVO_INFO = SALIDAS_DIR / "info_chromadb.json"

NOMBRE_COLECCION = "documentos_obesidad"
TAMANO_LOTE = 100


# BLOQUE 2: CARGAR Y VERIFICAR LOS DATOS

def cargar_chunks() -> list[dict]:
    """Carga todos los chunks desde el archivo JSONL."""
    chunks = []

    with ARCHIVO_CHUNKS.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea:
                continue

            try:
                chunks.append(json.loads(linea))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido en la línea {numero_linea}: {error}"
                ) from error

    return chunks


def cargar_embeddings() -> np.ndarray:
    """Carga la matriz NumPy creada en la etapa B.3."""
    return np.load(ARCHIVO_EMBEDDINGS)


def verificar_datos(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Comprueba que exista un embedding válido por cada chunk."""
    if not chunks:
        raise ValueError("No se encontró ningún chunk.")

    if embeddings.ndim != 2:
        raise ValueError("La matriz debe tener dos dimensiones.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Hay {len(chunks)} chunks y "
            f"{embeddings.shape[0]} embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Los embeddings contienen valores inválidos."
        )


# BLOQUE 3: CREAR LA COLECCIÓN

def crear_coleccion():
    """Crea una colección nueva dentro de ChromaDB."""
    BASE_VECTORIAL_DIR.mkdir(parents=True, exist_ok=True)

    cliente = chromadb.PersistentClient(
        path=str(BASE_VECTORIAL_DIR)
    )

    nombres = [
        item if isinstance(item, str) else item.name
        for item in cliente.list_collections()
    ]

    if NOMBRE_COLECCION in nombres:
        print(
            f"Eliminando colección anterior: "
            f"{NOMBRE_COLECCION}"
        )
        cliente.delete_collection(NOMBRE_COLECCION)

    return cliente.create_collection(
        name=NOMBRE_COLECCION,
        metadata={"hnsw:space": "cosine"},
    )


# BLOQUE 4: GUARDAR LOS DATOS EN CHROMADB

def guardar_en_chromadb(
    coleccion,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Guarda IDs, textos, embeddings y metadatos por lotes."""

    for inicio in tqdm(
        range(0, len(chunks), TAMANO_LOTE),
        desc="Guardando en ChromaDB",
    ):
        fin = min(inicio + TAMANO_LOTE, len(chunks))
        lote = chunks[inicio:fin]

        ids = [chunk["id"] for chunk in lote]
        documentos = [chunk["texto"] for chunk in lote]

        metadatos = [{
            "archivo": chunk["archivo"],
            "ruta_relativa": chunk["ruta_relativa"],
            "pagina": int(chunk["pagina"]),
            "chunk_en_pagina": int(chunk["chunk_en_pagina"]),
            "cantidad_caracteres": int(chunk["cantidad_caracteres"]),
        } for chunk in lote]

        coleccion.add(
            ids=ids,
            documents=documentos,
            embeddings=embeddings[inicio:fin].tolist(),
            metadatas=metadatos,
        )


# BLOQUE 5: GUARDAR INFORMACIÓN DE LA BASE

def guardar_info(coleccion, embeddings: np.ndarray) -> None:
    """Guarda un resumen de la colección creada."""

    info = {
        "coleccion": NOMBRE_COLECCION,
        "cantidad_registros": coleccion.count(),
        "dimension_embedding": int(embeddings.shape[1]),
        "metrica": "cosine",
        "tamano_lote": TAMANO_LOTE,
        "ruta_base_vectorial": str(BASE_VECTORIAL_DIR),
    }

    with ARCHIVO_INFO.open("w", encoding="utf-8") as archivo:
        json.dump(
            info,
            archivo,
            ensure_ascii=False,
            indent=4,
        )


# BLOQUE 6: PROGRAMA PRINCIPAL

def main() -> int:
    """Coordina la creación y verificación de ChromaDB."""

    print("=" * 70)
    print("ETAPA B.4 - GUARDAR DATOS EN CHROMADB")
    print("=" * 70)

    if not ARCHIVO_CHUNKS.exists():
        print(f"[ERROR] No existe: {ARCHIVO_CHUNKS}")
        return 1

    if not ARCHIVO_EMBEDDINGS.exists():
        print(f"[ERROR] No existe: {ARCHIVO_EMBEDDINGS}")
        return 1

    try:
        chunks = cargar_chunks()
        embeddings = cargar_embeddings()
        verificar_datos(chunks, embeddings)

        print(f"Chunks cargados        : {len(chunks)}")
        print(f"Embeddings cargados    : {embeddings.shape[0]}")
        print(f"Dimensión del embedding: {embeddings.shape[1]}")

        coleccion = crear_coleccion()
        guardar_en_chromadb(coleccion, chunks, embeddings)
        guardar_info(coleccion, embeddings)

        if coleccion.count() != len(chunks):
            raise ValueError(
                f"ChromaDB contiene {coleccion.count()} registros, "
                f"pero se esperaban {len(chunks)}."
            )

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)
    print(f"Colección creada       : {NOMBRE_COLECCION}")
    print(f"Registros almacenados  : {coleccion.count()}")
    print(f"Dimensión por vector   : {embeddings.shape[1]}")
    print("Métrica de distancia   : cosine")
    print(f"Base vectorial         : {BASE_VECTORIAL_DIR}")
    print(f"Archivo informativo    : {ARCHIVO_INFO}")
    print("\n[OK] ChromaDB se creó correctamente.")

    return 0


if __name__ == "__main__":
    sys.exit(main())