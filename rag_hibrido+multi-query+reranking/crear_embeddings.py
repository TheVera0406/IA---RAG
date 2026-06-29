"""
ETAPA B.3 - CREACIÓN DE EMBEDDINGS

Lee:
    salidas/chunks.jsonl

Genera:
    salidas/embeddings.npy
    salidas/info_embeddings.json

Todavía NO usa ChromaDB, API key ni OpenAI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Importamos la configuración centralizada
from config import SALIDAS_DIR, MODELO_EMBEDDINGS


# BLOQUE 1: CONFIGURACIÓN DE RUTAS

ARCHIVO_CHUNKS = SALIDAS_DIR / "chunks.jsonl"
ARCHIVO_EMBEDDINGS = SALIDAS_DIR / "embeddings.npy"
ARCHIVO_INFO = SALIDAS_DIR / "info_embeddings.json"
BATCH_SIZE = 32


# BLOQUE 2: CARGAR LOS CHUNKS

def cargar_chunks() -> list[dict]:
    """Lee chunks.jsonl y devuelve todos los chunks."""
    chunks = []

    with ARCHIVO_CHUNKS.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea:
                continue

            try:
                chunk = json.loads(linea)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido en la línea {numero_linea}: {error}"
                ) from error

            if not chunk.get("id") or not chunk.get("texto", "").strip():
                raise ValueError(
                    f"El chunk de la línea {numero_linea} no tiene ID o texto."
                )

            chunks.append(chunk)

    return chunks


# BLOQUE 3: CREAR LOS EMBEDDINGS

def crear_embeddings(chunks: list[dict]) -> np.ndarray:
    """Carga el modelo y convierte los textos en vectores."""
    print(f"Cargando modelo: {MODELO_EMBEDDINGS}")

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    
    # CORRECCIÓN CRÍTICA: Agregamos el prefijo 'passage: ' 
    # requerido por el modelo E5 para los documentos indexados.
    textos = [f"passage: {chunk['texto']}" for chunk in chunks]

    return modelo.encode(
        textos,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# BLOQUE 4: VERIFICAR Y GUARDAR

def verificar_embeddings(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Comprueba que exista un vector válido por cada chunk."""
    if embeddings.ndim != 2:
        raise ValueError("La matriz de embeddings debe tener dos dimensiones.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Hay {len(chunks)} chunks y {embeddings.shape[0]} embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Los embeddings contienen valores NaN o infinitos.")


def guardar_resultados(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Guarda la matriz y un resumen de sus características."""
    np.save(ARCHIVO_EMBEDDINGS, embeddings)

    info = {
        "modelo": MODELO_EMBEDDINGS,
        "cantidad_chunks": len(chunks),
        "cantidad_embeddings": int(embeddings.shape[0]),
        "dimension_embedding": int(embeddings.shape[1]),
        "tipo_dato": str(embeddings.dtype),
        "normalizados": True,
        "batch_size": BATCH_SIZE,
        "primer_chunk_id": chunks[0]["id"],
        "ultimo_chunk_id": chunks[-1]["id"],
    }

    with ARCHIVO_INFO.open("w", encoding="utf-8") as archivo:
        json.dump(info, archivo, ensure_ascii=False, indent=4)


# BLOQUE 5: PROGRAMA PRINCIPAL

def main() -> int:
    """Coordina la creación, verificación y almacenamiento."""
    print("=" * 70)
    print("ETAPA B.3 - CREACIÓN DE EMBEDDINGS")
    print("=" * 70)

    if not ARCHIVO_CHUNKS.exists():
        print(f"[ERROR] No existe el archivo: {ARCHIVO_CHUNKS}")
        print("Ejecuta primero: python crear_chunks.py")
        return 1

    try:
        chunks = cargar_chunks()

        if not chunks:
            raise ValueError("No se encontró ningún chunk.")

        print(f"Chunks cargados: {len(chunks)}")

        embeddings = crear_embeddings(chunks)
        verificar_embeddings(chunks, embeddings)
        guardar_resultados(chunks, embeddings)

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)
    print(f"Modelo utilizado       : {MODELO_EMBEDDINGS}")
    print(f"Chunks procesados      : {len(chunks)}")
    print(f"Embeddings creados     : {embeddings.shape[0]}")
    print(f"Dimensión por vector   : {embeddings.shape[1]}")
    print(f"Tipo de dato           : {embeddings.dtype}")
    print("Embeddings normalizados: Sí")

    print("\nArchivos generados:")
    print(f"  - {ARCHIVO_EMBEDDINGS}")
    print(f"  - {ARCHIVO_INFO}")

    print("\n[OK] Los embeddings se crearon correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())