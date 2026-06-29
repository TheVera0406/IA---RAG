import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CHUNKS = BASE_DIR / "salidas" / "chunks.jsonl"
ARCHIVO_INDICE = BASE_DIR / "salidas" / "indice_bm25.pkl"
ARCHIVO_INFO = BASE_DIR / "salidas" / "info_bm25.json"


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto.lower(), flags=re.UNICODE)


def cargar_chunks():
    if not ARCHIVO_CHUNKS.exists():
        raise FileNotFoundError("No existe salidas/chunks.jsonl. Ejecuta primero crear_chunks.py.")

    with ARCHIVO_CHUNKS.open("r", encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def main():
    chunks = cargar_chunks()

    if not chunks:
        raise ValueError("No existen chunks para construir el índice BM25.")

    corpus_tokenizado = [tokenizar(chunk["texto"]) for chunk in chunks]
    indice_bm25 = BM25Okapi(corpus_tokenizado)

    datos_indice = {
        "bm25": indice_bm25,
        "ids": [chunk["id"] for chunk in chunks],
    }

    ARCHIVO_INDICE.parent.mkdir(parents=True, exist_ok=True)

    with ARCHIVO_INDICE.open("wb") as archivo:
        pickle.dump(datos_indice, archivo)

    informacion = {
        "algoritmo": "BM25Okapi",
        "cantidad_documentos": len(chunks),
        "archivo_chunks": str(ARCHIVO_CHUNKS),
        "archivo_indice": str(ARCHIVO_INDICE),
    }

    with ARCHIVO_INFO.open("w", encoding="utf-8") as archivo:
        json.dump(informacion, archivo, ensure_ascii=False, indent=4)

    print("\nÍNDICE BM25 CREADO")
    print(f"Chunks indexados : {len(chunks)}")
    print(f"Índice generado  : {ARCHIVO_INDICE}")
    print(f"Información      : {ARCHIVO_INFO}")


if __name__ == "__main__":
    main()