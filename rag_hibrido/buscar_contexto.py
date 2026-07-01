import json
import pickle
import re
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    MAX_CHUNKS_CONTEXTO,
    MODELO_EMBEDDINGS,
    NOMBRE_COLECCION,
    PESO_BM25,
    PESO_VECTORIAL,
    RRF_K,
    TOP_K_BM25,
    TOP_K_VECTORIAL,
    UMBRAL_DISTANCIA,
)

BASE_DIR = Path(__file__).resolve().parent
RUTA_BASE_VECTORIAL = BASE_DIR / "base_vectorial"
ARCHIVO_CHUNKS = BASE_DIR / "salidas" / "chunks.jsonl"
ARCHIVO_BM25 = BASE_DIR / "salidas" / "indice_bm25.pkl"


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto.lower(), flags=re.UNICODE)


def cargar_recursos():
    if not ARCHIVO_CHUNKS.exists():
        raise FileNotFoundError("No existe salidas/chunks.jsonl.")

    if not ARCHIVO_BM25.exists():
        raise FileNotFoundError("No existe salidas/indice_bm25.pkl.")

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    cliente = chromadb.PersistentClient(path=str(RUTA_BASE_VECTORIAL))
    coleccion = cliente.get_collection(name=NOMBRE_COLECCION)

    with ARCHIVO_CHUNKS.open("r", encoding="utf-8") as archivo:
        chunks = [json.loads(linea) for linea in archivo if linea.strip()]

    chunks_por_id = {chunk["id"]: chunk for chunk in chunks}

    with ARCHIVO_BM25.open("rb") as archivo:
        datos_bm25 = pickle.load(archivo)

    bm25 = datos_bm25["bm25"]
    ids_bm25 = datos_bm25["ids"]

    if len(ids_bm25) != len(chunks):
        raise ValueError("El índice BM25 no coincide con los chunks actuales.")

    return modelo, coleccion, chunks_por_id, bm25, ids_bm25


def buscar_vectorial(pregunta, modelo, coleccion):
    texto_consulta = f"query: {pregunta}"

    embedding = modelo.encode(
        [texto_consulta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    cantidad = min(TOP_K_VECTORIAL, coleccion.count())

    consulta = coleccion.query(
        query_embeddings=embedding,
        n_results=cantidad,
        include=["documents", "metadatas", "distances"],
    )

    resultados = []

    for id_chunk, texto, metadata, distancia in zip(
        consulta["ids"][0],
        consulta["documents"][0],
        consulta["metadatas"][0],
        consulta["distances"][0],
    ):
        if UMBRAL_DISTANCIA is None or distancia <= UMBRAL_DISTANCIA:
            resultados.append({
                "id": id_chunk,
                "texto": texto,
                "metadata": metadata,
                "distancia": float(distancia),
                "puntaje_bm25": None,
                "origen": "vectorial",
            })

    return resultados


def buscar_bm25(pregunta, chunks_por_id, bm25, ids_bm25):
    tokens_consulta = tokenizar(pregunta)
    puntajes = bm25.get_scores(tokens_consulta)
    posiciones = np.argsort(puntajes)[::-1]

    resultados = []

    for posicion in posiciones:
        puntaje = float(puntajes[posicion])

        if puntaje <= 0:
            continue

        id_chunk = ids_bm25[posicion]
        chunk = chunks_por_id[id_chunk]

        resultados.append({
            "id": id_chunk,
            "texto": chunk["texto"],
            "metadata": {
                "archivo": chunk["archivo"],
                "pagina": chunk["pagina"],
                "chunk_en_pagina": chunk.get("chunk_en_pagina", 0),
            },
            "distancia": None,
            "puntaje_bm25": puntaje,
            "origen": "bm25",
        })

        if len(resultados) >= TOP_K_BM25:
            break

    return resultados


def fusionar_resultados(resultados_vectoriales, resultados_bm25):
    fusion = {}

    for posicion, resultado in enumerate(resultados_vectoriales, start=1):
        id_chunk = resultado["id"]
        fusion[id_chunk] = {
            **resultado,
            "puntaje_rrf": PESO_VECTORIAL / (RRF_K + posicion),
            "origenes": {"vectorial"},
        }

    for posicion, resultado in enumerate(resultados_bm25, start=1):
        id_chunk = resultado["id"]
        aporte = PESO_BM25 / (RRF_K + posicion)

        if id_chunk in fusion:
            fusion[id_chunk]["puntaje_rrf"] += aporte
            fusion[id_chunk]["puntaje_bm25"] = resultado["puntaje_bm25"]
            fusion[id_chunk]["origenes"].add("bm25")
        else:
            fusion[id_chunk] = {
                **resultado,
                "puntaje_rrf": aporte,
                "origenes": {"bm25"},
            }

    resultados = sorted(
        fusion.values(),
        key=lambda resultado: resultado["puntaje_rrf"],
        reverse=True,
    )

    for resultado in resultados:
        resultado["origen"] = " + ".join(sorted(resultado.pop("origenes")))

    return resultados[:MAX_CHUNKS_CONTEXTO]


def recuperar_contexto(pregunta, modelo, coleccion, chunks_por_id, bm25, ids_bm25):
    resultados_vectoriales = buscar_vectorial(pregunta, modelo, coleccion)
    resultados_bm25 = buscar_bm25(pregunta, chunks_por_id, bm25, ids_bm25)

    resultados = fusionar_resultados(
        resultados_vectoriales,
        resultados_bm25,
    )

    if not resultados:
        raise ValueError("No se encontraron fragmentos relevantes para la pregunta.")

    return resultados


def construir_contexto(resultados):
    bloques = []

    for resultado in resultados:
        metadata = resultado["metadata"]
        fuente = metadata.get("archivo", "Documento desconocido")
        pagina = metadata.get("pagina", "Sin página")

        bloques.append(
            f"[Fuente: {fuente}, página {pagina}]\n"
            f"{resultado['texto']}"
        )

    return "\n\n".join(bloques)


def mostrar_resultados(resultados):
    for numero, resultado in enumerate(resultados, start=1):
        metadata = resultado["metadata"]

        print(f"\nRESULTADO {numero}")
        print(f"Origen      : {resultado['origen']}")
        print(f"Documento   : {metadata.get('archivo', '')}")
        print(f"Página      : {metadata.get('pagina', '')}")
        print(f"Puntaje RRF : {resultado['puntaje_rrf']:.6f}")

        if resultado["distancia"] is not None:
            print(f"Distancia   : {resultado['distancia']:.6f}")

        if resultado["puntaje_bm25"] is not None:
            print(f"Puntaje BM25: {resultado['puntaje_bm25']:.6f}")

        print(f"Texto       : {resultado['texto']}")


def main():
    modelo, coleccion, chunks_por_id, bm25, ids_bm25 = cargar_recursos()

    print("\nRAG HÍBRIDO")
    print("Escribe una pregunta o 'salir' para terminar.")

    while True:
        pregunta = input("\nPregunta: ").strip()

        if pregunta.lower() == "salir":
            break

        if not pregunta:
            continue

        resultados = recuperar_contexto(
            pregunta,
            modelo,
            coleccion,
            chunks_por_id,
            bm25,
            ids_bm25,
        )

        mostrar_resultados(resultados)

        print("\nCONTEXTO FINAL")
        print(construir_contexto(resultados))


if __name__ == "__main__":
    main()