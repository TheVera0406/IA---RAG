"""
RECUPERACIÓN DEL RAG FINAL

Flujo:
Pregunta original
-> generación multi-query
-> búsqueda vectorial + BM25 para cada consulta
-> fusión RRF
-> eliminación de duplicados
-> reranking
-> contexto final
"""

import json
import pickle
import re
import sys
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    MAX_CANDIDATOS_RERANK,
    MAX_RESULTADOS_FUSION,
    MODELO_EMBEDDINGS,
    NOMBRE_COLECCION,
    PESO_BM25,
    PESO_VECTORIAL,
    RRF_K,
    TOP_K_BM25,
    TOP_K_VECTORIAL,
    UMBRAL_DISTANCIA,
)

from generar_consultas import (
    cargar_cliente_openai,
    generar_consultas,
    mostrar_consultas,
)

from reranking import (
    cargar_reranker,
    rerankear_resultados,
)


# BLOQUE 1: RUTAS

BASE_DIR = Path(__file__).resolve().parent
RUTA_BASE_VECTORIAL = BASE_DIR / "base_vectorial"
ARCHIVO_CHUNKS = BASE_DIR / "salidas" / "chunks.jsonl"
ARCHIVO_BM25 = BASE_DIR / "salidas" / "indice_bm25.pkl"


# BLOQUE 2: TOKENIZACIÓN BM25

def tokenizar(texto: str) -> list[str]:
    """Convierte el texto en palabras para la búsqueda BM25."""
    return re.findall(r"\b\w+\b", texto.lower(), flags=re.UNICODE)


# BLOQUE 3: CARGAR RECURSOS

def cargar_recursos():
    """Carga E5, ChromaDB, chunks, BM25 y el reranker."""

    if not ARCHIVO_CHUNKS.exists():
        raise FileNotFoundError(
            "No existe salidas/chunks.jsonl. "
            "Ejecuta primero crear_chunks.py."
        )

    if not ARCHIVO_BM25.exists():
        raise FileNotFoundError(
            "No existe salidas/indice_bm25.pkl. "
            "Ejecuta primero crear_indice_bm25.py."
        )

    if not RUTA_BASE_VECTORIAL.exists():
        raise FileNotFoundError(
            "No existe la carpeta base_vectorial."
        )

    modelo_embeddings = SentenceTransformer(
        MODELO_EMBEDDINGS
    )

    cliente_chroma = chromadb.PersistentClient(
        path=str(RUTA_BASE_VECTORIAL)
    )

    coleccion = cliente_chroma.get_collection(
        name=NOMBRE_COLECCION
    )

    with ARCHIVO_CHUNKS.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        chunks = [
            json.loads(linea)
            for linea in archivo
            if linea.strip()
        ]

    chunks_por_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    with ARCHIVO_BM25.open("rb") as archivo:
        datos_bm25 = pickle.load(archivo)

    bm25 = datos_bm25["bm25"]
    ids_bm25 = datos_bm25["ids"]

    if (
        len(ids_bm25) != len(chunks)
        or set(ids_bm25) != set(chunks_por_id)
    ):
        raise ValueError(
            "El índice BM25 no coincide con los chunks actuales. "
            "Ejecuta nuevamente crear_indice_bm25.py."
        )

    reranker = cargar_reranker()

    return (
        modelo_embeddings,
        coleccion,
        chunks_por_id,
        bm25,
        ids_bm25,
        reranker,
    )


# BLOQUE 4: BÚSQUEDA VECTORIAL

def buscar_vectorial(
    consulta: str,
    modelo_embeddings,
    coleccion,
) -> list[dict]:
    """Recupera chunks mediante E5 y ChromaDB."""

    cantidad = min(
        TOP_K_VECTORIAL,
        coleccion.count(),
    )

    if cantidad == 0:
        return []

    texto_consulta = f"query: {consulta}"

    embedding = modelo_embeddings.encode(
        [texto_consulta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    respuesta = coleccion.query(
        query_embeddings=embedding,
        n_results=cantidad,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    resultados = []

    for id_chunk, texto, metadata, distancia in zip(
        respuesta["ids"][0],
        respuesta["documents"][0],
        respuesta["metadatas"][0],
        respuesta["distances"][0],
    ):
        distancia = float(distancia)

        if (
            UMBRAL_DISTANCIA is not None
            and distancia > UMBRAL_DISTANCIA
        ):
            continue

        resultados.append({
            "id": id_chunk,
            "texto": texto,
            "metadata": metadata,
            "distancia": distancia,
            "puntaje_bm25": None,
            "origen": "vectorial",
        })

    return resultados


# BLOQUE 5: BÚSQUEDA BM25

def buscar_bm25(
    consulta: str,
    chunks_por_id: dict,
    bm25,
    ids_bm25: list[str],
) -> list[dict]:
    """Recupera chunks mediante coincidencias léxicas BM25."""

    tokens_consulta = tokenizar(consulta)

    if not tokens_consulta:
        return []

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
                "chunk_en_pagina": chunk.get(
                    "chunk_en_pagina",
                    0,
                ),
            },
            "distancia": None,
            "puntaje_bm25": puntaje,
            "origen": "bm25",
        })

        if len(resultados) >= TOP_K_BM25:
            break

    return resultados


# BLOQUE 6: FUSIÓN RRF DE UNA CONSULTA

def fusionar_resultados(
    resultados_vectoriales: list[dict],
    resultados_bm25: list[dict],
) -> list[dict]:
    """Combina los rankings vectorial y BM25 mediante RRF."""

    fusion = {}

    for posicion, resultado in enumerate(
        resultados_vectoriales,
        start=1,
    ):
        id_chunk = resultado["id"]

        fusion[id_chunk] = {
            **resultado,
            "puntaje_rrf": (
                PESO_VECTORIAL
                / (RRF_K + posicion)
            ),
            "origenes": {"vectorial"},
        }

    for posicion, resultado in enumerate(
        resultados_bm25,
        start=1,
    ):
        id_chunk = resultado["id"]
        aporte = PESO_BM25 / (RRF_K + posicion)

        if id_chunk in fusion:
            fusion[id_chunk]["puntaje_rrf"] += aporte
            fusion[id_chunk]["puntaje_bm25"] = (
                resultado["puntaje_bm25"]
            )
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
        resultado["origen"] = " + ".join(
            sorted(resultado.pop("origenes"))
        )

    return resultados[:MAX_RESULTADOS_FUSION]


# BLOQUE 7: COMBINAR LAS CONSULTAS

def combinar_resultados_multiquery(
    resultados_por_consulta: list[tuple[str, list[dict]]],
) -> list[dict]:
    """Une resultados de todas las consultas y elimina duplicados."""

    combinados = {}

    for consulta, resultados in resultados_por_consulta:
        for resultado in resultados:
            id_chunk = resultado["id"]
            origenes = set(
                resultado["origen"].split(" + ")
            )

            if id_chunk not in combinados:
                combinados[id_chunk] = {
                    **resultado,
                    "puntaje_multiquery": resultado[
                        "puntaje_rrf"
                    ],
                    "apariciones": 1,
                    "consultas_recuperacion": [consulta],
                    "origenes": origenes,
                }
                continue

            existente = combinados[id_chunk]

            existente["puntaje_multiquery"] += (
                resultado["puntaje_rrf"]
            )

            existente["apariciones"] += 1
            existente["origenes"].update(origenes)

            if consulta not in existente[
                "consultas_recuperacion"
            ]:
                existente[
                    "consultas_recuperacion"
                ].append(consulta)

            nueva_distancia = resultado["distancia"]
            distancia_actual = existente["distancia"]

            if (
                nueva_distancia is not None
                and (
                    distancia_actual is None
                    or nueva_distancia < distancia_actual
                )
            ):
                existente["distancia"] = nueva_distancia

            nuevo_bm25 = resultado["puntaje_bm25"]
            bm25_actual = existente["puntaje_bm25"]

            if (
                nuevo_bm25 is not None
                and (
                    bm25_actual is None
                    or nuevo_bm25 > bm25_actual
                )
            ):
                existente["puntaje_bm25"] = nuevo_bm25

    candidatos = sorted(
        combinados.values(),
        key=lambda resultado: (
            resultado["puntaje_multiquery"],
            resultado["apariciones"],
        ),
        reverse=True,
    )

    for candidato in candidatos:
        candidato["origen"] = " + ".join(
            sorted(candidato.pop("origenes"))
        )

        candidato["puntaje_rrf"] = candidato[
            "puntaje_multiquery"
        ]

    return candidatos[:MAX_CANDIDATOS_RERANK]


# BLOQUE 8: RECUPERACIÓN COMPLETA

def recuperar_contexto(
    pregunta: str,
    modelo_embeddings,
    coleccion,
    chunks_por_id: dict,
    bm25,
    ids_bm25: list[str],
    reranker,
    cliente_openai,
) -> tuple[list[dict], list[str]]:
    """Ejecuta multi-query, búsqueda híbrida y reranking."""

    consultas = generar_consultas(
        pregunta,
        cliente_openai,
    )

    resultados_por_consulta = []

    for consulta in consultas:
        resultados_vectoriales = buscar_vectorial(
            consulta,
            modelo_embeddings,
            coleccion,
        )

        resultados_bm25 = buscar_bm25(
            consulta,
            chunks_por_id,
            bm25,
            ids_bm25,
        )

        resultados_fusionados = fusionar_resultados(
            resultados_vectoriales,
            resultados_bm25,
        )

        resultados_por_consulta.append(
            (
                consulta,
                resultados_fusionados,
            )
        )

    candidatos = combinar_resultados_multiquery(
        resultados_por_consulta
    )

    if not candidatos:
        raise ValueError(
            "No se encontraron candidatos relevantes."
        )

    resultados_finales = rerankear_resultados(
        pregunta,
        candidatos,
        reranker,
    )

    if not resultados_finales:
        raise ValueError(
            "El reranker no devolvió resultados."
        )

    return resultados_finales, consultas


# BLOQUE 9: CONSTRUIR CONTEXTO

def construir_contexto(
    resultados: list[dict],
) -> str:
    """Construye el contexto que recibirá OpenAI."""

    bloques = []

    for resultado in resultados:
        metadata = resultado["metadata"]
        archivo = metadata.get(
            "archivo",
            "Documento desconocido",
        )
        pagina = metadata.get(
            "pagina",
            "desconocida",
        )

        bloques.append(
            f"[Fuente: {archivo}, página {pagina}]\n"
            f"{resultado['texto']}"
        )

    return "\n\n".join(bloques)


# BLOQUE 10: MOSTRAR RESULTADOS

def mostrar_resultados(
    resultados: list[dict],
) -> None:
    """Muestra los chunks finales después del reranking."""

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES RERANKEADOS")
    print("=" * 70)

    for numero, resultado in enumerate(
        resultados,
        start=1,
    ):
        metadata = resultado["metadata"]

        print(f"\nRESULTADO {numero}")
        print(
            f"Documento       : "
            f"{metadata.get('archivo', '')}"
        )
        print(
            f"Página          : "
            f"{metadata.get('pagina', '')}"
        )
        print(
            f"Origen          : "
            f"{resultado['origen']}"
        )
        print(
            f"Apariciones     : "
            f"{resultado['apariciones']}"
        )
        print(
            f"Puntaje conjunto: "
            f"{resultado['puntaje_multiquery']:.6f}"
        )
        print(
            f"Puntaje rerank  : "
            f"{resultado['puntaje_rerank']:.6f}"
        )

        if resultado["distancia"] is not None:
            print(
                f"Mejor distancia : "
                f"{resultado['distancia']:.6f}"
            )

        if resultado["puntaje_bm25"] is not None:
            print(
                f"Mejor BM25      : "
                f"{resultado['puntaje_bm25']:.6f}"
            )

        print(
            f"Consultas       : "
            f"{len(resultado['consultas_recuperacion'])}"
        )
        print(
            f"Texto           : "
            f"{resultado['texto']}"
        )


# BLOQUE 11: PROGRAMA PRINCIPAL

def main() -> int:
    """Permite probar la recuperación completa desde terminal."""

    print("=" * 70)
    print("RAG FINAL - MULTI-QUERY + HÍBRIDO + RERANKING")
    print("=" * 70)

    try:
        print("Cargando E5, ChromaDB, BM25 y reranker...")

        (
            modelo_embeddings,
            coleccion,
            chunks_por_id,
            bm25,
            ids_bm25,
            reranker,
        ) = cargar_recursos()

        cliente_openai = cargar_cliente_openai()

        print(f"Colección       : {coleccion.name}")
        print(f"Registros       : {coleccion.count()}")
        print(f"Chunks BM25     : {len(ids_bm25)}")
        print("Recursos cargados correctamente.")

    except Exception as error:
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )
        return 1

    while True:
        pregunta = input(
            "\nEscribe una pregunta o 'salir': "
        ).strip()

        if pregunta.lower() == "salir":
            print("Programa finalizado.")
            break

        if not pregunta:
            print(
                "[AVISO] La pregunta no puede estar vacía."
            )
            continue

        try:
            resultados, consultas = recuperar_contexto(
                pregunta,
                modelo_embeddings,
                coleccion,
                chunks_por_id,
                bm25,
                ids_bm25,
                reranker,
                cliente_openai,
            )

            mostrar_consultas(consultas)
            mostrar_resultados(resultados)

            print("\n" + "=" * 70)
            print("CONTEXTO FINAL")
            print("=" * 70)
            print(construir_contexto(resultados))

        except Exception as error:
            print(
                f"[ERROR] {type(error).__name__}: {error}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())