"""
GENERACIÓN DE RESPUESTAS DEL RAG FINAL

Flujo:
Pregunta original
-> generación multi-query
-> búsqueda vectorial + BM25
-> fusión RRF
-> eliminación de duplicados
-> reranking
-> contexto final
-> OpenAI
-> respuesta
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from generar_consultas import mostrar_consultas
from buscar_contexto import (cargar_recursos,construir_contexto,recuperar_contexto)

from config import MODELO_LLM


# BLOQUE 1: CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENV = BASE_DIR / ".env"

MOSTRAR_CONTEXTO = False


# BLOQUE 2: CARGAR EL CLIENTE DE OPENAI

def cargar_cliente_openai() -> OpenAI:
    """Carga la API key desde .env y crea el cliente de OpenAI."""
    load_dotenv(ARCHIVO_ENV)
    clave = os.getenv("OPENAI_API_KEY")

    if not clave or clave == "tu_clave_real":
        raise ValueError("No se encontró una OPENAI_API_KEY válida en el archivo .env.")

    return OpenAI(api_key=clave)


# BLOQUE 3: GENERAR LA RESPUESTA

def generar_respuesta(pregunta: str, resultados: list[dict], cliente: OpenAI) -> tuple[str, str]:
    """Envía la pregunta y el contexto recuperado a OpenAI."""
    
    # CORRECCIÓN: Control estricto de contexto vacío
    if not resultados:
        raise ValueError("No existe contexto suficiente para responder.")

    contexto = construir_contexto(resultados)

    # CORRECCIÓN: Se agregó un espacio después de "internos."
    instrucciones = (
        "Eres un asistente educativo especializado en obesidad. "
        "Responde únicamente con la información del contexto entregado. "
        "No inventes información ni utilices conocimientos externos. "
        "Si el contexto no permite responder correctamente, indica que la "
        "información recuperada es insuficiente. Responde en español de forma "
        "clara, precisa y comprensible. No menciones chunks, embeddings, "
        "bases vectoriales ni procesos internos. "
        "Responde directamente la pregunta, sin comenzar con frases como "
        "'según el contexto' o 'según el material proporcionado'. "
        "Utiliza un máximo de 180 palabras."
    )

    entrada = f"""PREGUNTA:
{pregunta}

CONTEXTO:
{contexto}

Redacta una respuesta utilizando solamente el contexto anterior."""

    respuesta = cliente.responses.create(
        model=MODELO_LLM,
        instructions=instrucciones,
        input=entrada,
        reasoning={"effort": "minimal"},
        max_output_tokens=1200,
    )

    if respuesta.status == "incomplete":
        motivo = getattr(respuesta.incomplete_details, "reason", "desconocido")
        raise ValueError(f"OpenAI devolvió una respuesta incompleta: {motivo}")

    # CORRECCIÓN: Manejo seguro por si output_text es None
    texto = (respuesta.output_text or "").strip()

    if not texto:
        raise ValueError("OpenAI no devolvió una respuesta de texto.")

    return texto, contexto


# BLOQUE 4: OBTENER LAS FUENTES

def obtener_fuentes(resultados: list[dict]) -> list[str]:
    """Obtiene documento y página sin repetir fuentes."""
    fuentes, registradas = [], set()

    for resultado in resultados:
        metadata = resultado["metadata"]
        archivo = metadata.get("archivo", "Documento desconocido")
        pagina = metadata.get("pagina", "desconocida")
        clave = (archivo, pagina)

        if clave not in registradas:
            registradas.add(clave)
            fuentes.append(f"{archivo}, página {pagina}")

    return fuentes


# BLOQUE 5: MOSTRAR LA RESPUESTA

def mostrar_salida(respuesta: str, resultados: list[dict], contexto: str) -> None:
    """Muestra la respuesta, las fuentes y opcionalmente el contexto."""
    print("\n" + "=" * 70)
    print("RESPUESTA")
    print("=" * 70)
    print(respuesta)

    print("\nFUENTES RECUPERADAS:")
    for fuente in obtener_fuentes(resultados):
        print(f"- {fuente}")

    if MOSTRAR_CONTEXTO:
        print("\n" + "=" * 70)
        print("CONTEXTO UTILIZADO")
        print("=" * 70)
        print(contexto)


# BLOQUE 6: PROGRAMA PRINCIPAL

def main() -> int:
    """Permite realizar preguntas hasta que el usuario escriba salir."""
    print("=" * 70)
    print("RAG FINAL - GENERACIÓN DE RESPUESTAS")
    print("MULTI-QUERY + HÍBRIDO + RERANKING")
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

        cliente = cargar_cliente_openai()

        print(f"Colección cargada     : {coleccion.name}")
        print(f"Registros disponibles : {coleccion.count()}")
        print(f"Modelo generativo     : {MODELO_LLM}")
        print(f"Chunks disponibles    : {len(chunks_por_id)}")
        print(f"Documentos en BM25    : {len(ids_bm25)}")
        print("Reranker cargado correctamente.")

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    while True:
        pregunta = input(
            "\nEscribe una pregunta o 'salir': "
        ).strip()

        if pregunta.lower() == "salir":
            print("Programa finalizado.")
            break

        if not pregunta:
            print("[AVISO] La pregunta no puede estar vacía.")
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
                cliente,
            )

            mostrar_consultas(consultas)

            respuesta, contexto = generar_respuesta(
                pregunta,
                resultados,
                cliente,
            )

            mostrar_salida(
                respuesta,
                resultados,
                contexto,
            )

        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")

    return 0

if __name__ == "__main__":
    sys.exit(main())