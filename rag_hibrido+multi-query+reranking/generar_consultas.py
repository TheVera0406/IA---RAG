"""
GENERACIÓN MULTI-QUERY

Flujo:
Pregunta original -> OpenAI -> consultas alternativas

Este archivo genera distintas formulaciones de una misma pregunta
para aumentar la cobertura de la recuperación híbrida.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from config import (
    INCLUIR_PREGUNTA_ORIGINAL,
    MODELO_MULTIQUERY,
    NUM_CONSULTAS_ALTERNATIVAS,
)


# BLOQUE 1: CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENV = BASE_DIR / ".env"


# BLOQUE 2: ESTRUCTURA DE LA SALIDA

class ConsultasGeneradas(BaseModel):
    consultas: list[str]


# BLOQUE 3: CARGAR EL CLIENTE DE OPENAI

def cargar_cliente_openai() -> OpenAI:
    """Carga la API key desde .env y crea el cliente de OpenAI."""
    load_dotenv(ARCHIVO_ENV)
    clave = os.getenv("OPENAI_API_KEY")

    if not clave or clave == "tu_clave_real":
        raise ValueError(
            "No se encontró una OPENAI_API_KEY válida en el archivo .env."
        )

    return OpenAI(api_key=clave)


# BLOQUE 4: LIMPIAR Y VALIDAR CONSULTAS

def limpiar_texto(texto: str) -> str:
    """Elimina espacios repetidos y espacios al inicio o final."""
    return " ".join(texto.split()).strip()


def limpiar_consultas(
    pregunta: str,
    alternativas: list[str],
) -> list[str]:
    """Elimina consultas vacías, duplicadas o iguales a la original."""
    pregunta_limpia = limpiar_texto(pregunta)
    consultas, registradas = [], {pregunta_limpia.casefold()}

    for consulta in alternativas:
        consulta_limpia = limpiar_texto(consulta)

        if not consulta_limpia:
            continue

        clave = consulta_limpia.casefold()

        if clave in registradas:
            continue

        registradas.add(clave)
        consultas.append(consulta_limpia)

        if len(consultas) >= NUM_CONSULTAS_ALTERNATIVAS:
            break

    if len(consultas) != NUM_CONSULTAS_ALTERNATIVAS:
        raise ValueError(
            "OpenAI no generó la cantidad necesaria de consultas "
            f"alternativas: {len(consultas)}/{NUM_CONSULTAS_ALTERNATIVAS}."
        )

    if INCLUIR_PREGUNTA_ORIGINAL:
        return [pregunta_limpia] + consultas

    return consultas


# BLOQUE 5: GENERAR CONSULTAS ALTERNATIVAS

def generar_consultas(
    pregunta: str,
    cliente: OpenAI,
) -> list[str]:
    """Genera consultas alternativas conservando la intención original."""
    pregunta = limpiar_texto(pregunta)

    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")

    instrucciones = (
        "Eres un sistema especializado en reformulación de consultas para "
        "recuperación de información sobre obesidad. Genera exactamente "
        f"{NUM_CONSULTAS_ALTERNATIVAS} consultas alternativas en español. "
        "Todas deben conservar la intención de la pregunta original, pero "
        "deben utilizar expresiones, sinónimos o enfoques distintos. "
        "No respondas la pregunta. No agregues explicaciones. "
        "No inventes medicamentos, enfermedades, cifras ni conceptos que "
        "no estén presentes o implícitos en la pregunta. "
        "Las consultas deben ser claras, independientes y diferentes entre sí."
    )

    respuesta = cliente.responses.parse(
        model=MODELO_MULTIQUERY,
        input=[
            {
                "role": "system",
                "content": instrucciones,
            },
            {
                "role": "user",
                "content": f"Pregunta original: {pregunta}",
            },
        ],
        text_format=ConsultasGeneradas,
        reasoning={"effort": "minimal"},
        max_output_tokens=500,
    )

    if respuesta.status == "incomplete":
        motivo = getattr(
            respuesta.incomplete_details,
            "reason",
            "desconocido",
        )
        raise ValueError(
            f"OpenAI devolvió una respuesta incompleta: {motivo}"
        )

    datos = respuesta.output_parsed

    if datos is None:
        raise ValueError(
            "OpenAI no devolvió consultas estructuradas."
        )

    return limpiar_consultas(
        pregunta,
        datos.consultas,
    )


# BLOQUE 6: MOSTRAR CONSULTAS

def mostrar_consultas(consultas: list[str]) -> None:
    """Muestra las consultas generadas en la terminal."""
    print("\n" + "=" * 70)
    print("CONSULTAS GENERADAS")
    print("=" * 70)

    for numero, consulta in enumerate(consultas, start=1):
        tipo = "original" if numero == 1 and INCLUIR_PREGUNTA_ORIGINAL else "alternativa"
        print(f"{numero}. [{tipo}] {consulta}")


# BLOQUE 7: PROGRAMA PRINCIPAL

def main() -> int:
    """Permite probar la generación multi-query desde la terminal."""
    print("=" * 70)
    print("GENERADOR MULTI-QUERY")
    print("=" * 70)

    try:
        cliente = cargar_cliente_openai()
        print(f"Modelo multi-query: {MODELO_MULTIQUERY}")

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
            consultas = generar_consultas(
                pregunta,
                cliente,
            )
            mostrar_consultas(consultas)

        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())