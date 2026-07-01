"""Verifica que el entorno de la etapa A esté correctamente configurado.

Este script NO consume la API de OpenAI. Solo comprueba:
1. La versión de Python.
2. Que las carpetas principales existan.
3. Que las librerías necesarias puedan importarse.
4. Que exista una variable OPENAI_API_KEY, sin mostrar su contenido.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CARPETAS_REQUERIDAS = (
    BASE_DIR / "documentos",
    BASE_DIR / "base_vectorial",
    BASE_DIR / "salidas",
)

# El nombre de instalación puede ser distinto del nombre usado en import.
# CORRECCIÓN: Librerías actualizadas para el RAG Vectorial
LIBRERIAS = {
    "pypdf": "pypdf",
    "langchain-text-splitters": "langchain_text_splitters",
    "sentence-transformers": "sentence_transformers",
    "chromadb": "chromadb",
    "numpy": "numpy",
    "openai": "openai",
    "python-dotenv": "dotenv",
    "tqdm": "tqdm",
    "rank-bm25": "rank_bm25",
}


def verificar_python() -> bool:
    version = sys.version_info
    correcta = version >= (3, 10)
    estado = "OK" if correcta else "ERROR"
    print(f"[{estado}] Python {version.major}.{version.minor}.{version.micro}")
    if not correcta:
        print("       Se necesita Python 3.10 o superior; recomendamos Python 3.11.")
    return correcta


def verificar_carpetas() -> bool:
    todo_correcto = True
    for carpeta in CARPETAS_REQUERIDAS:
        if carpeta.is_dir():
            print(f"[OK] Carpeta encontrada: {carpeta.name}/")
        else:
            print(f"[ERROR] Falta la carpeta: {carpeta}")
            todo_correcto = False
    return todo_correcto


def verificar_librerias() -> bool:
    todo_correcto = True
    for paquete, modulo in LIBRERIAS.items():
        try:
            importlib.import_module(modulo)
            print(f"[OK] Librería disponible: {paquete}")
        except Exception as error:
            print(f"[ERROR] No se pudo importar {paquete}: {error}")
            todo_correcto = False
    return todo_correcto


def verificar_api_key() -> bool:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key and api_key != "tu_clave_real_aqui":
        print("[OK] OPENAI_API_KEY fue encontrada (su contenido permanece oculto).")
        return True

    print("[AVISO] OPENAI_API_KEY todavía no está configurada.")
    print("        Esto no impide completar las primeras etapas del RAG.")
    print("        Copia .env.example como .env y agrega tu clave antes de generar respuestas.")
    return False


def main() -> None:
    print("=" * 62)
    print("VERIFICACIÓN DEL ENTORNO — RAG VECTORIAL E5")
    print("=" * 62)

    resultados_obligatorios = [
        verificar_python(),
        verificar_carpetas(),
        verificar_librerias(),
    ]
    verificar_api_key()

    print("=" * 62)
    if all(resultados_obligatorios):
        print("ENTORNO LISTO: podemos comenzar a procesar los documentos.")
    else:
        print("ENTORNO INCOMPLETO: corrige los elementos marcados como ERROR.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()