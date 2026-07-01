"""
ETAPA E - PROCESAMIENTO AUTOMÁTICO DE PREGUNTAS.CSV

Lee:
    preguntas.csv

Genera:
    salidas/respuestas.csv
    salidas/errores_procesamiento.csv

Columnas finales:
    id, answer, context
"""

import csv
import sys
import time
from pathlib import Path

import openai

from buscar_contexto import buscar_chunks, cargar_recursos
from generar_respuesta import cargar_cliente_openai, generar_respuesta


# =========================================================
# BLOQUE 1: CONFIGURACIÓN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
SALIDAS_DIR = BASE_DIR / "salidas"

ARCHIVO_PREGUNTAS = BASE_DIR / "preguntas.csv"
ARCHIVO_RESPUESTAS = SALIDAS_DIR / "respuestas.csv"
ARCHIVO_ERRORES = SALIDAS_DIR / "errores_procesamiento.csv"

LIMITE_PREGUNTAS = None    # None
PAUSA_ENTRE_PREGUNTAS = 0.5
MAX_INTENTOS = 3


# =========================================================
# BLOQUE 2: CARGAR LAS PREGUNTAS
# =========================================================

def cargar_preguntas() -> list[dict]:
    """Lee las columnas id y pregunta desde preguntas.csv."""
    with ARCHIVO_PREGUNTAS.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)

        if not lector.fieldnames or not {"id", "pregunta"}.issubset(lector.fieldnames):
            raise ValueError(
                "preguntas.csv debe contener las columnas 'id' y 'pregunta'."
            )

        preguntas = [
            {"id": fila["id"].strip(), "pregunta": fila["pregunta"].strip()}
            for fila in lector
            if fila.get("id") and fila.get("pregunta")
        ]

    preguntas = [fila for fila in preguntas if fila["id"] and fila["pregunta"]]
    return preguntas[:LIMITE_PREGUNTAS] if LIMITE_PREGUNTAS else preguntas


def cargar_ids_completados() -> set[str]:
    """Obtiene los id que ya fueron guardados en respuestas.csv."""
    if not ARCHIVO_RESPUESTAS.exists() or ARCHIVO_RESPUESTAS.stat().st_size == 0:
        return set()

    with ARCHIVO_RESPUESTAS.open("r", encoding="utf-8-sig", newline="") as archivo:
        return {
            fila["id"].strip()
            for fila in csv.DictReader(archivo)
            if fila.get("id")
        }


# =========================================================
# BLOQUE 3: PREPARAR LOS ARCHIVOS CSV
# =========================================================

def preparar_archivo(ruta: Path, columnas: list[str]) -> None:
    """Crea un CSV con encabezados solamente si no existe."""
    if ruta.exists() and ruta.stat().st_size > 0:
        return

    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        csv.DictWriter(archivo, fieldnames=columnas).writeheader()


def guardar_respuesta(id_pregunta: str, respuesta: str, contexto: str) -> None:
    """Agrega una respuesta al archivo respuestas.csv."""
    fila = {
        "id": id_pregunta,
        "answer": respuesta,
        "context": contexto,
    }

    # CORRECCIÓN: utf-8-sig para compatibilidad con Excel
    with ARCHIVO_RESPUESTAS.open("a", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=["id", "answer", "context"],
        )
        escritor.writerow(fila)


def guardar_error(id_pregunta: str, pregunta: str, error: Exception) -> None:
    """Registra una pregunta que no pudo ser procesada."""
    fila = {
        "id": id_pregunta,
        "pregunta": pregunta,
        "error": f"{type(error).__name__}: {error}",
    }

    # CORRECCIÓN: utf-8-sig para compatibilidad con Excel
    with ARCHIVO_ERRORES.open("a", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=["id", "pregunta", "error"],
        )
        escritor.writerow(fila)


# =========================================================
# BLOQUE 4: GENERAR RESPUESTAS CON REINTENTOS
# =========================================================

def generar_con_reintentos(pregunta: str, resultados: list[dict], cliente):
    """Reintenta la llamada cuando ocurre un error temporal."""
    
    # CORRECCIÓN: Se eliminó ValueError para no crear bucles infinitos
    # ante errores de contexto vacío o parámetros inválidos.
    errores_temporales = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )

    for intento in range(1, MAX_INTENTOS + 1):
        try:
            return generar_respuesta(pregunta, resultados, cliente)

        except errores_temporales:
            if intento == MAX_INTENTOS:
                raise

            espera = 2 ** intento
            print(f"    Error temporal. Nuevo intento en {espera} segundos...")
            time.sleep(espera)


# =========================================================
# BLOQUE 5: PROCESAR UNA PREGUNTA
# =========================================================

def procesar_pregunta(fila: dict, modelo_embeddings, coleccion, cliente) -> None:
    """Recupera contexto, genera la respuesta y la guarda."""
    id_pregunta = fila["id"]
    pregunta = fila["pregunta"]

    resultados = buscar_chunks(pregunta, modelo_embeddings, coleccion)
    respuesta, contexto = generar_con_reintentos(
        pregunta,
        resultados,
        cliente,
    )

    guardar_respuesta(id_pregunta, respuesta, contexto)


# =========================================================
# BLOQUE 6: PROGRAMA PRINCIPAL
# =========================================================

def main() -> int:
    """Procesa las preguntas pendientes y guarda cada resultado."""
    print("=" * 70)
    print("ETAPA E - PROCESAMIENTO DE PREGUNTAS.CSV")
    print("=" * 70)

    if not ARCHIVO_PREGUNTAS.exists():
        print(f"[ERROR] No existe: {ARCHIVO_PREGUNTAS}")
        return 1

    SALIDAS_DIR.mkdir(parents=True, exist_ok=True)

    preparar_archivo(
        ARCHIVO_RESPUESTAS,
        ["id", "answer", "context"],
    )

    preparar_archivo(
        ARCHIVO_ERRORES,
        ["id", "pregunta", "error"],
    )

    try:
        preguntas = cargar_preguntas()
        completados = cargar_ids_completados()
        pendientes = [
            fila for fila in preguntas
            if fila["id"] not in completados
        ]

        print("Cargando modelo de embeddings, ChromaDB y OpenAI...")
        modelo_embeddings, coleccion = cargar_recursos()
        cliente = cargar_cliente_openai()

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    ids_preguntas = {fila["id"] for fila in preguntas}

    print(f"Preguntas encontradas : {len(preguntas)}")
    print(f"Ya completadas        : {len(completados & ids_preguntas)}")
    print(f"Pendientes            : {len(pendientes)}")

    exitosas = 0
    fallidas = 0

    for posicion, fila in enumerate(pendientes, start=1):
        id_pregunta = fila["id"]
        pregunta = fila["pregunta"]

        print(f"\n[{posicion}/{len(pendientes)}] ID {id_pregunta}: {pregunta}")

        try:
            procesar_pregunta(
                fila,
                modelo_embeddings,
                coleccion,
                cliente,
            )

            exitosas += 1
            print("    [OK] Respuesta guardada.")
            time.sleep(PAUSA_ENTRE_PREGUNTAS)

        except Exception as error:
            fallidas += 1
            guardar_error(id_pregunta, pregunta, error)
            print(f"    [ERROR] {type(error).__name__}: {error}")

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)
    print(f"Procesadas correctamente: {exitosas}")
    print(f"Fallidas               : {fallidas}")
    print(f"Archivo final          : {ARCHIVO_RESPUESTAS}")
    print(f"Registro de errores    : {ARCHIVO_ERRORES}")

    if fallidas == 0:
        print("\n[OK] Todas las preguntas pendientes fueron procesadas.")
    else:
        print(
            "\n[AVISO] Revisa errores_procesamiento.csv "
            "y vuelve a ejecutar para reintentar."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())