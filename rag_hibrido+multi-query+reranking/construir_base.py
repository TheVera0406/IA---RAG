"""
ETAPA B.1 - EXTRACCIÓN DE TEXTO DESDE DOCUMENTOS PDF

Este programa:

1. Busca todos los archivos PDF dentro de "documentos".
2. Lee cada PDF página por página.
3. Extrae y limpia el texto.
4. Conserva metadatos:
   - nombre del archivo;
   - ruta;
   - número de página;
   - cantidad de caracteres;
   - errores.
5. Guarda los resultados en:
   - salidas/paginas_extraidas.jsonl
   - salidas/resumen_documentos.csv

En esta etapa todavía NO:
- creamos chunks;
- creamos embeddings;
- utilizamos ChromaDB;
- utilizamos la API key;
- llamamos a OpenAI.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# BLOQUE 1: CONFIGURACIÓN DE RUTAS

# __file__ representa este mismo archivo.
# resolve().parent obtiene la carpeta donde está ubicado.
BASE_DIR = Path(__file__).resolve().parent

# Carpeta en la que deben estar los documentos PDF.
DOCUMENTOS_DIR = BASE_DIR / "documentos"

# Carpeta en la que se guardarán los resultados.
SALIDAS_DIR = BASE_DIR / "salidas"

# Archivo que guardará una página por línea.
ARCHIVO_PAGINAS = SALIDAS_DIR / "paginas_extraidas.jsonl"

# Archivo que guardará un resumen por documento.
ARCHIVO_RESUMEN = SALIDAS_DIR / "resumen_documentos.csv"

# BLOQUE 2: LIMPIEZA BÁSICA DEL TEXTO

def limpiar_texto(texto: str) -> str:
    """
    Limpia el texto extraído desde una página PDF.

    La función:

    - Une palabras separadas por guion y salto de línea.
    - Convierte múltiples espacios y saltos en un solo espacio.
    - Elimina espacios al inicio y al final.

    Todavía no elimina:
    - encabezados;
    - pies de página;
    - referencias;
    - títulos duplicados.
    """

    # Si la página no contiene texto, retornamos una cadena vacía.
    if not texto:
        return ""

    # Ejemplo:
    # "trata-\nmiento" se convierte en "tratamiento".
    texto = re.sub(r"-\s*\n\s*", "", texto,)

    # Reemplaza saltos de línea, tabulaciones y grupos
    # de espacios por un solo espacio.
    texto = re.sub(r"\s+", " ", texto,)

    # Elimina espacios al comienzo y al final.
    return texto.strip()

# BLOQUE 3: EXTRACCIÓN DE UN DOCUMENTO PDF

def extraer_paginas_pdf(ruta_pdf: Path,) -> tuple[list[dict], dict]:
    """
    Lee un PDF página por página.

    Parámetro:
        ruta_pdf:
            Ruta completa del documento PDF.

    Retorna:
        paginas_extraidas:
            Lista con la información de cada página.

        resumen_documento:
            Resumen general del PDF.
    """

    # Lista donde guardaremos las páginas.
    paginas_extraidas: list[dict] = []

    # Contadores del documento.
    paginas_con_texto = 0
    caracteres_totales = 0

    # Si el documento completo falla,
    # guardaremos aquí el error.
    error_documento = ""

    try:
        # Abrimos el archivo PDF.
        lector = PdfReader(str(ruta_pdf))

        # Contamos sus páginas.
        total_paginas = len(lector.pages)

        # Recorremos el PDF página por página.
        for numero_pagina, pagina in enumerate(lector.pages, start=1,):
            try:
                # Extraemos el texto.
                # Si extract_text() devuelve None,
                # usamos una cadena vacía.
                texto_original = (pagina.extract_text() or "")

                # Limpiamos el texto.
                texto_limpio = limpiar_texto(texto_original)

                # La página se procesó sin error.
                error_pagina = ""

            except Exception as error:
                # Si una página falla, no detenemos
                # la lectura de todo el documento.
                texto_limpio = ""

                error_pagina = (f"{type(error).__name__}: " f"{error}")

            # Actualizamos los contadores solamente
            # si la página tiene texto.
            if texto_limpio:
                paginas_con_texto += 1

                caracteres_totales += len(texto_limpio)

            # Creamos los datos de la página.
            registro_pagina = {"archivo": ruta_pdf.name, "ruta_relativa": str(ruta_pdf.relative_to(BASE_DIR)), "pagina": numero_pagina, "texto": texto_limpio, "cantidad_caracteres": len(texto_limpio), "error": error_pagina,}

            # Guardamos la página en la lista.
            paginas_extraidas.append(registro_pagina)

    except Exception as error:
        # Este bloque se ejecuta si el PDF completo
        # no se puede abrir.
        total_paginas = 0

        error_documento = (f"{type(error).__name__}: " f"{error}")

    # Creamos un resumen general del documento.
    resumen_documento = {"archivo": ruta_pdf.name, "total_paginas": total_paginas, "paginas_con_texto": paginas_con_texto, "paginas_sin_texto": (total_paginas - paginas_con_texto), "caracteres_totales": caracteres_totales, "error_documento": error_documento,}

    return (paginas_extraidas, resumen_documento,)

# BLOQUE 4: GUARDAR LAS PÁGINAS EN JSONL

def guardar_paginas_jsonl(paginas: list[dict],) -> None:
    """
    Guarda todas las páginas en formato JSONL.

    JSONL significa JSON Lines.

    Cada línea contiene un objeto JSON independiente.
    """

    # Abrimos el archivo en modo escritura.
    with ARCHIVO_PAGINAS.open("w", encoding="utf-8",) as archivo:

        # Escribimos una página por línea.
        for pagina in paginas:
            json.dump(pagina, archivo, ensure_ascii=False,)

            # Agregamos un salto de línea
            # después de cada página.
            archivo.write("\n")

# BLOQUE 5: GUARDAR EL RESUMEN EN CSV

def guardar_resumen_csv(resumenes: list[dict],) -> None:
    """
    Guarda un resumen por documento en un archivo CSV.
    """

    # Definimos el orden de las columnas.
    columnas = ["archivo", "total_paginas", "paginas_con_texto", "paginas_sin_texto", "caracteres_totales", "error_documento",]

    # utf-8-sig permite que Excel lea correctamente
    # letras como ñ y caracteres con tilde.
    with ARCHIVO_RESUMEN.open("w", encoding="utf-8-sig", newline="",) as archivo:

        escritor = csv.DictWriter(archivo, fieldnames=columnas,)

        # Escribimos la primera fila con
        # los nombres de las columnas.
        escritor.writeheader()

        # Escribimos una fila por cada documento.
        escritor.writerows(resumenes)

# BLOQUE 6: PROGRAMA PRINCIPAL

def main() -> int:
    """
    Coordina todo el proceso de extracción.

    Códigos de salida:

    0:
        Todo salió correctamente.

    1:
        Faltan carpetas o documentos.

    2:
        Uno o más PDF presentaron errores generales.
    """

    print("=" * 70)
    print("ETAPA B.1 - EXTRACCIÓN DE TEXTO DESDE PDF")
    print("=" * 70)

    # Comprobamos que exista la carpeta documentos.
    if not DOCUMENTOS_DIR.exists():
        print("[ERROR] No existe la carpeta: " f"{DOCUMENTOS_DIR}")

        return 1

    # Creamos la carpeta salidas si aún no existe.
    SALIDAS_DIR.mkdir(parents=True, exist_ok=True,)

    # Buscamos todos los archivos PDF.
    #
    # rglob("*") también revisa subcarpetas.
    archivos_pdf = sorted(ruta for ruta in DOCUMENTOS_DIR.rglob("*") if (ruta.is_file() and ruta.suffix.lower() == ".pdf"))

    # Si no existen PDF, detenemos el programa.
    if not archivos_pdf:
        print("[ERROR] No se encontraron PDF dentro de " f"{DOCUMENTOS_DIR}")

        return 1

    print(f"PDF encontrados: " f"{len(archivos_pdf)}\n")

    # Aquí guardaremos las páginas de todos los documentos.
    todas_las_paginas: list[dict] = []

    # Aquí guardaremos los resúmenes.
    resumenes: list[dict] = []

    # Recorremos los PDF encontrados.
    for posicion, ruta_pdf in enumerate(archivos_pdf, start=1,):
        print(f"[{posicion}/{len(archivos_pdf)}] " f"Leyendo: {ruta_pdf.name}")

        # Procesamos el documento.
        paginas, resumen = extraer_paginas_pdf(ruta_pdf)

        # Agregamos las páginas a la lista general.
        todas_las_paginas.extend(paginas)

        # Agregamos el resumen.
        resumenes.append(resumen)

        # Mostramos el resultado del documento.
        if resumen["error_documento"]:
            print("    ERROR: " f"{resumen['error_documento']}")

        else:
            print("    " f"Páginas: " f"{resumen['total_paginas']} | " f"Con texto: " f"{resumen['paginas_con_texto']} | " f"Sin texto: " f"{resumen['paginas_sin_texto']} | " f"Caracteres: " f"{resumen['caracteres_totales']:,}")

    # Guardamos las páginas.
    guardar_paginas_jsonl(todas_las_paginas)

    # Guardamos el resumen.
    guardar_resumen_csv(resumenes)

    # Calculamos estadísticas generales.
    total_documentos = len(resumenes)

    total_paginas = sum(resumen["total_paginas"] for resumen in resumenes)

    total_paginas_con_texto = sum(resumen["paginas_con_texto"] for resumen in resumenes)

    total_caracteres = sum(resumen["caracteres_totales"] for resumen in resumenes)

    documentos_con_error = sum(bool(resumen["error_documento"]) for resumen in resumenes)

    # Mostramos el resumen final.
    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)

    print(f"Documentos procesados : " f"{total_documentos}")

    print(f"Páginas totales       : " f"{total_paginas}")

    print(f"Páginas con texto     : " f"{total_paginas_con_texto}")

    print(f"Caracteres extraídos  : " f"{total_caracteres:,}")

    print(f"Documentos con error  : " f"{documentos_con_error}")

    print("\nArchivos generados:")

    print(f"  - {ARCHIVO_PAGINAS}")

    print(f"  - {ARCHIVO_RESUMEN}")

    # Si algún documento completo falló,
    # mostramos una advertencia.
    if documentos_con_error > 0:
        print("\n[AVISO] Algunos documentos presentaron " "errores. Revisa resumen_documentos.csv.")

        return 2

    print("\n[OK] La extracción terminó correctamente.")

    return 0

# BLOQUE 7: PUNTO DE ENTRADA

# Este código se ejecuta únicamente al utilizar:
#
# python construir_base.py
#
# No se ejecutará si este archivo se importa
# desde otro programa.
if __name__ == "__main__":
    sys.exit(main())
