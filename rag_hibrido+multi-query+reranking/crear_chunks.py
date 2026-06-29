"""
ETAPA B.2 - CREACIÓN DE CHUNKS

Lee:
    salidas/paginas_extraidas.jsonl

Genera:
    salidas/chunks.jsonl
    salidas/resumen_chunks.csv

En esta etapa todavía NO:
- creamos embeddings;
- utilizamos ChromaDB;
- utilizamos la API key;
- llamamos a OpenAI.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importamos la configuración centralizada
from config import BASE_DIR, SALIDAS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# BLOQUE 1: CONFIGURACIÓN DE RUTAS

ARCHIVO_PAGINAS = SALIDAS_DIR / "paginas_extraidas.jsonl"
ARCHIVO_CHUNKS = SALIDAS_DIR / "chunks.jsonl"
ARCHIVO_RESUMEN = SALIDAS_DIR / "resumen_chunks.csv"


# BLOQUE 2: CARGAR LAS PÁGINAS

def cargar_paginas() -> list[dict]:
    """Lee paginas_extraidas.jsonl y devuelve una lista de páginas."""
    paginas = []

    with ARCHIVO_PAGINAS.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea:
                continue

            try:
                paginas.append(json.loads(linea))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON inválido en la línea {numero_linea}: {error}") from error

    return paginas


# BLOQUE 3: CONFIGURAR EL DIVISOR

def crear_divisor() -> RecursiveCharacterTextSplitter:
    """Crea el divisor que separará cada página en chunks usando los valores de config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[". ", "; ", ", ", " ", ""],
    )


# BLOQUE 4: CREAR LOS CHUNKS

def crear_chunks(paginas: list[dict]) -> list[dict]:
    """Divide las páginas en chunks y conserva sus metadatos sin ensuciar el texto."""
    divisor = crear_divisor()
    chunks = []
    id_global = 1

    for pagina in paginas:
        texto = pagina.get("texto", "").strip()
        if not texto:
            continue

        fragmentos = divisor.split_text(texto)

        for numero_chunk, fragmento in enumerate(fragmentos, start=1):
            # Guardamos el fragmento limpio (sin inyectar el nombre del archivo)
            # para no añadir ruido a los embeddings del modelo E5.
            chunks.append({
                "id": f"chunk_{id_global:06d}",
                "archivo": pagina["archivo"],
                "ruta_relativa": pagina["ruta_relativa"],
                "pagina": pagina["pagina"],
                "chunk_en_pagina": numero_chunk,
                "texto": fragmento, 
                "cantidad_caracteres": len(fragmento),
            })
            id_global += 1

    return chunks


# BLOQUE 5: GUARDAR LOS CHUNKS

def guardar_chunks(chunks: list[dict]) -> None:
    """Guarda un chunk por línea en formato JSONL."""
    with ARCHIVO_CHUNKS.open("w", encoding="utf-8") as archivo:
        for chunk in chunks:
            json.dump(chunk, archivo, ensure_ascii=False)
            archivo.write("\n")


# BLOQUE 6: CREAR EL RESUMEN

def guardar_resumen(chunks: list[dict]) -> None:
    """Genera estadísticas de chunks para cada documento."""
    datos = defaultdict(lambda: {"chunks": 0, "caracteres": 0, "paginas": set()})

    for chunk in chunks:
        documento = datos[chunk["archivo"]]
        documento["chunks"] += 1
        documento["caracteres"] += chunk["cantidad_caracteres"]
        documento["paginas"].add(chunk["pagina"])

    columnas = ["archivo", "paginas_con_chunks", "total_chunks", "caracteres_en_chunks"]

    with ARCHIVO_RESUMEN.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()

        for nombre in sorted(datos):
            escritor.writerow({
                "archivo": nombre,
                "paginas_con_chunks": len(datos[nombre]["paginas"]),
                "total_chunks": datos[nombre]["chunks"],
                "caracteres_en_chunks": datos[nombre]["caracteres"],
            })


# BLOQUE 7: PROGRAMA PRINCIPAL

def main() -> int:
    """Coordina la creación y almacenamiento de los chunks."""
    print("=" * 70)
    print("ETAPA B.2 - CREACIÓN DE CHUNKS")
    print("=" * 70)

    if not ARCHIVO_PAGINAS.exists():
        print(f"[ERROR] No existe el archivo: {ARCHIVO_PAGINAS}")
        print("Ejecuta primero: python construir_base.py")
        return 1

    SALIDAS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        paginas = cargar_paginas()
        chunks = crear_chunks(paginas)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 1

    if not chunks:
        print("[ERROR] No se creó ningún chunk.")
        return 1

    guardar_chunks(chunks)
    guardar_resumen(chunks)

    longitudes = [chunk["cantidad_caracteres"] for chunk in chunks]
    paginas_con_texto = sum(bool(pagina.get("texto", "").strip()) for pagina in paginas)
    paginas_vacias = len(paginas) - paginas_con_texto

    print(f"Páginas leídas         : {len(paginas)}")
    print(f"Páginas con texto      : {paginas_con_texto}")
    print(f"Páginas vacías         : {paginas_vacias}")
    print(f"Tamaño del chunk       : {CHUNK_SIZE}")
    print(f"Overlap                : {CHUNK_OVERLAP}")
    print(f"Chunks creados         : {len(chunks)}")
    print(f"Chunk más pequeño      : {min(longitudes)} caracteres")
    print(f"Chunk promedio         : {sum(longitudes) / len(longitudes):.2f} caracteres")
    print(f"Chunk más grande       : {max(longitudes)} caracteres")

    print("\nArchivos generados:")
    print(f"  - {ARCHIVO_CHUNKS}")
    print(f"  - {ARCHIVO_RESUMEN}")
    print("\n[OK] Los chunks se crearon correctamente.")

    return 0


if __name__ == "__main__":
    sys.exit(main())