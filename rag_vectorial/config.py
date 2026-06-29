"""
CONFIGURACIÓN CENTRAL DEL SISTEMA RAG VECTORIAL E5
Contiene las variables compartidas por múltiples módulos.
"""

from pathlib import Path

# Rutas principales
BASE_DIR = Path(__file__).resolve().parent
SALIDAS_DIR = BASE_DIR / "salidas"
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

# Configuración de Chunks
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configuración de Embeddings y ChromaDB
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
NOMBRE_COLECCION = "documentos_obesidad_vectorial_e5"

# Configuración de Búsqueda
TOP_K_CANDIDATOS = 8
MAX_CHUNKS_CONTEXTO = 5
UMBRAL_DISTANCIA = None # Lo dejamos en None para probar primero y calibrarlo

# Configuración de LLM
MODELO_LLM = "gpt-5-mini"