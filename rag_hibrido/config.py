"""
CONFIGURACIÓN CENTRAL DEL SISTEMA RAG VECTORIAL E5
Contiene las variables compartidas por múltiples módulos.
"""

from pathlib import Path

# Rutas principales
BASE_DIR = Path(__file__).resolve().parent
SALIDAS_DIR = BASE_DIR / "salidas"
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

# Configuración general del RAG híbrido

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Recuperación vectorial
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NOMBRE_COLECCION = "documentos_obesidad_hibrido_e5"
TOP_K_VECTORIAL = 8
UMBRAL_DISTANCIA = None

# Recuperación BM25
TOP_K_BM25 = 8

# Fusión de rankings
RRF_K = 60
PESO_VECTORIAL = 1.0
PESO_BM25 = 1.0
MAX_CHUNKS_CONTEXTO = 5

# Generación
MODELO_LLM = "gpt-5-mini"