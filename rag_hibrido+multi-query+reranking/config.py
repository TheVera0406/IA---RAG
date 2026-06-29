"""
CONFIGURACIÓN CENTRAL DEL SISTEMA RAG VECTORIAL E5
Contiene las variables compartidas por múltiples módulos.
"""

from pathlib import Path

# Rutas principales
BASE_DIR = Path(__file__).resolve().parent
SALIDAS_DIR = BASE_DIR / "salidas"
BASE_VECTORIAL_DIR = BASE_DIR / "base_vectorial"

# Configuración general del RAG

# División de documentos
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Recuperación vectorial
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
NOMBRE_COLECCION = "documentos_obesidad_final_e5"
TOP_K_VECTORIAL = 8
UMBRAL_DISTANCIA = None

# Recuperación BM25
TOP_K_BM25 = 8

# Fusión híbrida mediante RRF
RRF_K = 60
PESO_VECTORIAL = 1.0
PESO_BM25 = 1.0
MAX_RESULTADOS_FUSION = 10

# Multi-query
NUM_CONSULTAS_ALTERNATIVAS = 3
INCLUIR_PREGUNTA_ORIGINAL = True
MODELO_MULTIQUERY = "gpt-5-mini"

# Reranking
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_CANDIDATOS_RERANK = 20
MAX_CHUNKS_CONTEXTO = 5

# Generación de la respuesta final
MODELO_LLM = "gpt-5-mini"