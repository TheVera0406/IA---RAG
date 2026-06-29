"""
RAG HÍBRIDO - Sistema de Recuperación Aumentada
Combina: Embeddings + BM25 + Reranking

Autor: Cristian Vera
Etapa A → Híbrido
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
from collections import Counter
import math


class EmbeddingsSimulator:
    """
    Simulador de embeddings determinístico.
    En producción: usar sentence-transformers o OpenAI
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed(self, text: str) -> np.ndarray:
        """Genera embedding determinístico basado en SHA256"""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convertir bytes a valores [0, 1]
        embedding = np.array([float(b) / 256.0 for b in hash_bytes])
        
        # Expandir dimensionalidad
        embedding = np.tile(embedding, math.ceil(self.dimension / len(embedding)))[:self.dimension]
        
        # Normalizar
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para múltiples textos"""
        return np.array([self.embed(text) for text in texts])


class BM25Retriever:
    """
    Implementación de BM25 para retrieval léxico.
    Algoritmo clásico que pondera frecuencia de términos.
    """
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        
        # Procesar documentos
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0
        
        # Calcular IDF (Inverse Document Frequency)
        self.idf = self._compute_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza y normaliza texto"""
        text = text.lower()
        # Remover puntuación básica y dividir
        for char in ".,;:!?()[]{}\"'":
            text = text.replace(char, " ")
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filtrar tokens cortos
    
    def _compute_idf(self) -> Dict[str, float]:
        """Calcula IDF para todos los términos"""
        idf = {}
        doc_count = len(self.tokenized_docs)
        
        # Contar documentos que contienen cada término
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                idf[term] = idf.get(term, 0) + 1
        
        # Convertir a IDF: log(N / df)
        for term in idf:
            idf[term] = math.log((doc_count - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
        
        return idf
    
    def score(self, query: str) -> List[float]:
        """Calcula BM25 score para todos los documentos"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            
            # Contar frecuencias en el documento
            term_freq = Counter(doc_tokens)
            
            for term in query_tokens:
                if term not in term_freq:
                    continue
                
                freq = term_freq[term]
                idf = self.idf.get(term, 0)
                
                # Fórmula BM25
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                
                score += numerator / denominator
            
            scores.append(score)
        
        return scores


class HybridRetriever:
    """
    Retriever Híbrido: combina embeddings + BM25
    """
    
    def __init__(self, documents: List[Dict], embedding_weight: float = 0.5):
        """
        Args:
            documents: Lista de diccionarios con 'id', 'content', 'metadata'
            embedding_weight: Peso para embeddings (0-1), BM25 obtiene (1-weight)
        """
        self.documents = documents
        self.embedding_weight = embedding_weight
        self.bm25_weight = 1 - embedding_weight
        
        # Inicializar componentes
        self.embedder = EmbeddingsSimulator(dimension=384)
        
        # Extraer contenidos para BM25
        doc_contents = [doc['content'] for doc in documents]
        self.bm25 = BM25Retriever(doc_contents)
        
        # Pre-calcular embeddings de documentos
        print("🔄 Calculando embeddings de documentos...")
        self.doc_embeddings = self.embedder.embed_batch(doc_contents)
        print(f"✓ {len(self.doc_embeddings)} documentos indexados")
    
    def retrieve(self, query: str, top_k: int = 3, return_scores: bool = False) -> List[Dict]:
        """
        Retrieval híbrido: combina múltiples señales
        """
        # 1. EMBEDDINGS: similitud coseno
        query_embedding = self.embedder.embed(query)
        embedding_scores = np.dot(self.doc_embeddings, query_embedding)
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-10)
        
        # 2. BM25: relevancia léxica
        bm25_scores = np.array(self.bm25.score(query))
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        # 3. SCORE HÍBRIDO
        hybrid_scores = (
            self.embedding_weight * embedding_scores + 
            self.bm25_weight * bm25_scores
        )
        
        # 4. TOP-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            if return_scores:
                doc['retrieval_score'] = float(hybrid_scores[idx])
                doc['embedding_score'] = float(embedding_scores[idx])
                doc['bm25_score'] = float(bm25_scores[idx])
            results.append(doc)
        
        return results


class RAGHibrido:
    """
    Sistema RAG Híbrido completo.
    """
    
    def __init__(self, corpus_file: str = None):
        """
        Inicializa el sistema RAG.
        
        Args:
            corpus_file: Ruta a archivo JSON con documentos
        """
        self.documents = []
        self.retriever = None
        
        if corpus_file and os.path.exists(corpus_file):
            self.load_corpus(corpus_file)
    
    def load_corpus(self, corpus_file: str):
        """Carga corpus desde archivo JSON"""
        print(f"📂 Cargando corpus desde {corpus_file}...")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.documents = data
        elif isinstance(data, dict) and 'documents' in data:
            self.documents = data['documents']
        else:
            raise ValueError("Formato de corpus inválido")
        
        print(f"✓ {len(self.documents)} documentos cargados")
        
        # Inicializar retriever
        self.retriever = HybridRetriever(self.documents, embedding_weight=0.5)
    
    def add_documents(self, documents: List[Dict]):
        """Añade documentos al corpus"""
        self.documents.extend(documents)
        print(f"✓ {len(documents)} documentos añadidos (total: {len(self.documents)})")
        
        # Re-inicializar retriever
        self.retriever = HybridRetriever(self.documents)
    
    def search(self, query: str, top_k: int = 3, verbose: bool = True) -> List[Dict]:
        """
        Busca documentos relevantes.
        
        Args:
            query: Pregunta o texto de búsqueda
            top_k: Número de documentos a recuperar
            verbose: Mostrar scores de retrieval
        """
        if not self.retriever:
            raise ValueError("Corpus no cargado. Usa load_corpus() primero.")
        
        results = self.retriever.retrieve(query, top_k=top_k, return_scores=True)
        
        if verbose:
            print(f"\n🔍 Pregunta: {query}")
            print(f"📊 Top {top_k} resultados:\n")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. [{doc.get('title', 'Sin título')}]")
                print(f"     Hybrid: {doc['retrieval_score']:.3f} | "
                      f"Embedding: {doc['embedding_score']:.3f} | "
                      f"BM25: {doc['bm25_score']:.3f}")
                print(f"     {doc['content'][:100]}...\n")
        
        return results
    
    def format_context(self, documents: List[Dict]) -> str:
        """Formatea documentos como contexto para LLM"""
        context_parts = []
        
        for doc in documents:
            source = doc.get('source', 'Unknown')
            title = doc.get('title', 'Sin título')
            content = doc.get('content', '')
            
            context_parts.append(
                f"[{source} - {title}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate_rag_prompt(self, query: str, context: str) -> str:
        """Genera prompt para el LLM"""
        prompt = f"""Eres un asistente experto que responde basándose ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{query}

INSTRUCCIONES:
1. Responde SOLO con información del contexto
2. Si no sabes, dilo explícitamente
3. Cita las fuentes usadas
4. Sé claro y conciso

RESPUESTA:"""
        
        return prompt


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # 1. Crear corpus de ejemplo
    print("=" * 70)
    print("RAG HÍBRIDO - DEMO")
    print("=" * 70)
    
    corpus_data = [
        {
            "id": "doc_1",
            "title": "Definición de Obesidad",
            "source": "OMS",
            "content": """La obesidad es una enfermedad crónica caracterizada por un exceso de grasa 
            corporal que presenta un riesgo para la salud. Se define usando el Índice de Masa Corporal (IMC).
            Un IMC de 30 o mayor se considera obesidad."""
        },
        {
            "id": "doc_2",
            "title": "Causas de Obesidad",
            "source": "NIH",
            "content": """La obesidad es multifactorial. Las principales causas incluyen factores genéticos (40-70% heredabilidad),
            factores ambientales (alimentos ultraprocesados, sedentarismo), y factores fisiológicos (metabolismo, hormonas)."""
        },
        {
            "id": "doc_3",
            "title": "Complicaciones de Salud",
            "source": "Mayo Clinic",
            "content": """La obesidad aumenta riesgo de enfermedades cardiovasculares, diabetes tipo 2, hipertensión, apnea del sueño,
            osteoartritis, y ciertos tipos de cáncer. El riesgo es proporcional a la duración e intensidad de la obesidad."""
        },
        {
            "id": "doc_4",
            "title": "Tratamiento: Dieta y Ejercicio",
            "source": "ACSM",
            "content": """El tratamiento incluye cambios en la dieta, aumento de actividad física, y apoyo psicológico.
            Se recomienda déficit calórico de 500 kcal/día para pérdida de 0.5 kg/semana. El ejercicio mínimo es 150 min/semana."""
        },
        {
            "id": "doc_5",
            "title": "Medicamentos para Pérdida de Peso",
            "source": "FDA",
            "content": """Existen medicamentos aprobados como GLP-1 agonistas (semaglutida) que producen pérdida de 15-22% del peso.
            Deben combinarse con dieta y ejercicio. Requieren prescripción médica."""
        },
        {
            "id": "doc_6",
            "title": "Cirugía Bariátrica",
            "source": "American Surgery Society",
            "content": """La cirugía bariátrica (bypass gástrico, manga gástrica) es opción para IMC ≥ 40.
            Produce pérdida de 50-70% del exceso de peso. Requiere seguimiento nutricional de por vida."""
        }
    ]
    
    # 2. Guardar corpus
    corpus_file = "corpus_rag.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)
    
    # 3. Inicializar RAG
    rag = RAGHibrido(corpus_file)
    
    # 4. Realizar búsquedas
    queries = [
        "¿Qué es la obesidad?",
        "¿Por qué causa complicaciones?",
        "¿Cómo se trata la obesidad?",
        "¿Funciona la cirugía?"
    ]
    
    print("\n" + "=" * 70)
    print("BÚSQUEDAS HÍBRIDAS")
    print("=" * 70)
    
    for query in queries:
        retrieved = rag.search(query, top_k=2, verbose=True)
        
        # Mostrar prompt generado
        context = rag.format_context(retrieved)
        prompt = rag.generate_rag_prompt(query, context)
        print("💬 PROMPT PARA LLM (primeros 300 caracteres):")
        print(prompt[:300] + "...\n")
        print("-" * 70 + "\n")
    
    print("\n✅ Demo completada. Ahora puedes:")
    print("   1. Cargar tu propio corpus")
    print("   2. Integrar con un LLM real (Claude, OpenAI, etc.)")
    print("   3. Ajustar pesos: embedding_weight=0.5 (50% embedding, 50% BM25)")
    print("   4. Añadir reranking o filtros adicionales")
