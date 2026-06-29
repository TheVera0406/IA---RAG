"""
RERANKING DE RESULTADOS

Flujo:
Pregunta original + chunks candidatos
-> CrossEncoder
-> puntuación de relevancia
-> orden descendente
-> mejores chunks
"""

from sentence_transformers import CrossEncoder

from config import MAX_CHUNKS_CONTEXTO, MODELO_RERANKER


# BLOQUE 1: CARGAR EL MODELO RERANKER

def cargar_reranker() -> CrossEncoder:
    """Carga el modelo CrossEncoder utilizado para reranking."""
    return CrossEncoder(MODELO_RERANKER)


# BLOQUE 2: RERANKEAR RESULTADOS

def rerankear_resultados(
    pregunta: str,
    resultados: list[dict],
    reranker: CrossEncoder,
    max_resultados: int = MAX_CHUNKS_CONTEXTO,
) -> list[dict]:
    """Ordena los chunks según su relación directa con la pregunta."""

    if not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    if not resultados:
        return []

    pares = [
        (pregunta, resultado["texto"])
        for resultado in resultados
    ]

    puntajes = reranker.predict(
        pares,
        show_progress_bar=False,
    )

    resultados_rerankeados = []

    for resultado, puntaje in zip(resultados, puntajes):
        resultado_actualizado = resultado.copy()
        resultado_actualizado["puntaje_rerank"] = float(puntaje)
        resultados_rerankeados.append(resultado_actualizado)

    resultados_rerankeados.sort(
        key=lambda resultado: resultado["puntaje_rerank"],
        reverse=True,
    )

    return resultados_rerankeados[:max_resultados]


# BLOQUE 3: MOSTRAR RESULTADOS

def mostrar_reranking(resultados: list[dict]) -> None:
    """Muestra los resultados ordenados por el reranker."""

    print("\n" + "=" * 70)
    print("RESULTADOS RERANKEADOS")
    print("=" * 70)

    for numero, resultado in enumerate(resultados, start=1):
        print(f"\nRESULTADO {numero}")
        print(f"Puntaje rerank : {resultado['puntaje_rerank']:.6f}")
        print(f"Texto           : {resultado['texto']}")


# BLOQUE 4: PRUEBA INDEPENDIENTE

def main() -> None:
    """Prueba el reranker sin utilizar todavía ChromaDB ni BM25."""

    pregunta = "¿Cuáles son las causas de la obesidad?"

    resultados_prueba = [
        {
            "id": "prueba_1",
            "texto": (
                "La obesidad está relacionada con factores genéticos, "
                "metabólicos, conductuales, ambientales y sociales."
            ),
        },
        {
            "id": "prueba_2",
            "texto": (
                "La actividad física mejora la capacidad cardiovascular "
                "y puede favorecer el bienestar general."
            ),
        },
        {
            "id": "prueba_3",
            "texto": (
                "Las causas de la obesidad incluyen factores psicológicos, "
                "neurohormonales, socioculturales y medicamentos."
            ),
        },
    ]

    print("=" * 70)
    print("PRUEBA DEL MODELO RERANKER")
    print("=" * 70)
    print(f"Modelo   : {MODELO_RERANKER}")
    print(f"Pregunta : {pregunta}")
    print("\nCargando reranker...")

    reranker = cargar_reranker()

    resultados = rerankear_resultados(
        pregunta,
        resultados_prueba,
        reranker,
    )

    mostrar_reranking(resultados)


if __name__ == "__main__":
    main()