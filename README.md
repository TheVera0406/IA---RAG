# IA---RAG
Distintos sistemass RAG
## Como ejecutar los distintos sistemas RAG
Antes de ejecutar, los PDF deben estar en documentos/, preguntas.csv debe estar en la carpeta principal y el archivo .env debe contener una clave válida en OPENAI_API_KEY. Luego se activa el entorno virtual y se instalan las dependencias.

python -m pip install -r requirements.txt

## La verificación del entorno es opcional:
python verificar_entorno.py

## Construcción de la base de conocimiento:
python construir_base.py

python crear_chunks.py

python crear_embeddings.py

python crear_chromadb.py

python crear_indice_bm25.py **Este paso es solo para el rag hibrido y el rag hibrido+multi-query+reranking**

## Pruebas manuales opcionales:
python buscar_contexto.py

python generar_respuesta.py

## Procesamiento del archivo de preguntas y generación del resultado final:
python procesar_preguntas.py
