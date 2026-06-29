# 🚀 GUÍA: IMPLEMENTAR RAG HÍBRIDO EN TU PROYECTO

## ¿Qué es el RAG Híbrido?

Tu RAG actual usa **solo embeddings** (similitud semántica):
```
Pregunta → Embedding → ChromaDB → Chunks similares
```

El RAG **Híbrido** combina **dos métodos**:
```
Pregunta → 
         ├─ Embedding (semántica)  ─┐
         └─ BM25 (palabras clave) ──┼→ Combinar scores → Mejores chunks
```

**Ventaja**: Recupera documentos relevantes tanto por significado como por términos exactos.

---

## 📋 PASOS PARA IMPLEMENTAR

### OPCIÓN 1:
Seguir los mismos pasos de la guia del RAG basico solo cambiar que hay que colocar buscar_contexto_hibrido.py generar_respuesta_compatible.py procesar_preguntas_compatible.py

## 🔍 EJEMPLO DE USO INTERACTIVO

```bash
$ python buscar_contexto_hibrido.py

🚀 ETAPA C - BÚSQUEDA HÍBRIDA DE CONTEXTO
================================================================================

✨ VERSIÓN MEJORADA CON RETRIEVAL HÍBRIDO
   • Embeddings: 50%
   • BM25: 50%
   • TOP-K: 5

❓ Escribe una pregunta o 'salir': ¿Qué es la obesidad?

🔍 Búsqueda híbrida (E:50.0% + B:50.0%)
  1️⃣  Buscando por embeddings (semántica)... ✓ (2156 resultados)
  2️⃣  Buscando por BM25 (palabras clave)... ✓ (2156 resultados)
  3️⃣  Combinando scores... ✓
  4️⃣  Ordenando y recuperando TOP-5... ✓

📊 CHUNKS RECUPERADOS (HÍBRIDO)
================================================================================

📌 RESULTADO #1
   ID              : chunk_000456
   Documento       : obesidad_definicion.pdf
   Página          : 1
   Chunk en página : 2

   🎯 SCORES:
      Híbrido      : 0.8923 ⭐
      Embedding    : 0.8456 (semántica)
      BM25         : 0.9390 (léxica)

   📄 Texto: La obesidad es una enfermedad crónica caracterizada por un 
   exceso de grasa corporal...
```


## 🎓 CONCEPTO: ¿Por qué funciona mejor?

**Ejemplo práctico**:

Pregunta: "¿IMC significa qué?"

### Vectorial puro (solo embeddings):
```
❌ Puede NO encontrar documento con título "Índice de Masa Corporal"
   porque "IMC" y "Índice de Masa Corporal" son muy diferentes textualmente
✓ Pero SÍ captura el significado semántico
```

### BM25 puro:
```
✓ Encuentra "IMC" como término exacto
✓ Rankea bien documentos que mencionan "IMC" explícitamente
❌ Pero puede no entender sinónimos o paráfrasis
```

### Híbrido (⭐ Lo mejor):
```
✓ Embedding captura: "IMC" ≈ "Índice de Masa Corporal" (semántica)
✓ BM25 captura: "IMC" exactamente en el texto
✓ Combina ambos → Recupera documentos relevantes con máxima precisión
```