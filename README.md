# umi-chunking

## Chunking de documentos

El proyecto incluye dos flujos de chunking, según el formato de entrada:

| Flujo | Script | Entradas | Enfoque |
|---|---|---|---|
| Markdown | `chunking/run_MD.py` | Archivos `.md` y `.markdown` | Ejecuta las estrategias habilitadas en la configuración para comparar distintos métodos sobre cada documento. |
| JSON | `chunking/run_JSON.py` | Archivos `.json` estructurados | Detecta el tipo documental y aplica un plan jerárquico específico para ese tipo. |

Los dos buscan los documentos recursivamente bajo `data/raw/markdown` y
`data/raw/json`, respectivamente. Los límites y parámetros usados para contar
tokens se configuran en `configs/tokenizer_config.json`. Las estrategias y
sus parámetros se configuran en `configs/chunker_config.json`.

### Chunking Markdown

El script procesa por separado cada estrategia activada en
`configs/chunker_config.json`. Para cada archivo, detecta `doc_type` en el
frontmatter YAML: admite `paper`, `guideline` y `clinical_guideline` (también
`guia_clinica`). El contenido completo se entrega a cada método; el tipo
documental queda registrado como metadato.

Las estrategias disponibles se organizan por enfoque:

**Basadas en reglas y estructura textual**

- `FixedCharacterChunker`: fragmentos de longitud fija en caracteres, con
  solapamiento opcional.
- `FixedTokenChunker`: fragmentos de un máximo fijo de tokens, con
  solapamiento configurable.
- `OverlappingTokenChunker`: ventanas de tokens con una cantidad configurable
  de tokens compartidos entre fragmentos.
- `SlidingWindowTokenChunker`: recorre el texto con una ventana de tokens y
  un paso configurables.
- `SentenceBasedChunker` y `SentenceGroupChunker`: separan por frases; el
  segundo agrupa una cantidad configurable de frases y permite solapamiento.
- `ParagraphBasedChunker` y `ParagraphGroupChunker`: separan por párrafos o
  agrupan varios párrafos con solapamiento opcional.
- `LengthAwareChunker`: agrupa frases completas intentando acercarse a una
  longitud objetivo en tokens dentro de una tolerancia.

**Recursivas y jerárquicas**

- `RecursiveChunker`: divide recursivamente, priorizando separadores
  estructurales para conservar párrafos y frases siempre que sea posible.
- `RecursiveTokenChunker`: aplica un enfoque recursivo con límites medidos en
  tokens.
- `ParentChildChunker`: produce fragmentos padre con más contexto y
  fragmentos hijo más pequeños asociados a ellos.

**Semánticas basadas en embeddings**

- `SemanticEmbeddingChunker`: detecta cambios de significado entre grupos de
  frases consecutivas y corta en los límites semánticos detectados.
- `SemanticSimilarityThresholdChunker`: agrupa frases consecutivas mientras
  su similitud supere el umbral configurado.
- `SemanticBoundaryChunker`: detecta rupturas semánticas entre frases
  consecutivas mediante cambios en la similitud.
- `TopicBasedChunker`: agrupa frases mediante clustering jerárquico por
  similitud semántica; las frases de un mismo grupo pueden no ser contiguas
  en el documento original.

**Tamaño adaptativo**

- `DynamicTokenSizeChunker`: ajusta los límites de cada fragmento dentro de
  un rango de tokens y prioriza finales de frase o saltos de línea.
- `ContentDensityAdaptiveChunker`: modifica el tamaño según la densidad
  léxica; tiende a generar fragmentos menores para texto más denso.
- `SemanticVarianceAdaptiveChunker`: adapta los límites según la variación
  semántica local entre frases.

**Basadas en un LLM**

- `LLMBoundaryDetectionChunker`: detecta límites semánticos con ayuda de un
  modelo de lenguaje y después forma los fragmentos.
- `LLMSegmentThenChunker`: primero segmenta el documento en unidades
  semánticas y luego las ajusta al tamaño configurado.

Ambas estrategias LLM están activadas en la configuración actual y usan
`Qwen/Qwen2.5-0.5B-Instruct` de forma local. `LLMSegmentThenChunker` conserva
primero la segmentación por párrafos y solo pide al modelo que refine los
párrafos que superan `llm_refine_threshold_tokens` (512 por defecto). Los
párrafos menores pasan directamente a la etapa de chunking; además, el modelo
se carga de forma diferida, solo si algún segmento necesita refinamiento.
Estos ajustes reducen llamadas y evitan cargar el modelo cuando no se usa.
Todos los parámetros —incluido el umbral— se pueden modificar en
`configs/chunker_config.json`.

Ejemplo de ejecución:

```bash
python -m chunking.run_MD --workers 4 --gpu-id 0
```

`--workers` limita cuántos documentos de una estrategia se procesan en
paralelo; también se puede fijar `execution.workers` en la configuración.
Las estrategias que calculan embeddings usan CPU si no se indica GPU. Con
`--gpu-id` se selecciona una GPU CUDA visible para los chunkers semánticos y
los LLM; también se puede establecer `execution.gpu_id`. Si no se indica, los
LLM usan `cuda:0` cuando CUDA está disponible y CPU en caso contrario. Las
estrategias basadas en LLM se ejecutan secuencialmente para no cargar varias
copias del modelo a la vez. Las estrategias paralelizadas con varios workers
construyen su propio tokenizer y chunker, por lo que usar GPU con ellas carga
una instancia del modelo por hilo. En modo paralelo no se informan métricas de
RAM por documento, ya que el uso de memoria del proceso no se puede atribuir
de forma fiable a cada hilo.

### Chunking JSON

`chunking/run_JSON.py` usa `HierarchicalJSONChunker`, no la colección de
estrategias Markdown. Cada entrada debe ser un objeto JSON y su tipo se
determina de la siguiente manera:

- **Ficha técnica**: reconoce la presencia de `medicine_name`, `composition`,
  `pharmaceutical_form`, `indications` y `posology`.
- **Guía clínica**: reconoce `document_type` con valor `guideline` o
  `clinical_guideline`.
- **Artículo (paper)**: reconoce el campo `article_type`.

Una vez detectado el tipo, el script selecciona el plan jerárquico
correspondiente (`DOCUMENT_PLAN_FC`, `DOCUMENT_PLAN_GL` o
`DOCUMENT_PLAN_PAPER`). El plan define qué campos se agrupan y la jerarquía
de secciones. El chunker conserva las rutas de origen del JSON, los niveles
y grupos semánticos, y evita dividir estructuras configuradas como
indivisibles. También registra el recuento de tokens y si un fragmento supera
los límites objetivo o máximo. Los límites `target_tokens` y `max_tokens`
se leen de `configs/tokenizer_config.json`.

`target_tokens` orienta la división jerárquica: si una sección supera ese
tamaño, el plan intenta subdividirla siguiendo su estructura. Algunas
unidades —por ejemplo, campos escalares, diccionarios y listas marcadas como
indivisibles— pueden superar el objetivo para conservar su significado y
estructura. Después de esa división, cualquier chunk que exceda `max_tokens`
pasa por un fallback para ajustarlo al límite máximo. El fallback intenta
cortar por oraciones y equilibrar el tamaño usando el menor número de partes
posible; añade un pequeño solapamiento entre partes para mantener contexto.
Si una oración es demasiado larga, la divide por palabras y, como último
recurso, por caracteres. No elimina contenido y marca en los metadatos los
chunks que se generaron con este procedimiento.

Ejemplo:

```bash
python -m chunking.run_JSON --workers 4
```

El valor predeterminado es un worker. Cada hilo carga su propio tokenizer y
procesa los documentos con una instancia independiente del chunker. Igual que
en Markdown, las métricas de RAM por documento no se informan en modo
paralelo.

### Resultados, caché incremental y resumen

Cada flujo escribe sus resultados en una subcarpeta de ejecución cuyo nombre
se toma de `chunk_fold` en `configs/chunker_config.json`:

- JSON: `output/chunks/<chunk_fold>/json/`.
- Markdown: `output/chunks/<chunk_fold>/markdown/`.

Cada subcarpeta de formato contiene:

- `chunk_results/`: un JSON por documento y estrategia (Markdown) o por
  documento (JSON), con los fragmentos y sus metadatos.
- `metrics/`: métricas individuales en JSON.
- `summary.csv`: resumen de los resultados, con una fila por combinación de
  documento y estrategia en Markdown, y una fila por documento en JSON.

Por defecto, ambos scripts validan una caché incremental a partir del
contenido y de la configuración relevante del chunker. Si hay resultados
completos y coincidentes, reutilizan los chunks y las métricas sin ejecutar
otra vez el chunker. Los documentos nuevos, modificados, con resultados
incompletos o afectados por cambios de configuración se procesan de nuevo.
Al reprocesar un documento presente, se actualizan sus filas del resumen. Si
un documento deja de estar en la carpeta de entrada, sus archivos de salida
y su fila en `summary.csv` se conservan; no se eliminan automáticamente.

Para ignorar la caché y volver a procesar todos los documentos disponibles,
se puede usar `--no-cache`:

```bash
python -m chunking.run_MD --no-cache
python -m chunking.run_JSON --no-cache
```

## Indexación en Qdrant

La indexación toma los chunks ya generados por los flujos Markdown y JSON,
los normaliza en un formato común, calcula sus embeddings y los guarda en una
colección de Qdrant. No ejecuta el chunking: antes, hay que generar los
resultados con `chunking/run_MD.py` y/o `chunking/run_JSON.py`.

El proceso se controla desde `scripts/run_indexing.py` y, por defecto, ejecuta
todas las fases:

1. **Lee los resultados del chunking.** Busca archivos JSON en
   `output/chunks/<chunk_fold>/`, donde `chunk_fold` se define en
   `configs/indexing_config.json`. Cada archivo aporta los chunks de un
   documento y sus metadatos.
2. **Normaliza los chunks.** Convierte los resultados de Markdown y JSON a un
   esquema común y escribe un registro JSON por línea en
   `output/indexing/normalized_chunks.jsonl`. Se conservan, entre otros datos,
   el texto, el formato, el documento, la estrategia de chunking y los
   metadatos estructurales. Los identificadores normalizados distinguen
   formato, documento, estrategia y posición del chunk.
3. **Selecciona los formatos configurados.** Solo los formatos incluidos en
   `indexing.formats` de `configs/indexing_config.json` pasan a la generación
   de embeddings. Por ejemplo, se puede indexar únicamente `json` o combinar
   `json` y `markdown`.
4. **Calcula los embeddings.** Usa el modelo de `embeddings.model_name` y la
   normalización indicada en `embeddings.normalize_embeddings`. Por defecto,
   la configuración actual usa `BAAI/bge-m3` y normaliza los vectores.
   Reutiliza por defecto la caché de
   `output/indexing/embeddings_cache/`: cada entrada está asociada al modelo,
   al identificador del chunk y al hash de su texto; si cambia el texto, se
   vuelve a calcular el vector.
5. **Recrea la colección y carga los vectores.** Conecta con Qdrant usando el
   host y puerto de `qdrant` (también se pueden sobrescribir con
   `QDRANT_HOST` y `QDRANT_PORT`), elimina la colección configurada si ya
   existe y la crea de nuevo con distancia coseno. Después inserta los
   vectores por lotes. Cada punto incluye el texto y los metadatos del chunk,
   para que Retrieval pueda aplicar filtros y devolver el contexto.

> **Importante:** cada ejecución que llega a la fase de indexación elimina y
> vuelve a crear la colección configurada. No es una actualización incremental
> de Qdrant; se vuelve a cargar el conjunto de chunks seleccionado. La caché
> evita recalcular embeddings cuando no han cambiado, pero no evita la
> reconstrucción de la colección.

### Ejecución

Para normalizar y después indexar en una sola ejecución:

```bash
python scripts/run_indexing.py
```

Las fases también se pueden ejecutar por separado:

```bash
# Genera normalized_chunks.jsonl sin conectarse a Qdrant.
python scripts/run_indexing.py --normalize-only

# Indexa el JSONL existente sin volver a leer los resultados del chunking.
python scripts/run_indexing.py --index-only
```

`--index-only` usa el archivo normalizado que ya existe; si los resultados del
chunking han cambiado, ejecuta primero la normalización para evitar indexar
datos desactualizados. Para recalcular los embeddings en lugar de reutilizar
la caché:

```bash
python scripts/run_indexing.py --no-cache
```

Los tiempos, el número de chunks y las estadísticas de caché se añaden a
`output/indexing/logs/run_log.jsonl`.

### Indexación con recursos configurables

`scripts/run_indexing.py` usa CPU por defecto (`num_gpus: 0`, un worker y un
hilo por worker). Para repartir embeddings entre GPUs, configura `num_gpus` y,
opcionalmente, los índices CUDA visibles en `gpu_ids` dentro de
`configs/indexing_config.json`. Cada GPU seleccionada ejecuta un worker; el
tamaño de lote de embeddings (`embedding_batch_size` en configuración o
`--embedding-batch-size`) se puede configurar independientemente del upsert a
Qdrant (`batch_size` o `--batch-size`).

Por ejemplo, para usar las GPUs visibles 0 y 2 con lotes moderados:

```bash
python scripts/run_indexing.py --num-gpus 2 --gpu-ids 0,2 \
  --embedding-batch-size 16 --batch-size 32 --cpu-threads 2
```

Los IDs son índices CUDA visibles dentro del proceso/contenedor (y pueden
verse afectados por `CUDA_VISIBLE_DEVICES`). Para limitar el uso de CPU se
pueden ajustar `--workers` y `--cpu-threads`; los valores también se pueden
fijar en `indexing.num_workers` e `indexing.cpu_threads`. Para usar las GPUs
CUDA 0 y 1 según la configuración, establece `num_gpus: 2` y deje `gpu_ids`
vacío. La selección explícita de GPUs falla con un error si CUDA o algún ID no
están disponibles.

## Retrieval en Qdrant

`scripts/run_retrieval.py` busca chunks relevantes para un conjunto de
preguntas y guarda los resultados para su evaluación. Requiere que Qdrant esté
en ejecución, que la colección configurada en
`configs/indexing_config.json` ya se haya creado mediante la indexación y que
el modelo de embeddings de Retrieval sea el mismo que se usó para indexar.

Por defecto, el proceso sigue estos pasos:

1. **Carga la configuración y las preguntas.** Usa
   `configs/retrieval_config.json` y lee el JSONL indicado en
   `retrieval.ground_truth_file`, relativo a `data/ground_truth/`. Cada línea
   debe ser un objeto JSON con `question_id`, `question` y `gold_answer`.
2. **Genera los embeddings de las preguntas.** Usa el modelo y la opción de
   normalización de `embeddings` en la configuración de Retrieval. Los textos
   se procesan por lotes y cada pregunta se vectoriza una sola vez, incluso si
   se van a buscar varias estrategias. El modelo se carga en `cuda:0` si CUDA
   está disponible y, en caso contrario, en CPU. Si se usa un modelo
   `intfloat/multilingual-e5`, antepone `query: ` a cada pregunta.
3. **Busca en la colección de Qdrant.** Para cada pregunta recupera hasta
   `retrieval.top_k` chunks más similares. Los filtros opcionales
   `retrieval.document_type` y `retrieval.format` restringen cada búsqueda.
   Si `retrieval.strategies` contiene una lista, se hace una búsqueda
   independiente por estrategia y se guardan los resultados separados; si es
   `null`, busca sin filtrar por estrategia.
4. **Guarda los resultados.** Escribe un JSON en `output/retrieval/` con las
   preguntas, las respuestas de referencia y los chunks recuperados. Cada
   resultado incluye el rango, la puntuación de similitud, los identificadores
   y metadatos principales del chunk y su texto. Las respuestas se guardan en
   el orden original de las preguntas.

La configuración predeterminada consulta `BAAI/bge-m3`, recupera los cinco
primeros resultados (`top_k: 5`) y no aplica filtros por estrategia, tipo de
documento ni formato. Por tanto, cada pregunta busca en todos los puntos de
la colección seleccionada. Para cambiar estos valores, edita
`configs/retrieval_config.json`; el host, puerto y nombre de la colección
proceden de `configs/indexing_config.json`.

### Ejecución y concurrencia

```bash
python -m scripts.run_retrieval --batch-size 16 --workers 4
```

El primer argumento posicional opcional permite indicar otro archivo JSON de
configuración, ubicado dentro de `configs/`; por defecto se usa
`retrieval_config.json`.

`--batch-size` controla cuántas preguntas se envían juntas al modelo para
generar embeddings; `--workers` limita las preguntas consultadas en paralelo a
Qdrant. Ambos valores deben ser enteros positivos. La concurrencia afecta a
las consultas, no a la carga del modelo: se usa una única instancia y el
embedding de cada pregunta se reutiliza para todas las estrategias
configuradas.

Los resultados se guardan en
`output/retrieval/retrieval_<modelo>[_<tipo_de_documento>].json` (por ejemplo,
`retrieval_bge_m3.json`). El archivo se reescribe al ejecutar de nuevo una
configuración que genera el mismo nombre.

## Judge local

`eval/run_judge_qa_local.py` evalúa los chunks con `Qwen/Qwen3-8B`. Por
defecto usa una sola copia del modelo en `cuda:0` y batch 1; si CUDA no está
disponible, usa CPU. Las GPUs solicitadas se comprueban al inicio y el modelo
se carga antes de evaluar para detectar problemas de memoria sin descartar
resultados anteriores:

```bash
python eval/run_judge_qa_local.py retrieval_paper.json \
  --gpu-ids 0 --batch-size 1
```

Se pueden seleccionar varias GPUs explícitamente, por ejemplo
`--gpu-ids 0,1`. Cada una carga una copia completa e independiente del modelo
y juzga batches de chunks distintos; la VRAM no se comparte entre GPUs. Batch
1 es la opción conservadora; si se aumenta `--batch-size`, hay que vigilar la
memoria disponible. Los IDs son los CUDA visibles dentro del proceso o
contenedor. `--cpu` fuerza la ejecución en CPU. La rúbrica local es
la misma que la del juez GPT; por defecto no solicita `reason` para limitar
la generación, pero se puede activar con `--include-reason`.

Si el CSV ya existe, el script conserva los juicios correctos y reintenta en
la siguiente ejecución las filas con error o pendientes. `--restart` descarta
el CSV previo. Si el CSV viene de una versión anterior del juez o cambia la
opción `--include-reason`, el script pide `--restart` para evitar mezclar
resultados de rúbricas distintas. Los errores de inferencia se registran; no
se reintentan automáticamente dentro de la misma ejecución.
