# umi-chunking

## Puesta en marcha

### Requisitos e instalación

Se necesita Python 3.10 o superior. Crea un entorno virtual e instala las
dependencias desde la raíz del repositorio:

```bash
python -m venv .venv
```

Activa el entorno e instala los paquetes:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell (usa esta línea en lugar de la anterior)
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El proyecto descarga los tokenizadores y modelos de Hugging Face la primera
vez que se necesitan; hace falta acceso a internet durante esa descarga. El
judge GPT requiere además una clave de API de OpenAI. Para usar GPU con
PyTorch, instala una versión de PyTorch compatible con los controladores CUDA
del servidor siguiendo las instrucciones de PyTorch; la instalación genérica
de `requirements.txt` no fija una variante CUDA concreta. Chunking, indexación
y el judge local también pueden ejecutarse en CPU, aunque los modelos grandes
pueden ser lentos o necesitar mucha memoria.

### Iniciar Qdrant con Podman

El proyecto se conecta a Qdrant mediante su API HTTP. Podman sirve igual que
Docker para ejecutar el contenedor; por ejemplo, en el servidor:

```bash
podman volume create umi-qdrant-storage
podman run -d --name umi-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v umi-qdrant-storage:/qdrant/storage \
  docker.io/qdrant/qdrant
```

El volumen conserva la base de datos aunque se recree el contenedor. El puerto
se publica solo en `localhost`, que es apropiado cuando los scripts corren en
el mismo servidor. Comprueba que el contenedor está activo con
`podman ps`. Si ya existe, se puede iniciar con `podman start umi-qdrant`.
Para detenerlo, usa `podman stop umi-qdrant`.

Con Docker, los comandos equivalentes son `docker volume create` y
`docker run`; la imagen y la configuración de puertos/volumen son las mismas.
Qdrant no debe exponerse directamente a una red pública sin configurar
protecciones de acceso. Si los scripts se ejecutan en otra máquina o dentro
de otro contenedor, configura `qdrant.host` en
`configs/indexing_config.json` para que apunte a una dirección alcanzable y
limita el acceso al servicio a una red de confianza.

### Datos de entrada

Los documentos fuente van en `data/raw/markdown/` o `data/raw/json/`; ambos
flujos buscan recursivamente y solo consideran archivos de su formato.
Mantén los nombres de archivo estables para facilitar la identificación de
documentos y la reutilización de la caché.

Los Markdown pueden incluir metadatos YAML al principio. Para registrar el
tipo documental, usa `doc_type` con uno de los valores reconocidos:

```markdown
---
title: Ejemplo de guía
doc_type: guideline
---

# Ejemplo de guía
Contenido del documento...
```

En los JSON, la raíz debe ser un objeto. El chunker detecta fichas técnicas
por sus campos `medicine_name`, `composition`, `pharmaceutical_form`,
`indications` y `posology`; guías por `document_type` (`guideline` o
`clinical_guideline`), y artículos por el campo `article_type`. Después
aplica el plan estructural correspondiente, por lo que los campos del
documento deben seguir el esquema de ese tipo.

Las preguntas de Retrieval se guardan como JSONL en
`data/ground_truth/`: un objeto JSON por línea, con los tres campos
obligatorios `question_id`, `question` y `gold_answer`. Ejemplo:

```json
{"question_id":"q-001","question":"¿Cuál es el tratamiento recomendado?","gold_answer":"El tratamiento indicado en la guía."}
```

### Configuración y recorrido de extremo a extremo

| Archivo | Controla |
|---|---|
| `configs/chunker_config.json` | Estrategias Markdown habilitadas, parámetros y `chunk_fold` de salida. |
| `configs/tokenizer_config.json` | Tokenizador y límites de tokens usados por chunking. |
| `configs/indexing_config.json` | `chunk_fold` que se indexa, formatos, modelo de embeddings, conexión y colección Qdrant, lotes y recursos. |
| `configs/retrieval_config.json` | Modelo de embeddings para las preguntas, archivo de ground truth, `top_k` y filtros de Retrieval. |
| `configs/eval_config.json` | Fold de Retrieval, modelos/versiones y carpeta de salida de cada judge; judge elegido para métricas. |

`chunk_fold` debe señalar la misma ejecución de chunking en las configuraciones
de chunking e indexación. El modelo de embeddings y su normalización deben
coincidir entre indexación y Retrieval; el nombre de la colección y la
conexión de Qdrant se toman de `indexing_config.json`.

La configuración predeterminada usa el conjunto definitivo
`data/ground_truth/4n_questions.jsonl`. Para probar otro conjunto, cambia
`retrieval.ground_truth_file` por una ruta relativa a `data/ground_truth/`.

La evaluación usa `fold` en `configs/eval_config.json` para localizar los
resultados de Retrieval. Cada bloque de judge también define su
`output_folder`, que se usa para sus resultados y para las métricas y gráficas
cuando ese judge está seleccionado. Así se pueden mantener las salidas de
distintos modelos separadas. `metrics_judge` debe ser `local` o `gpt` y
selecciona tanto el CSV que procesan las métricas como la carpeta que usan las
gráficas. El modelo y la versión activos se leen del JSON; si falta una clave
o no se puede leer la configuración, la ejecución falla en lugar de usar un
valor alternativo del código. Esta configuración no depende de la colección
indicada en `indexing_config.json`.

Para usar el judge GPT, crea un archivo `.env` en la raíz del repositorio con
la clave de API:

```dotenv
OPENAI_API_KEY=tu_clave
```

No compartas ni subas este archivo; `.env` está excluido de Git.

Una ejecución completa sigue este orden:

1. Coloca los documentos fuente y las preguntas en las carpetas indicadas.
2. Ejecuta `python -m chunking.run_MD` y/o
   `python -m chunking.run_JSON` para producir chunks.
3. Inicia Qdrant y ejecuta `python scripts/run_indexing.py`.
4. Comprueba que `retrieval.ground_truth_file` en
   `configs/retrieval_config.json` apunta a un JSONL existente y ejecuta
   `python -m scripts.run_retrieval`.
5. Ejecuta uno de los judges sobre el JSON generado, después `run_metrics.py`
   sobre el CSV y finalmente `run_graphics.py` sobre el Excel.

Los artefactos generados se guardan bajo `output/`, que está excluido de Git;
los modelos descargados y las cachés tampoco forman parte de los resultados
versionados.

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
   vuelve a calcular el vector. Los nuevos embeddings se guardan en lotes
   durante el cálculo, por defecto cada 1.024 chunks; si el proceso se
   interrumpe, la siguiente ejecución reutiliza los lotes ya guardados y
   calcula solo los pendientes.
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
Qdrant (`batch_size` o `--batch-size`) y del lote de persistencia de la caché
(`indexing.cache_batch_size` o `--cache-batch-size`, por defecto 1.024). Reducir
el lote de caché limita cuánto trabajo de embeddings se pierde si se
interrumpe el proceso; no cambia el batch que se ejecuta en cada GPU.

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

La configuración predeterminada consulta `BAAI/bge-m3`, recupera los diez
primeros resultados (`top_k: 10`) y no aplica filtros por estrategia, tipo de
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
`output/retrieval/<colección>/<archivo>`, por ejemplo
`output/retrieval/all_bge_m3/retrieval_bge_m3.json`. El archivo se reescribe
al ejecutar de nuevo una configuración que genera el mismo nombre. Para que
los jueces encuentren este resultado, `fold` en `eval_config.json` debe
coincidir con el nombre de esa carpeta.

## Judge: evaluación de relevancia

Los scripts `eval/run_judge_qa_local.py` y `eval/run_judge_qa_gpt.py` evalúan
cada chunk recuperado comparándolo con la pregunta y su respuesta de
referencia (`gold_answer`). Ambos usan la misma rúbrica:

- **2**: el chunk contiene la respuesta o información suficiente para
  responder correctamente.
- **1**: contiene información útil o parte de la respuesta, pero es
  incompleta.
- **0**: no contiene información útil para responder.

El judge no genera una nueva respuesta: asigna una puntuación de relevancia
por chunk. Lee el JSON producido por Retrieval desde
`output/retrieval/<fold>/` y guarda el CSV bajo
`output/judge/<output_folder>/`. Si el resultado de Retrieval contiene una
lista global `retrieved_chunks`, evalúa
esa lista; si contiene resultados separados por `strategies`, evalúa los
chunks de cada estrategia por separado. Las preguntas sin identificador,
texto o respuesta de referencia se omiten. `--top-k` permite evaluar solo
los primeros chunks de cada lista; si se omite, se evalúan todos los que
estén en el JSON de Retrieval.

### Judge local

`eval/run_judge_qa_local.py` ejecuta el modelo definido en
`local_judge.model_name` en `configs/eval_config.json`. Por defecto
selecciona `cuda:0` si CUDA está disponible y, si no, CPU. Comprueba los
dispositivos y carga el modelo antes de empezar a evaluar, de modo que los
errores de carga se detecten antes de modificar el CSV de resultados:

```bash
python eval/run_judge_qa_local.py --gpu-ids 0 --batch-size 1
```

`--gpu-ids` acepta uno o varios índices CUDA visibles, por ejemplo
`--gpu-ids 0,1`; cada GPU carga una copia completa del modelo y procesa
batches de chunks distintos. Cada copia usa la memoria de su propia GPU.
`--cpu` fuerza CPU y no se puede combinar con `--gpu-ids`. `--batch-size`
controla los chunks procesados por llamada de generación en cada worker; el
valor conservador es 1 y batches mayores requieren más memoria. Se puede
limitar el trabajo a los primeros resultados con `--top-k`.

Por defecto el juez guarda solo la puntuación, lo que reduce la generación.
`--include-reason` solicita y guarda además una justificación breve. Los
resultados se escriben en
`output/judge/<output_folder>/local_judge_<sufijo_del_retrieval>.csv`.
`output_folder` se toma del bloque `local_judge` en `eval_config.json`.

Si el CSV ya existe, la ejecución conserva las filas correctas y vuelve a
intentar las filas fallidas o pendientes. `--restart` descarta el CSV y
empieza desde cero. El script comprueba que la versión de la rúbrica y la
opción `--include-reason` coincidan con las del CSV; si no coinciden, exige
`--restart` para evitar mezclar resultados incompatibles. Los errores de
inferencia quedan registrados y se pueden reintentar en una ejecución
posterior.

### Judge GPT

`eval/run_judge_qa_gpt.py` aplica la misma rúbrica mediante la API de OpenAI,
usando Structured Outputs para restringir la respuesta a un `score` de 0, 1 o
2. No produce una justificación. Antes de ejecutarlo, configura
`OPENAI_API_KEY` (o `OPENAI_KEY`) en el entorno o en un archivo `.env` en la
raíz del proyecto.

```bash
python eval/run_judge_qa_gpt.py \
  --workers 4 --top-k 5
```

El script usa el modelo definido en `gpt_judge.model_name` en
`configs/eval_config.json`. `--workers` controla las llamadas a la API en
paralelo (por defecto, 1); se puede aumentar si los límites de concurrencia
y solicitudes lo permiten. Ante errores transitorios, como límites de
solicitudes o problemas de red, reintenta hasta cinco veces con espera
creciente. Los errores de autenticación o cuota detienen la ejecución porque
no se resuelven reintentando. `--top-k` limita cuántos chunks evalúa por
pregunta y estrategia.

Los resultados se guardan en
`output/judge/<output_folder>/gpt_judge_<sufijo_del_retrieval>.csv`, donde
`output_folder` se toma del bloque `gpt_judge` en `eval_config.json`. Se
guardan a medida que terminan las evaluaciones: al reanudar, se conservan
las filas que tienen puntuación y se repiten las que fallaron o quedaron
pendientes. Si cambia `gpt_judge.version`, usa `--restart` para recalcular
el CSV. `--restart` también descarta el CSV anterior y vuelve a evaluar todo.

## Métricas de evaluación

`eval/run_metrics.py` resume las puntuaciones del judge seleccionado por
`metrics_judge` en `eval_config.json` y guarda el Excel en
`output/metrics/<output_folder>/`:

```bash
python eval/run_metrics.py
```

Para calcular métricas del otro juez, cambia `metrics_judge` en
`eval_config.json` a `local` o `gpt` y ejecuta de nuevo el comando. Si se
indica un `judge_file` posicional, se usa ese CSV dentro del
`output_folder` del juez seleccionado.

El script calcula **Mean nDCG**, **HitRate** y **MRR** tanto en los cortes
`@5` como `@10`. HitRate y MRR consideran suficiente solo un chunk con score
`2`; nDCG usa las ganancias graduadas `2^score - 1` y normaliza cada ranking
contra el ideal obtenido al ordenar de mayor a menor las relevancias juzgadas
disponibles para esa pregunta. Si ninguna relevancia disponible es mayor que
cero, nDCG es `0`. Los porcentajes **%score2@10**, **%score1@10** y
**%score0@10** describen la distribución entre los diez primeros chunks.

Las métricas se calculan por pregunta dentro de cada estrategia y luego se
promedian. Si una pregunta presente en el CSV no tiene chunks para una
estrategia, cuenta como cero para las métricas de ranking. Si una pregunta
tiene algún error de evaluación entre los primeros `k` resultados, se excluye
del promedio de ese corte (`@k`), pero no necesariamente del otro: por ejemplo,
un error en el rango 8 excluye la pregunta de `@10`, no de `@5`. Los errores y
las preguntas excluidas se cuentan por separado en `n_judge_errors@5/@10` y
`n_questions_excluded@5/@10`; `n_questions_evaluated@5/@10` indica cuántas
preguntas entraron en cada promedio. Las distribuciones de scores @10 usan
solo evaluaciones válidas. El Excel contiene una hoja `summary` con una fila
por estrategia y los recuentos correspondientes.

El archivo se guarda en `output/metrics/<output_folder>/` y su nombre se
deriva del CSV de entrada; por ejemplo, `gpt_judge_bge_m3.csv` produce
`metrics_gpt_judge_bge_m3.xlsx`. Si el judge incluye varios tipos
documentales, el resumen no separa las filas por tipo documental.

## Gráficas de resultados

`eval/run_graphics.py` genera visualizaciones a partir de la hoja `summary`
de un Excel de métricas. Produce un heatmap con las métricas @5 y @10 y la
distribución de scores @10, además de un dotplot con paneles separados para
comparar las métricas de cada corte. Las estrategias se ordenan por
Mean nDCG@10:

```bash
python eval/run_graphics.py
```

El script usa el Excel correspondiente al `metrics_judge` seleccionado en
`eval_config.json`. Si se indica un `metrics_file` posicional, se usa ese
Excel dentro del `output_folder` de ese juez. Las imágenes PNG se guardan en
`output/graphics/<output_folder>/` con resolución de 300 dpi. Sus nombres se
derivan del archivo de métricas seleccionado.

### Flujo completo de evaluación

El flujo habitual, partiendo de un archivo de resultados de Retrieval, es:

1. Evalúa los chunks con uno de los jueces, local o GPT, para crear un CSV.
2. Ejecuta `run_metrics.py` sobre ese CSV para calcular las métricas y crear
   el Excel.
3. Ejecuta `run_graphics.py` sobre el Excel para generar las imágenes.

Los scripts de `eval/` quedan así cubiertos: los dos jueces, el cálculo de
métricas y la generación de gráficas. La colección de Qdrant y los resultados
de Retrieval son entradas de las etapas anteriores; estos pasos de evaluación
trabajan sobre el JSON de Retrieval y los CSV/Excel generados en cada fase.
