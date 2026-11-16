# Ground truth para evaluación RAG

Este repositorio contiene los datos, scripts y experimentos utilizados para
construir un ground truth médico y evaluar la recuperación de documentos JSON.
El ground truth sigue siendo experimental: las distintas versiones se
conservan para poder comparar cómo se construyeron y no deben interpretarse
como un benchmark final validado.

## Cómo evolucionó el ground truth

1. **`early_qa/`**: preguntas generadas por IA para los cuatro tipos de
   documentos. En una muestra de 1.500 preguntas se observó que algunas
   incluían la respuesta dentro de la propia pregunta; se conserva como
   antecedente, no como ground truth recomendado.
2. **Datasets de preguntas**: se descargaron MedMCQA, MedQA y S-MedQA y se
   prepararon conjuntos con especialidad. Los archivos fuente y el conjunto
   combinado están en `data/ground_truth/datasets/`.
3. **Selección por retrieval**: se usó chunking JSON, embeddings BGE y retrieval
   para puntuar preguntas. `4n_questions.jsonl` contiene 4.000 preguntas,
   1.000 por cada una de las cuatro categorías, seleccionadas por similitud.
4. **`397_questions.jsonl`**: contiene 1.588 preguntas (397 por categoría).
5. **`ground_truth_filtered.jsonl`**: contiene 10.548 preguntas y es el ground
   truth de la configuración general. Se generó con
   `scripts/ground_truth/filter_ground_truth.py`, que filtra preguntas según
   las especialidades/subespecialidades del CSV de categorías.
6. **Preguntas generadas desde documentos**: se selecciona una muestra
   reproducible de documentos JSON y se generan preguntas con un modelo
   compatible con la API de OpenAI. Esta prueba dio mejores resultados que
   los conjuntos anteriores y apunta a que la calidad/cobertura del ground
   truth era un factor importante. Sigue siendo una prueba exploratoria; aún
   faltan más experimentos y validación de las preguntas generadas.

## Estructura

```text
data/ground_truth/
  datasets/
    source/                 # MedMCQA, MedQA (no usado) y S-MedQA
    combined/               # conjunto combinado y tabla de especialidades
  experiments/
    early_qa/               # primeras preguntas generadas por IA
    retrieval_selection/    # 4n_questions.jsonl
    curated/                # snapshots de ground truth curados
  generated/
    selection/              # documentos elegidos y manifiesto de selección
    question_generation/    # preguntas generadas y exportación compartible
  metadata/                 # categorías y grupos de especialidades

scripts/
  ground_truth/              # preparación, filtrado, selección y generación
  run_indexing.py            # normalización, embeddings e indexación en Qdrant
  run_retrieval.py           # recuperación de chunks para las preguntas

chunking/                    # chunking jerárquico de documentos JSON
configs/                     # configuración de chunking, indexación y retrieval
data/raw/json/               # documentos fuente usados por chunking y generación
output/                      # resultados locales; no es fuente de ground truth
```

Los resultados de ejecución se escriben en `output/`. Los datasets y snapshots
de `data/ground_truth/` son los artefactos versionados para los experimentos.

## Scripts de ground truth

Todos se ejecutan desde la raíz del repositorio:

- MedQA está descargado, pero no se incorporó al ground truth: se consideró
  redundante porque S-MedQA ya se basa en MedQA y añade clasificación por
  especialidad.
- `scripts/ground_truth/merge_ground_truth.py`: combina los splits disponibles
  con respuestas de MedMCQA y S-MedQA en JSONL (MedMCQA dev y S-MedQA
  test/validation).
- `scripts/ground_truth/ground_truth_table.py`: cuenta preguntas por
  especialidad y subespecialidad y guarda un CSV junto al JSONL. Por ejemplo:
  `python scripts/ground_truth/ground_truth_table.py experiments/curated/397_questions.jsonl`.
- `scripts/ground_truth/select_top_retrieval_by_category.py`: selecciona el
  mismo número de preguntas para cada una de las cuatro categorías, priorizando
  mayor score de retrieval. Requiere `--retrieval`, `--categories` y `--output`;
  el umbral predeterminado es `0.60`.
- `scripts/ground_truth/filter_ground_truth.py`: filtra preguntas por las
  combinaciones de especialidad/subespecialidad en el CSV de categorías.
- `scripts/ground_truth/select_documents_for_question_generation.py`: selecciona
  documentos de forma reproducible. Por defecto utiliza cinco documentos por
  grupo, semilla `42` y el mapa de cuatro grupos de `metadata/`.
- `scripts/ground_truth/generate_questions_from_selected_documents.py`: genera
  preguntas a partir de la selección. Usa como máximo diez preguntas por
  documento de forma predeterminada.
- `scripts/ground_truth/info_medmcqa.py`,
  `scripts/ground_truth/specialty.py` y
  `scripts/ground_truth/download_medqa.py`: utilidades para inspeccionar o
  descargar los datasets fuente.

Ejemplos:

```bash
python scripts/ground_truth/merge_ground_truth.py
python scripts/ground_truth/filter_ground_truth.py
python scripts/ground_truth/select_top_retrieval_by_category.py \
  --retrieval output/retrieval/COLECCION/retrieval_bge_m3.json \
  --categories data/ground_truth/metadata/ground_truth_categories.csv \
  --output data/ground_truth/experiments/retrieval_selection/4n_questions.jsonl
```

### Regenerar preguntas desde documentos

La selección guarda el CSV y el manifiesto en
`data/ground_truth/generated/selection/`:

```bash
python scripts/ground_truth/select_documents_for_question_generation.py \
  --docs-per-specialty 5 \
  --question-total 200 \
  --seed 42
```

Probar el flujo sin invocar un modelo:

```bash
python scripts/ground_truth/generate_questions_from_selected_documents.py \
  --dry-run \
  --output output/ground_truth_dry_run/generated.jsonl \
  --shareable-output output/ground_truth_dry_run/shareable.jsonl
```

Para generar preguntas reales, configura `OPENAI_API_KEY` (y, opcionalmente,
`OPENAI_BASE_URL` y `OPENAI_MODEL`) en el entorno y ejecuta:

```bash
python scripts/ground_truth/generate_questions_from_selected_documents.py
```

Las salidas predeterminadas están en
`data/ground_truth/generated/question_generation/`. Se pueden cambiar con
`--output` y `--shareable-output`.

## Chunking, indexación y retrieval

Esta rama procesa documentos **JSON** de `data/raw/json/`; no incluye el
pipeline de chunking Markdown ni la evaluación automática de respuestas.
El chunker jerárquico detecta el tipo de documento (paper, guideline o ficha
técnica) y aplica su plan JSON correspondiente.

```bash
python -m chunking.run_JSON
python scripts/run_indexing.py --help
python scripts/run_retrieval.py --help
```

El chunking y sus métricas se guardan bajo `output/chunking/`. La indexación
lee el `chunk_fold` de `configs/indexing_config.json`, calcula embeddings y
usa Qdrant. Los ground truths que consume cada retrieval se indican en
`configs/retrieval*.json`; el general apunta a
`data/ground_truth/experiments/curated/ground_truth_filtered.jsonl`.

## Notas

- Los comandos con rutas relativas se deben ejecutar desde la raíz del repo.
- La selección aleatoria es reproducible con `--seed`; la generación mediante
  LLM puede variar entre ejecuciones.
- No se ha concluido que los resultados de retrieval midan únicamente el
  sistema RAG: la calidad, cobertura y respondibilidad de las preguntas también
  influyen y siguen pendientes de validación.
