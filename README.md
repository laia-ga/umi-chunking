# umi-chunking

## Ejecución del chunking

`chunking/run_MD.py` procesa las estrategias en orden y puede paralelizar los
documentos de cada estrategia con un número moderado de hilos:

```bash
python -m chunking.run_MD --workers 4 --gpu-id 0
```

`--workers` limita la concurrencia y `--gpu-id` selecciona una GPU CUDA
visible para las cinco estrategias que calculan embeddings. Si no se indica
GPU, esas estrategias usan CPU. Ambos valores también se pueden fijar en la
sección `execution` de `configs/chunker_config.json`; por defecto se usa un
worker y CPU. Cada hilo construye su propio tokenizer y chunker, así que al
usar GPU se carga una instancia del modelo por worker. En modo paralelo no se
registran métricas de RAM por documento, porque el RSS del proceso no permite
atribuir memoria a cada hilo; los tiempos por documento se mantienen. Las
estrategias basadas en LLM se mantienen secuenciales.

`chunking/run_JSON.py` también permite paralelizar documentos JSON de forma
limitada, manteniendo un worker por defecto:

```bash
python -m chunking.run_JSON --workers 4
```

Cada hilo carga su tokenizer privado y crea un `HierarchicalJSONChunker`
independiente para cada documento. En modo paralelo, las métricas de RAM por
documento no se informan porque el RSS del proceso no se puede atribuir con
fiabilidad a cada hilo; se mantienen los tiempos de chunking.

## Indexación con recursos configurables

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

## Retrieval concurrente

`scripts/run_retrieval.py` genera los embeddings de las preguntas por lotes
con una sola instancia del modelo (usa `cuda:0` si CUDA está disponible, y
CPU en caso contrario) y ejecuta hasta cuatro trabajos de pregunta en
paralelo contra Qdrant. Los valores predeterminados son `--batch-size 16` y
`--workers 4`; ambos pueden ajustarse:

```bash
python scripts/run_retrieval.py retrieval_paper_config.json \
  --batch-size 16 --workers 4
```

Cada embedding de pregunta se calcula una vez y se reutiliza para todas las
estrategias configuradas. Los resultados se recogen en el orden original de
las preguntas.

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
