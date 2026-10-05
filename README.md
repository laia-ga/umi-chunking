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
