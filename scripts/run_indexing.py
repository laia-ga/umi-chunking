# ============================================================
# INDEXACIÓN
# ============================================================

"""
Normaliza e indexa los chunks generados por los runners.

El script convierte los outputs JSON de Markdown y JSON a un formato común
JSONL y, después, genera embeddings e inserta los chunks configurados en Qdrant.
Cada fase puede ejecutarse por separado mediante los argumentos de la línea
de comandos.
"""

# ============================================================
# IMPORTS GENERALES
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ============================================================
# RUTAS
# ============================================================

# Carpeta donde se encuentra este script:
# main/scripts
SCRIPT_DIR = Path(__file__).resolve().parent

# Carpeta raíz del proyecto:
# main
PROJECT_ROOT = SCRIPT_DIR.parent

# Carpeta que contiene los chunks generados por los runners:
# main/output/chunks
# CHUNKS_ROOT = PROJECT_ROOT / "output" / "chunks"
CHUNKS_ROOT = PROJECT_ROOT / "output" / "test"

# Carpeta general de salida de la indexación:
# main/output/indexing
INDEXING_OUTPUT_DIR = PROJECT_ROOT / "output" / "indexing"

# Archivo normalizado, con un registro JSON por línea:
# main/output/indexing/normalized_chunks.jsonl
NORMALIZED_OUTPUT_FILE = INDEXING_OUTPUT_DIR / "normalized_chunks.jsonl"

# Carpeta donde se guarda la caché de embeddings:
# main/output/indexing/embeddings_cache
EMBEDDINGS_CACHE_DIR = INDEXING_OUTPUT_DIR / "embeddings_cache"

# Log de tiempos de cada ejecución, un JSON por línea:
# main/output/indexing/logs/run_log.jsonl
RUN_LOG_FILE = INDEXING_OUTPUT_DIR / "logs" / "run_log.jsonl"

# Archivo de configuración de la indexación:
# main/configs/indexing_config.json
INDEXING_CONFIG_FILE = PROJECT_ROOT / "configs" / "indexing_config.json"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def load_chunk_file(chunk_file: Path) -> Dict[str, Any]:
    """
    Carga y valida un archivo JSON generado por un runner de chunking.

    Parameters
    ----------
    chunk_file:
        Ruta del archivo JSON que contiene los chunks de un documento.

    Returns
    -------
    Dict[str, Any]
        Datos completos del archivo, incluyendo su lista de chunks.
    """

    with chunk_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"El archivo de chunks debe contener un objeto JSON: {chunk_file}"
        )

    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(
            f"El archivo no contiene una lista 'chunks': {chunk_file}"
        )

    return data


def infer_document_type(document: Dict[str, Any]) -> str:
    """
    Obtiene la tipología documental guardada por los runners.

    Si el runner no ha guardado el tipo documental, devuelve ``unknown``
    para mantener el registro normalizado completo.
    """

    document_type = document.get("document_type")
    if isinstance(document_type, str) and document_type:
        return document_type
    return "unknown"


def normalize_json_chunk(
    chunk: Dict[str, Any],
    document: Dict[str, Any],
    chunk_index: int,
    source_file: Path,
) -> Dict[str, Any]:
    """
    Convierte un chunk JSON al esquema común de indexación.

    Conserva en ``metadata`` la información estructural específica del JSON,
    como el grupo documental, el nivel y las rutas JSON representadas.
    """

    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"El chunk {chunk_index} no contiene texto válido: {source_file}"
        )

    document_id = str(document.get("doc_id", source_file.stem))
    abbreviation = str(document.get("abbreviation", "unknown"))
    strategy = str(document.get("strategy", "unknown"))
    chunk_id = (
        f"json__{document_id}__{abbreviation}__{chunk_index:06d}"
    )

    metadata = dict(chunk.get("metadata") or {})
    for field in (
        "block",
        "group_name",
        "level",
        "json_paths",
        "exceeds_target_limit",
        "exceeds_max_limit",
    ):
        if field in chunk:
            metadata[field] = chunk[field]

    return {
        "chunk_id": chunk_id,
        "text": text,
        "format": "json",
        "document_type": infer_document_type(document),
        "document_id": document_id,
        "source_file": document.get("input_file", source_file.name),
        "source_path": document.get("source_path"),
        "strategy": strategy,
        "strategy_abbreviation": abbreviation,
        "tokenizer_model": document.get("tokenizer_model"),
        "strategy_params": document.get("params", {}),
        "chunk_index": chunk_index,
        "token_count": chunk.get("token_count"),
        "metadata": metadata,
    }


def normalize_markdown_chunk(
    chunk: Dict[str, Any],
    document: Dict[str, Any],
    chunk_index: int,
    source_file: Path,
) -> Dict[str, Any]:
    """
    Convierte un chunk Markdown al esquema común de indexación.

    Conserva el identificador original del chunk dentro de ``metadata``
    y genera un identificador común independiente de la implementación
    interna del runner Markdown.
    """

    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"El chunk {chunk_index} no contiene texto válido: {source_file}"
        )

    document_id = str(document.get("doc_id", source_file.stem))
    abbreviation = str(document.get("abbreviation", "unknown"))
    strategy = str(document.get("strategy", "unknown"))
    original_chunk_id = chunk.get("chunk_id")
    chunk_id = (
        f"markdown__{document_id}__{abbreviation}__{chunk_index:06d}"
    )

    metadata = dict(chunk.get("metadata") or {})
    if original_chunk_id is not None:
        metadata["original_chunk_id"] = original_chunk_id

    return {
        "chunk_id": chunk_id,
        "text": text,
        "format": "markdown",
        "document_type": infer_document_type(document),
        "document_id": document_id,
        "source_file": document.get("input_file", source_file.name),
        "source_path": document.get("source_path"),
        "strategy": strategy,
        "strategy_abbreviation": abbreviation,
        "tokenizer_model": document.get("tokenizer_model"),
        "strategy_params": document.get("params", {}),
        "chunk_index": chunk_index,
        "token_count": chunk.get("token_count"),
        "metadata": metadata,
    }


def iter_normalized_chunks() -> Iterator[Dict[str, Any]]:
    """
    Recorre y normaliza todos los archivos de chunks disponibles.

    Yields
    ------
    Dict[str, Any]
        Un registro normalizado por cada chunk encontrado.
    """

    if not CHUNKS_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de chunks: {CHUNKS_ROOT}. "
            "Ejecuta primero run_MD.py o run_JSON.py."
        )

    chunk_files = sorted(CHUNKS_ROOT.glob("*/chunk_results/*.json"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No se encontraron archivos de chunks en: {CHUNKS_ROOT}"
        )

    for chunk_file in chunk_files:
        document = load_chunk_file(chunk_file)
        format_name = chunk_file.parent.parent.name

        if format_name == "json":
            normalizer = normalize_json_chunk
        elif format_name == "markdown":
            normalizer = normalize_markdown_chunk
        else:
            raise ValueError(
                f"Formato de chunks no soportado en la ruta: {chunk_file}"
            )

        for chunk_index, chunk in enumerate(document["chunks"]):
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"El chunk {chunk_index} no es un objeto JSON: {chunk_file}"
                )
            yield normalizer(
                chunk=chunk,
                document=document,
                chunk_index=chunk_index,
                source_file=chunk_file,
            )


def normalize_chunks() -> int:
    """
    Escribe un registro JSONL por chunk.

    Returns
    -------
    int
        Número total de chunks normalizados y escritos.
    """

    INDEXING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    with NORMALIZED_OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for normalized_chunk in iter_normalized_chunks():
            file.write(
                json.dumps(
                    normalized_chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def load_indexing_config() -> Dict[str, Any]:
    """
    Carga y valida la configuración de Qdrant, embeddings e indexación.

    Returns
    -------
    Dict[str, Any]
        Configuración completa de la indexación.
    """

    if not INDEXING_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No se ha encontrado la configuración: {INDEXING_CONFIG_FILE}"
        )

    with INDEXING_CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("La configuración de indexación debe ser un objeto JSON.")

    return config


def load_normalized_chunks(
    formats: List[str],
) -> List[Dict[str, Any]]:
    """
    Carga los registros normalizados que corresponden a documentos JSON.

    Returns
    -------
    List[Dict[str, Any]]
        Lista de chunks de los formatos configurados preparados para generar
        embeddings.
    """

    if not NORMALIZED_OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo normalizado: {NORMALIZED_OUTPUT_FILE}. "
            "Ejecuta primero la fase de normalización."
        )

    chunks = []
    with NORMALIZED_OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("format") in formats:
                chunks.append(record)

    if not chunks:
        raise ValueError(
            "No se han encontrado chunks de los formatos configurados."
        )

    return chunks


def create_embeddings(config: Dict[str, Any]) -> HuggingFaceEmbeddings:
    """
    Crea el modelo de embeddings utilizado en esta configuración.

    Returns
    -------
    HuggingFaceEmbeddings
        Modelo BGE-M3 configurado para generar embeddings de documentos.
    """

    embeddings_config = config["embeddings"]
    return HuggingFaceEmbeddings(
        model_name=embeddings_config["model_name"],
        model_kwargs={"device": embeddings_config.get("device", "cpu")},
        encode_kwargs={
            "normalize_embeddings": embeddings_config.get(
                "normalize_embeddings",
                True,
            )
        },
    )


# ============================================================
# CACHÉ DE EMBEDDINGS EN DISCO
# ============================================================


def _sanitize_model_name(model_name: str) -> str:
    """
    Convierte el nombre del modelo en un nombre de archivo seguro.

    Reemplaza caracteres no alfanuméricos por guiones bajos.
    """

    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name)


def _text_hash(text: str) -> str:
    """
    Genera un hash corto y estable del texto de un chunk,
    usado para invalidar la caché si el texto cambia 
    aunque el chunk_id se mantenga igual.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embeddings_cache_path(model_name: str) -> Path:
    """
    Devuelve la ruta del archivo de caché para un modelo específico.
    """
    
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDINGS_CACHE_DIR / f"{_sanitize_model_name(model_name)}.jsonl"


def load_embeddings_cache(model_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Carga la caché de embeddings desde disco para un modelo específico.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Diccionario con chunk_id como clave y un diccionario con
        'vector' y 'text_hash' como valor.
        Si hay líneas duplicadas para el mismo chunk_id, se queda con
        la última (permite sobrescribir por append)
    """

    cache_file = embeddings_cache_path(model_name)
    cache: Dict[str, Dict[str, Any]] = {}

    if not cache_file.exists():
        return cache

    with cache_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["chunk_id"]] = {
                "vector": record["vector"],
                "text_hash": record["text_hash"],
            }
    return cache


def append_to_embedding_cache(
        model_name: str,
        entries: List[Tuple[str, str, List[float]]]
) -> None:
    """
    Añade nuevas entradas a la caché de embeddings de un modelo.

    Parameters
    ----------
    entries:
        Lista de tuplas (chunk_id, text_hash, vector)
    """
    
    if not entries:
        return
    
    cache_file = embeddings_cache_path(model_name)
    with cache_file.open("a", encoding="utf-8") as file:
        for chunk_id, text_hash, vector in entries:
            file.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "text_hash": text_hash,
                        "vector": vector,
                    },
                    ensure_ascii=False,
                    )
                    +"\n"
            )


# ============================================================
# LOG DE TIEMPOS
# ============================================================


def append_run_log(record: Dict [str, Any]) -> None:
    """
    Añade un registro de ejecución al log de tiempos.

    Guardar un log acumulativo permite comparar entre sí distintas
    ejecuciones: JSON vs Markdown, distinto número de workers,
    con/sin caché, distintos modelos de embedding, etc.
    """

    RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_run_summary(record: Dict[str, Any]) -> None:
    """
    Imprime un resumen de los tiempos de la ejecución actual.
    """

    print("-" * 70)
    print("RESUMEN DE TIEMPOS")
    print("-" * 70)
    if record.get("normalize_seconds") is not None:
        print(f"Normalización:      {record['normalize_seconds']:.2f} s")
    if record.get("chunks_by_format"):
        print(f"Chunks por formato: {record['chunks_by_format']}")
    if record.get("cache_hits") is not None:
        print(
            f"Embeddings de caché:  {record['cache_hits']}  |   "
            f"calculados: {record['cache_misses']}"
        )
    if record.get("embedding_seconds") is not None:
        print(f"Generación embeddings:  {record['embedding_seconds']:.2f} s")
        if record.get("cache_misses"):
            rate = record["cache_misses"] / max (record["embedding_seconds"], 1e-9)
            print(f"    ({rate:.1f} chunks/s)")
    if record.get("upsert_seconds") is not None:
        print(f"Indexación Qdrant:  {record['upsert_seconds']:.2f} s")
    print(f"Total:              {record['total_seconds']:.2f} s")
    print(f"Log guardado en:    {RUN_LOG_FILE}")
    print("-" * 70)


# ============================================================
# GENERACIÓN DE EMBEDDINGS (secuencial y paralela)
# ============================================================


def embed_texts_paralel(
    embeddings: HuggingFaceEmbeddings,
    texts: List[str],
    num_workers: int,
    batch_size: int,
) -> List[List[float]]:
    """
    Genera embeddings de manera paralela usando múltiples workers.

    Usa el pool multiproceso de sentence-transformers a través del modelo 
    interno que envuelve HuggingFaceEmbeddings (``embeddings.client``).
    Como el pool no aplica automáticamente encode_kwargs (p. ej.
    normalize_embeddings), la normalización se hace a mano después si 
    estaba activada en la configuración

    Parameters
    ----------
    num_workers:
        Numero de procesos worker. Cada uno carga su propia copia del
        modelo en memoria: no debe subirse más de lo que aguante la RAM
    """

    model = embeddings.client  # SentenceTransformer subyacente

    # Evita oversubscription: si cada uno de los N procesos usa todos los
    # hilos de la CPU, se pisan entre ellos y va más lento que en secuencial
    previous_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"

    pool = model.start_multi_process_pool(
        target_devices=["cpu"] * num_workers
    )
    try:
        vectors = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
        )
    finally:
        model.stop_multi_process_pool(pool)
        if previous_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_omp

    vectors = np.asarray(vectors)
    if embeddings.encode_kwargs.get("normalize_embeddings", True):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

    return vectors.tolist()


def embed_with_cache(
        embeddings: HuggingFaceEmbeddings,
        chunks: List[Dict[str, Any]],
        model_name: str,
        num_workers: int,
        batch_size: int,
        use_cache: bool,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Genera los embeddings de una lista de chunks, reutilizando la caché
    en disco cuando sea posible y calculando en paralelo el resto.

    Returns
    -------
    Tuple[List[List[float]], Dict[str, Any]]
        Un vector por chunk (mismo orden que ``chunks``) y un diccionario
        de estadísticas: cache_hits, cache_misses, embedding_secons (solo
        el tiempo de cómputo real, sin contar la lectura de la caché)
    """

    cache = load_embeddings_cache(model_name) if use_cache else {}

    vectors: List[List[float] | None] = [None] * len(chunks)
    pending_indices: List[int] = []
    pending_texts: List[str] = []
    pending_hashes: List[str] = []

    hits = 0
    for index, chunk in enumerate(chunks):
        text_hash = _text_hash(chunk["text"])
        cached = cache.get(chunk["chunk_id"])
        if cached is not None and cached["text_hash"] == text_hash:
            vectors[index] = cached["vector"]
            hits += 1
        else:
            pending_indices.append(index)
            pending_texts.append(chunk["text"])
            pending_hashes.append(text_hash)
    
    print(
        f"Embeddings reutilizados de caché: {hits}/{len(chunks)}"
        f"Pendientes de calcular: {len(pending_texts)}"
    )

    embedding_seconds = 0.0
    if pending_texts:
        start = time.perf_counter()
        if num_workers > 1:
            new_vectors = embed_texts_paralel(
                embeddings, pending_texts, num_workers, batch_size
            )
        else:
            new_vectors = embeddings.embed_documents(pending_texts)
        embedding_seconds = time.perf_counter() - start

        new_cache_entries = []
        for position, index in enumerate(pending_indices):
            vectors[index] = new_vectors[position]
            new_cache_entries.append(
                (
                    chunks[index]["chunk_id"],
                    pending_hashes[position],
                    new_vectors[position],
                )
            )

        if use_cache:
            append_to_embedding_cache(model_name, new_cache_entries)

    stats = {
        "cache_hits": hits,
        "cache_misses": len(pending_texts),
        "embedding_seconds": embedding_seconds
    }
    return vectors, stats   # type: ignore[return-value]


# ============================================================
# QDRANT
# ============================================================


def connect_to_qdrant(config: Dict[str, Any]) -> QdrantClient:
    """
    Crea un cliente conectado al servidor local de Qdrant.

    Returns
    -------
    QdrantClient
        Cliente configurado con el host y puerto del entorno.
    """

    qdrant_config = config["qdrant"]
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", qdrant_config["host"]),
        port=int(os.getenv("QDRANT_PORT", qdrant_config["port"])),
        timeout=60,
    )


def recreate_collection(
    client: QdrantClient,
    vector_size: int,
    collection_name: str,
) -> None:
    """
    Recrea la colección densa para evitar mezclar ejecuciones distintas.

    Parameters
    ----------
    client:
        Cliente conectado a Qdrant.

    vector_size:
        Dimensión de los embeddings del modelo seleccionado.
    """

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def index_chunks(
    client: QdrantClient,
    embeddings: HuggingFaceEmbeddings,
    chunks: List[Dict[str, Any]],
    collection_name: str,
    batch_size: int,
    model_name: str,
    num_workers: int,
    use_cache: bool,
) -> Dict[str, Any]:
    """
    Genera embeddings densos  (con caché y, opcionalmente, en 
    paralelo) y los inserta en Qdrant por lotes.

    Parameters
    ----------
    client:
        Cliente conectado a Qdrant.

    embeddings:
        Modelo utilizado para vectorizar los textos.

    chunks:
        Chunks normalizados que se van a indexar.

    Returns
    -------
    Dict[str, Any]
        Estadísticas de tiempo y caché (embedding + upsert).
    """

    print(f"Generando embeddings para {len(chunks)} chunks...")
    vectors, stats = embed_with_cache(
        embeddings=embeddings,
        chunks=chunks,
        model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        use_cache=use_cache,
    )

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            key: value
            for key, value in chunk.items()
            if key not in {"chunk_id", "text"}
        }
        payload["original_chunk_id"] = chunk["chunk_id"]
        payload["text"] = chunk["text"]
        points.append(
            models.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        chunk["chunk_id"],
                    )
                ),
                vector=vector,
                payload=payload,
            )
        )

    upsert_start = time.perf_counter()
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        print(f"Puntos indexados: {min(start + batch_size, len(points))}/{len(points)}")
    stats["upsert_seconds"] = time.perf_counter() - upsert_start

    return stats

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(
    normalize_only: bool = False, 
    index_only: bool = False,
    workers_override: int | None = None,
    use_cache: bool = True,    
) -> None:
    """
    Normaliza los outputs de chunking y, salvo que se indique lo
    contrario, genera embeddings y los indexa en Qdrant
    """

    run_start = time.perf_counter()
    normalize_seconds = None

    print("=" * 70)
    print("INICIO DE LA NORMALIZACIÓN PARA INDEXACIÓN")
    print("=" * 70)

    # 1. Normalizar los chunks y escribir el archivo JSONL
    if not index_only:
        normalize_start = time.perf_counter()
        count = normalize_chunks()
        normalize_seconds = time.perf_counter() - normalize_start
        print(f"Chunks normalizados: {count}")
        print(f"Archivo generado: {NORMALIZED_OUTPUT_FILE}")

    if normalize_only:
        print("Solo se ha ejecutado la fase de normalización.")
        print("=" * 70)
        print("FIN DE LA INDEXACIÓN")
        print("=" * 70)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "normalize-only",
            "normalize_seconds": normalize_seconds,
            "total_seconds": time.perf_counter() - run_start,
        }
        append_run_log(record)
        print_run_summary(record)
        return

    # 2. Cargar la configuración y los chunks JSON para la indexación densa
    config = load_indexing_config()
    formats = config["indexing"]["formats"]
    chunks = load_normalized_chunks(formats)
    chunks_by_format = dict(Counter(chunk["format"] for chunk in chunks))
    embeddings = create_embeddings(config)
    model_name = config["embeddings"]["model_name"]

    num_workers = workers_override or int(config["indexing"].get("num_workers", 1))
    num_workers = max(1, min(num_workers, os.cpu_count() or 1))
    print(f"Workers para generación de embeddings: {num_workers}")
    print(f"Chunks por formato: {chunks_by_format}")

    # 3. Obtener la dimensión y preparar la colección Qdrant
    vector_size = len(embeddings.embed_query("dimension check"))
    collection_name = config["qdrant"]["collection_name"]
    batch_size = int(config["indexing"]["batch_size"])
    client = connect_to_qdrant(config)
    recreate_collection(client, vector_size, collection_name)

    # 4. Generar embeddings e insertar los puntos
    index_stats = index_chunks(
        client,
        embeddings,
        chunks,
        collection_name,
        batch_size,
        model_name=model_name,
        num_workers=num_workers,
        use_cache=use_cache,
    )

    print(f"Colección Qdrant: {collection_name}")
    print("=" * 70)
    print("FIN DE LA NORMALIZACIÓN PARA INDEXACIÓN")
    print("=" * 70)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "index-only" if index_only else "full",
        "model_name": model_name,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "use_cache": use_cache,
        "collection_name": collection_name,
        "total_chunks": len(chunks),
        "chunks_by_format": chunks_by_format,
        "normalize_seconds": normalize_seconds,
        **index_stats,
        "total_seconds": time.perf_counter() - run_start,
    }
    append_run_log(record)
    print_run_summary(record)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normaliza e indexa chunks en Qdrant."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--normalize-only",
        action="store_true",
        help="Solo genera normalized_chunks.jsonl.",
    )
    mode.add_argument(
        "--index-only",
        action="store_true",
        help="Indexa el JSONL existente sin regenerarlo.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers para generar embeddings (por defecto, 1).",
    )
    parser.add_argument(
        "--no-cache",
        action = "store_true",
        help=(
            "Ignora la caché de embeddings en disco: recalcula todo y no"
            "guarda los resultados nuevos."
        ),
    )

    arguments = parser.parse_args()
    main(
        normalize_only=arguments.normalize_only,
        index_only=arguments.index_only,
        workers_override=arguments.workers,
        use_cache=not arguments.no_cache,
    )
