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

import json
import os
import uuid
import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List

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
CHUNKS_ROOT = PROJECT_ROOT / "output" / "chunks"

# Carpeta general de salida de la indexación:
# main/output/indexing
INDEXING_OUTPUT_DIR = PROJECT_ROOT / "output" / "indexing"

# Archivo normalizado, con un registro JSON por línea:
# main/output/indexing/normalized_chunks.jsonl
NORMALIZED_OUTPUT_FILE = INDEXING_OUTPUT_DIR / "normalized_chunks.jsonl"

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
        "embedding_model": document.get("embedding_model"),
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
        "embedding_model": document.get("embedding_model"),
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
) -> None:
    """
    Genera embeddings densos y los inserta en Qdrant por lotes.

    Parameters
    ----------
    client:
        Cliente conectado a Qdrant.

    embeddings:
        Modelo utilizado para vectorizar los textos.

    chunks:
        Chunks normalizados que se van a indexar.
    """

    texts = [chunk["text"] for chunk in chunks]
    print(f"Generando embeddings para {len(texts)} chunks...")
    vectors = embeddings.embed_documents(texts)

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

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        print(f"Puntos indexados: {min(start + batch_size, len(points))}/{len(points)}")

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(normalize_only: bool = False, index_only: bool = False) -> None:
    """
    Normaliza los outputs de chunking para la futura indexación vectorial.

    El script no genera embeddings ni modifica las fuentes originales.
    Solo prepara un archivo JSONL común para la siguiente fase.
    """

    print("=" * 70)
    print("INICIO DE LA NORMALIZACIÓN PARA INDEXACIÓN")
    print("=" * 70)

    # 1. Normalizar los chunks y escribir el archivo JSONL
    if not index_only:
        count = normalize_chunks()
        print(f"Chunks normalizados: {count}")
        print(f"Archivo generado: {NORMALIZED_OUTPUT_FILE}")

    if normalize_only:
        print("Solo se ha ejecutado la fase de normalización.")
        print("=" * 70)
        print("FIN DE LA INDEXACIÓN")
        print("=" * 70)
        return

    # 2. Cargar la configuración y los chunks JSON para la indexación densa
    config = load_indexing_config()
    formats = config["indexing"]["formats"]
    chunks = load_normalized_chunks(formats)
    embeddings = create_embeddings(config)

    # 3. Obtener la dimensión y preparar la colección Qdrant
    vector_size = len(embeddings.embed_query("dimension check"))
    collection_name = config["qdrant"]["collection_name"]
    batch_size = int(config["indexing"]["batch_size"])
    client = connect_to_qdrant(config)
    recreate_collection(client, vector_size, collection_name)

    # 4. Generar embeddings e insertar los puntos
    index_chunks(
        client,
        embeddings,
        chunks,
        collection_name,
        batch_size,
    )

    print(f"Colección Qdrant: {collection_name}")
    print("=" * 70)
    print("FIN DE LA NORMALIZACIÓN PARA INDEXACIÓN")
    print("=" * 70)

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
    arguments = parser.parse_args()
    main(
        normalize_only=arguments.normalize_only,
        index_only=arguments.index_only,
    )
