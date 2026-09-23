# ============================================================
# RETRIEVAL
# ============================================================

"""
Realiza búsquedas vectoriales sobre los chunks indexados en Qdrant.

El script:
1. Carga una lista de preguntas.
2. Genera el embedding de cada pregunta (una sola vez).
3. Busca los chunks más similares en Qdrant para cada
   combinación strategy × document_type × document_format.
4. Guarda los resultados recuperados para su posterior evaluación.

En retrieval_config.json, cada uno de estos parámetros puede ser
null (sin filtro), un único valor o una lista de valores:

    "strategies", "document_type", "document_format"
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import torch

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ============================================================
# RUTAS
# ============================================================

# Carpeta donde se encuentra este script
SCRIPTS_DIR = Path(__file__).resolve().parent

# Carpeta raíz del proyecto
BASE_DIR = SCRIPTS_DIR.parent

# Archivo de configuración de la indexación
INDEXING_CONFIG_FILE = (
    BASE_DIR
    / "configs"
    / "indexing_config.json"
)

# Archivo de configuración de la búsqueda
RETRIEVAL_CONFIG_FILE = (
    BASE_DIR
    / "configs"
    / "retrieval_config.json"
)

# Archivo que contiene las preguntas y respuestas
QUERY_FILE = (
    BASE_DIR
    / "data"
    / "ground_truth"
    / "rag_questions_1500_general.jsonl"
)

# Carpeta donde se guardarán los resultados
OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "retrieval"
)

sys.path.append(str(BASE_DIR))
from chunking.utilities import load_config


# ============================================================
# UTILIDADES
# ============================================================

def as_list(value):

    """
    Convierte un parámetro de configuración en lista:

        null        -> [None]  (sin filtro)
        "valor"     -> ["valor"]
        ["a", "b"]  -> ["a", "b"]
    """

    if value is None:
        return [None]

    if isinstance(value, list):
        return value

    return [value]


# ============================================================
# CARGAR PREGUNTAS
# ============================================================

def load_queries(
    filepath: Path,
) -> List[Dict[str, Any]]:

    """
    Carga las preguntas y respuestas desde un archivo JSONL.

    Si la pregunta tiene el campo 'document_type', se conserva
    para buscarla solo en los chunks de su tipo de documento.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo de preguntas: {filepath}"
        )

    queries = []

    with filepath.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            item = json.loads(line)

            for field in ["question", "gold_answer", "question_id"]:

                if field not in item:
                    raise ValueError(
                        f"La pregunta de la línea {line_number} "
                        f"no contiene el campo '{field}'."
                    )

            queries.append(
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "document_type": item.get("document_type"),
                }
            )

    return queries


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def create_output_file(
    indexing_config: Dict[str, Any],
) -> Path:

    """
    Crea el nombre del archivo de resultados a partir
    del modelo de embeddings.

    Ejemplo: "BAAI/bge-m3" -> retrieval_bge_m3.json
    """

    model_name = indexing_config["embeddings"]["model_name"]

    model_name = model_name.split("/")[-1]
    model_name = model_name.replace("-", "_").lower()

    return OUTPUT_DIR / f"retrieval_{model_name}.json"


def save_results(
    results: List[Dict[str, Any]],
    filepath: Path,
) -> None:

    """
    Guarda los resultados del retrieval en formato JSON.
    """

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CREAR MODELO DE EMBEDDINGS
# ============================================================

def create_embeddings(
    indexing_config: Dict[str, Any],
) -> HuggingFaceEmbeddings:

    """
    Carga el mismo modelo de embeddings utilizado
    durante la indexación.
    """

    embeddings_config = indexing_config["embeddings"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Cargando modelo de embeddings: "
        f"{embeddings_config['model_name']} "
        f"en {device}"
    )

    return HuggingFaceEmbeddings(
        model_name=embeddings_config["model_name"],
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings":
                embeddings_config.get("normalize_embeddings", True)
        },
    )


# ============================================================
# CONECTAR A QDRANT
# ============================================================

def connect_to_qdrant(
    indexing_config: Dict[str, Any],
) -> QdrantClient:

    """
    Conecta con el servidor local de Qdrant.
    """

    qdrant_config = indexing_config["qdrant"]

    return QdrantClient(
        host=qdrant_config["host"],
        port=qdrant_config["port"],
        timeout=60,
    )


# ============================================================
# BUSCAR CHUNKS
# ============================================================

def retrieve_chunks(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int,
    strategy: str = None,
    document_type: str = None,
    document_format: str = None,
):

    """
    Recupera los chunks más similares a una pregunta
    a partir de su embedding ya calculado.

    Los filtros por estrategia, tipo de documento y formato
    son opcionales (None = sin filtro).
    """

    filters = {
        "strategy": strategy,
        "document_type": document_type,
        "format": document_format,
    }

    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )
        for key, value in filters.items()
        if value is not None
    ]

    query_filter = models.Filter(must=conditions) if conditions else None

    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points


# ============================================================
# FORMATEAR RESULTADOS
# ============================================================

def format_hits(
    hits,
) -> List[Dict[str, Any]]:

    """
    Convierte los puntos devueltos por Qdrant
    a un formato sencillo para guardar.
    """

    retrieved_chunks = []

    for rank, hit in enumerate(hits, start=1):

        payload = hit.payload or {}
        metadata = payload.get("metadata", {})

        retrieved_chunks.append(
            {
                "rank": rank,
                "score": hit.score,
                "chunk_id": payload.get("original_chunk_id"),
                "document_id": payload.get("document_id"),
                "document_type": payload.get("document_type"),
                "strategy": payload.get("strategy"),
                "chunk_index": payload.get("chunk_index"),
                "group_name": metadata.get("group_name"),
                "text": payload.get("text"),
            }
        )

    return retrieved_chunks


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:

    print("=" * 70)
    print("INICIO DEL RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Cargar configuración
    # --------------------------------------------------------

    indexing_config = load_config(INDEXING_CONFIG_FILE)
    retrieval_config = load_config(RETRIEVAL_CONFIG_FILE)

    collection_name = indexing_config["qdrant"]["collection_name"]

    retrieval_params = retrieval_config["retrieval"]

    top_k = retrieval_params["top_k"]

    strategies = as_list(retrieval_params.get("strategies"))
    document_types = as_list(retrieval_params.get("document_type"))
    document_formats = as_list(retrieval_params.get("document_format"))

    combinations = list(
        product(
            strategies,
            document_types,
            document_formats,
        )
    )

    output_file = create_output_file(indexing_config)

    print(f"Colección Qdrant: {collection_name}")
    print(f"Top K: {top_k}")
    print(f"Estrategias: {strategies}")
    print(f"Tipos de documento: {document_types}")
    print(f"Formatos: {document_formats}")
    print(f"Combinaciones posibles: {len(combinations)}")

    # --------------------------------------------------------
    # 2. Cargar preguntas
    # --------------------------------------------------------

    queries = load_queries(QUERY_FILE)

    print(f"Preguntas cargadas: {len(queries)}")

    # --------------------------------------------------------
    # 3. Cargar embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(indexing_config)

    # --------------------------------------------------------
    # 4. Conectar con Qdrant
    # --------------------------------------------------------

    client = connect_to_qdrant(indexing_config)

    # --------------------------------------------------------
    # 5. Ejecutar retrieval
    # --------------------------------------------------------

    all_results = []

    for i, query_item in enumerate(queries, start=1):

        question_id = query_item["question_id"]
        question = query_item["question"]
        gold_answer = query_item["gold_answer"]
        question_doc_type = query_item["document_type"]

        print(f"\n[{i}/{len(queries)}] {question_id}: {question}")

        # Embedding una sola vez por pregunta
        query_vector = embeddings.embed_query(question)

        for strategy, doc_type, doc_format in combinations:

            # Si la pregunta indica su tipo de documento,
            # solo se busca en ese tipo
            if (
                question_doc_type is not None
                and doc_type is not None
                and question_doc_type != doc_type
            ):
                continue

            hits = retrieve_chunks(
                client=client,
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                strategy=strategy,
                document_type=doc_type,
                document_format=doc_format,
            )

            # Combinación sin chunks indexados
            # (p. ej. estrategia markdown × ficha_tecnica)
            if not hits:
                continue

            all_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "strategy": strategy,
                    "document_type": doc_type,
                    "document_format": doc_format,
                    "retrieved_chunks": format_hits(hits),
                }
            )

    # --------------------------------------------------------
    # 6. Guardar resultados
    # --------------------------------------------------------

    save_results(all_results, output_file)

    print()
    print(f"Entradas guardadas: {len(all_results)}")
    print(f"Resultados guardados en: {output_file}")

    print("=" * 70)
    print("FIN DEL RETRIEVAL")
    print("=" * 70)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()