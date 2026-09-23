# ============================================================
# RETRIEVAL
# ============================================================

"""
Realiza búsquedas vectoriales sobre los chunks indexados en Qdrant.

El script:
1. Carga una lista de preguntas.
2. Genera el embedding de cada pregunta.
3. Busca los chunks más similares en Qdrant.
4. Guarda los resultados recuperados para su posterior evaluación.
"""

# ============================================================
# IMPORTS
# ============================================================

import json
from pathlib import Path
from typing import Any, Dict, List
import sys
import argparse

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

# Carpeta con los archivos de configuración
CONFIGS_DIR = (
    BASE_DIR
    / "configs"
)

# Archivo que contiene las preguntas y respuestas
GROUND_TRUTH_DIR = (
    BASE_DIR / "data" / "ground_truth"
)

QUERY_FILES = {
    "ficha_tecnica": GROUND_TRUTH_DIR / "rag_questions_ficha_tecnica.jsonl",
    "paper": GROUND_TRUTH_DIR / "rag_questions_paper.jsonl",
    "guideline": GROUND_TRUTH_DIR / "rag_questions_guideline.jsonl"
}

# Carpeta donde se guardarán los resultados
OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "retrieval"
)

sys.path.append(str(BASE_DIR))
from chunking.utilities import load_config

# ============================================================
# CARGAR PREGUNTAS
# ============================================================

def load_queries(
    filepath: Path,
) -> List[Dict[str, Any]]:

    """
    Carga las preguntas y respuestas desde un archivo JSONL.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo de preguntas: {filepath}"
        )

    queries = []

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            item = json.loads(line)

            if "question" not in item:
                raise ValueError(
                    f"La pregunta de la línea {line_number} "
                    "no contiene el campo 'question'."
                )

            if "gold_answer" not in item:
                raise ValueError(
                    f"La pregunta de la línea {line_number} "
                    "no contiene el campo 'gold_answer'."
                )

            if "question_id" not in item:
                raise ValueError(
                    f"La pregunta de la línea {line_number} "
                    "no contiene el campo 'question_id'."
                )

            queries.append(
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                }
            )

    return queries


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

# Crear nombre del archivo de salida según la configuración
# retrieval 
# + modelo
# + formato (si no es null) --> no lo usamos
# + estrategia (si no es null) --> no lo usamos
# + documento (si no es null)
# .json

def create_output_file(
    indexing_config: Dict[str, Any],
    retrieval_params: Dict[str, Any],
) -> Path:

    """
    Crea el nombre del archivo de resultados según
    los parámetros utilizados en el retrieval.
    """

    # Modelo de embeddings
    model_name = indexing_config["embeddings"]["model_name"]

    # Quedarnos solo con el nombre final del modelo
    # Ejemplo: "BAAI/bge-m3" -> "bge_m3"
    model_name = model_name.split("/")[-1]
    model_name = model_name.replace("-", "_").lower()

    parts = [
        "retrieval",
        model_name,
    ]

    # # Formato
    # format = retrieval_params.get(
    #     "format"
    # )

    # if format is not None:
    #     parts.append(format.upper())

    # # Estrategias
    # strategies = retrieval_params.get(
    #     "strategies"
    # )

    # if strategies is not None:
    #     if isinstance(strategies, list):
    #         parts.extend(strategies)
    #     else:
    #         parts.append(strategies)

    # Tipo de documento
    document_type = retrieval_params.get(
        "document_type"
    )

    if document_type is not None:
        parts.append(document_type)

    filename = "_".join(parts) + ".json"

    return OUTPUT_DIR / filename

def save_results(
    results: List[Dict[str, Any]],
    filepath: Path,
) -> None:

    """
    Guarda los resultados del retrieval en formato JSON.
    """

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with filepath.open(
        "w",
        encoding="utf-8",
    ) as file:

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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Cargando modelo de embeddings: "
        f"{embeddings_config['model_name']} "
        f"en {device}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_config["model_name"],
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings":
                embeddings_config.get(
                    "normalize_embeddings",
                    True,
                )
        },
    )

    return embeddings



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
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    question: str,
    top_k: int,
    strategy: str = None,
    document_type: str = None,
    format: str = None,
):
    """
    Recupera los chunks más similares a una pregunta.

    Los filtros por estrategia, tipo de documento y formato
    son opcionales.
    """

    # Generar embedding de la pregunta
    query_vector = embeddings.embed_query(
        question
    )

    # Lista de condiciones para Qdrant
    conditions = []

    # Filtro por estrategia
    if strategy is not None:
        conditions.append(
            models.FieldCondition(
                key="strategy",
                match=models.MatchValue(
                    value=strategy
                ),
            )
        )

    # Filtro por tipo de documento
    if document_type is not None:
        conditions.append(
            models.FieldCondition(
                key="document_type",
                match=models.MatchValue(
                    value=document_type
                ),
            )
        )

    # Filtro por formato
    if format is not None:
        conditions.append(
            models.FieldCondition(
                key="format",
                match=models.MatchValue(
                    value=format
                ),
            )
        )

    # Solo crear filtro si hay alguna condición
    query_filter = None

    if conditions:
        query_filter = models.Filter(
            must=conditions
        )

    # Buscar en Qdrant
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    return search_result


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

    for rank, hit in enumerate(
        hits,
        start=1,
    ):

        payload = hit.payload or {}

        metadata = payload.get(
            "metadata",
            {},
        )

        retrieved_chunks.append(
            {
                "rank": rank,
                "score": hit.score,
                "chunk_id": payload.get(
                    "original_chunk_id"
                ),
                "document_id": payload.get(
                    "document_id"
                ),
                "document_type": payload.get(
                    "document_type"
                ),
                "strategy": payload.get(
                    "strategy"
                ),
                "chunk_index": payload.get(
                    "chunk_index"
                ),
                "group_name": metadata.get(
                    "group_name"
                ),
                "text": payload.get(
                    "text"
                ),
            }
        )

    return retrieved_chunks


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Retrieval sobre Qdrant a partir de un archivo de configuración."
    )

    parser.add_argument(
        "retrieval_config",
        help="Nombre del JSON de configuración dentro de configs/ "
             "(p. ej. retrieval_config_bge_m3.json)",
    )

    return parser.parse_args()

def main() -> None:

    print("=" * 70)
    print("INICIO DEL RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Cargar configuración
    # --------------------------------------------------------

    indexing_config = load_config(
        INDEXING_CONFIG_FILE
    )

    args = parse_args()

    retrieval_config_file = CONFIGS_DIR / args.retrieval_config

    if not retrieval_config_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuración:\n{retrieval_config_file}"
        )

    print(f"Configuración de retrieval: {retrieval_config_file}")

    retrieval_config = load_config(retrieval_config_file)

    # Configuración de Qdrant
    collection_name = (
        indexing_config["qdrant"]["collection_name"]
    )

    retrieval_params = (
        retrieval_config["retrieval"]
    )

    output_file = create_output_file(
        indexing_config=indexing_config,
        retrieval_params=retrieval_params,
    )

    top_k = retrieval_params["top_k"]

    strategies = retrieval_params[
        "strategies"
    ]

    document_type = retrieval_params[
        "document_type"
    ]

    format = retrieval_params[
        "format"
    ]


    print(
        f"Colección Qdrant: {collection_name}"
    )

    print(
        f"Top K: {top_k}"
    )

    print(
        f"Tipo de documento: {document_type}"
    )

    print(
        f"Formato: {format}"
    )

    print(
        f"Estrategias: {strategies}"
    )
    # --------------------------------------------------------
    # 2. Cargar preguntas
    # --------------------------------------------------------

    query_file = QUERY_FILES[document_type]

    queries = load_queries(query_file)

    print(
        f"Preguntas cargadas: {len(queries)}"
    )

    # --------------------------------------------------------
    # 3. Cargar embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        indexing_config
    )

    # --------------------------------------------------------
    # 4. Conectar con Qdrant
    # --------------------------------------------------------

    client = connect_to_qdrant(
        indexing_config
    )

    # --------------------------------------------------------
    # 5. Ejecutar retrieval
    # --------------------------------------------------------

    all_results = []

    for query_item in queries:

        # ID de la pregunta
        question_id = query_item["question_id"]

        # Texto de la pregunta
        question = query_item["question"]

        # Respuesta
        gold_answer = query_item["gold_answer"]

        print(
            f"\nPregunta: {question}"
        )

        # ----------------------------------------------------
        # CASO 1: No se comparan estrategias
        # ----------------------------------------------------

        if strategies is None:

            hits = retrieve_chunks(
                client=client,
                embeddings=embeddings,
                collection_name=collection_name,
                question=question,
                top_k=top_k,
                strategy=None,
                document_type=document_type,
                format=format,
            )

            all_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "document_type": document_type,
                    "format": format,
                    "retrieved_chunks":
                        format_hits(hits),
                }
            )

        # ----------------------------------------------------
        # CASO 2: Se comparan varias estrategias
        # ----------------------------------------------------

        else:

            strategy_results = {}

            for strategy in strategies:

                print(
                    f"  Estrategia: {strategy}"
                )

                hits = retrieve_chunks(
                    client=client,
                    embeddings=embeddings,
                    collection_name=collection_name,
                    question=question,
                    top_k=top_k,
                    strategy=strategy,
                    document_type=document_type,
                    format=format,
                )

                strategy_results[
                    strategy
                ] = format_hits(
                    hits
                )

            all_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "document_type": document_type,
                    "format": format,
                    "strategies":
                        strategy_results,
                }
            )

    # --------------------------------------------------------
    # 6. Guardar resultados
    # --------------------------------------------------------

    save_results(
        all_results,
        output_file,
    )

    print()
    print(
        f"Resultados guardados en: "
        f"{output_file}"
    )

    print("=" * 70)
    print("FIN DEL RETRIEVAL")
    print("=" * 70)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()