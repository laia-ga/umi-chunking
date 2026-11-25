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
import time
from pathlib import Path
from typing import Any, Dict, List
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

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

# Carpeta con los archivos de configuración
CONFIGS_DIR = (
    BASE_DIR
    / "configs"
)

# Archivo que contiene las preguntas y respuestas
GROUND_TRUTH_DIR = (
    BASE_DIR / "data" / "ground_truth"
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
    retrieval_config: Dict[str, Any],
    retrieval_params: Dict[str, Any],
    collection_name: str,
) -> Path:

    """
    Crea el nombre del archivo de resultados según
    los parámetros utilizados en el retrieval.
    """

    # Modelo de embeddings
    model_name = retrieval_config["embeddings"]["model_name"]

    # Quedarnos solo con el nombre final del modelo
    # Ejemplo: "BAAI/bge-m3" -> "bge_m3"
    model_name = model_name.split("/")[-1]
    model_name = model_name.replace("-", "_").lower()

    # Identificar cada variante para no sobrescribir los resultados del benchmark.
    method = retrieval_params.get("method", "dense")
    parts = ["retrieval", model_name, method]
    if method in ("rrf", "weighted_rrf"):
        parts.extend([
            f"n{retrieval_params['prefetch_limit']}",
            f"rrfk{retrieval_params['rrf_k']}",
        ])
    if method == "weighted_rrf":
        d, b = retrieval_params["rrf_weights"]
        parts.append(f"d{str(d).replace('.', 'p')}_b{str(b).replace('.', 'p')}")
    parts.append(f"top{retrieval_params['top_k']}")

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

    specialty = retrieval_params.get("specialty")

    if specialty is not None:
        parts.append(specialty)

    filename = "_".join(parts) + ".json"

    return OUTPUT_DIR / collection_name / filename

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
    retrieval_config: Dict[str, Any],
    batch_size: int,
) -> HuggingFaceEmbeddings:

    """
    Carga el mismo modelo de embeddings utilizado
    durante la indexación.
    """

    embeddings_config = retrieval_config["embeddings"]

    device = (
        "cuda:0"
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
                ),
            "batch_size": batch_size,
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
    query_vector: List[float] | None,
    question: str,
    collection_name: str,
    top_k: int,
    strategy: str = None,
    document_type: str = None,
    format: str = None,
    specialty: str = None,
    method: str = "dense",
    bm25_field: str = "text",
    prefetch_limit: int = 30,
    rrf_k: int = 2,
    rrf_weights: List[float] | None = None,
):
    """Dense, BM25, RRF o Weighted RRF con idénticos filtros de Qdrant."""
    conditions = []
    for field, value in (
        ("strategy", strategy),
        ("document_type", document_type),
        ("format", format),
        ("specialty", specialty),
    ):
        if value is not None:
            conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value),
                )
            )
    query_filter = models.Filter(must=conditions) if conditions else None

    if method == "dense":
        if query_vector is None:
            raise ValueError("La búsqueda dense requiere un embedding de la pregunta.")
        search_kwargs = {"query": query_vector}

    elif method == "bm25":
        search_kwargs = {
            "query": models.TextQuery(text=question),
            "using": bm25_field,
        }

    elif method in ("rrf", "weighted_rrf"):
        if query_vector is None:
            raise ValueError("RRF requiere un embedding de la pregunta.")
        weights = [1.0, 1.0] if method == "rrf" else list(rrf_weights)
        search_kwargs = {
            "prefetch": [
                # ORDEN IMPORTANTE: primer peso = Dense, segundo = BM25.
                models.Prefetch(
                    query=query_vector,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.TextQuery(text=question),
                    using=bm25_field,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            "query": models.RrfQuery(
                rrf=models.Rrf(k=rrf_k, weights=weights)
            ),
        }
    else:
        raise ValueError(f"Método de retrieval desconocido: {method}")

    return client.query_points(
        collection_name=collection_name,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        **search_kwargs,
    ).points


def retrieve_question(
    client: QdrantClient,
    query_item: Dict[str, Any],
    query_vector: List[float] | None,
    collection_name: str,
    top_k: int,
    strategies: List[str] | None,
    document_type: str | None,
    format: str | None,
    specialty: str | None,
    retrieval_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Conserva el mismo formato JSON para el evaluador AtlaAI."""
    result = {
        "question_id": query_item["question_id"],
        "question": query_item["question"],
        "gold_answer": query_item["gold_answer"],
        "document_type": document_type,
        "format": format,
        "specialty": specialty,
    }

    if strategies is None:
        hits = retrieve_chunks(
            client=client,
            query_vector=query_vector,
            question=query_item["question"],
            collection_name=collection_name,
            top_k=top_k,
            strategy=None,
            document_type=document_type,
            format=format,
            specialty=specialty,
            **retrieval_options,
        )
        result["retrieved_chunks"] = format_hits(hits)
        return result

    strategy_results = {}
    for strategy in strategies:
        hits = retrieve_chunks(
            client=client,
            query_vector=query_vector,
            question=query_item["question"],
            collection_name=collection_name,
            top_k=top_k,
            strategy=strategy,
            document_type=document_type,
            format=format,
            specialty=specialty,
            **retrieval_options,
        )
        strategy_results[strategy] = format_hits(hits)

    result["strategies"] = strategy_results
    return result


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
                "specialty": payload.get(
                    "specialty"
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
        "--config",
        type=str,
        default="retrieval_config.json",
        help=(
            "Nombre del archivo de configuración de retrieval dentro de configs/ "
            "(por defecto: retrieval_config.json)."
        ),
    )

    parser.add_argument(
        "--indexing-config",
        type=str,
        default="indexing_config.json",
        help=(
            "Nombre del archivo de configuración de indexación dentro de configs/ "
            "(por defecto: indexing_config.json)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Consultas de preguntas concurrentes a Qdrant (por defecto: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=16,
        help="Tamaño de lote para embeddings de preguntas (por defecto: 16).",
    )
    parser.add_argument(
        "--method",
        choices=("dense", "bm25", "rrf", "weighted_rrf"),
        default=None,
        help="Sobrescribe retrieval.method del JSON de configuración.",
    )
    parser.add_argument(
        "--rrf-weights",
        type=float,
        nargs=2,
        metavar=("DENSE", "BM25"),
        default=None,
        help="Pesos de Weighted RRF: primero Dense, después BM25.",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=_positive_int,
        default=None,
        help="Candidatos por rama antes de la fusión.",
    )
    parser.add_argument(
        "--rrf-k",
        type=_positive_int,
        default=None,
        help="Constante de RRF (distinta del top_k final).",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero positivo."
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "El valor debe ser al menos 1."
        )
    return parsed


def main() -> None:

    print("=" * 70)
    print("INICIO DEL RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Cargar configuración
    # --------------------------------------------------------

    args = parse_args()

    # Archivo de configuración de indexación
    indexing_config_file = CONFIGS_DIR / args.indexing_config

    if not indexing_config_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuración:\n{indexing_config_file}"
        )

    print(f"Configuración de indexación: {indexing_config_file}")

    indexing_config = load_config(indexing_config_file)

    # Archivo de configuración de retrieval
    retrieval_config_file = CONFIGS_DIR / args.config

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

    retrieval_params = retrieval_config["retrieval"]

    # Opciones del nuevo benchmark, configurables en JSON y desde la CLI.
    method = args.method or retrieval_params.get("method", "dense")
    if method not in {"dense", "bm25", "rrf", "weighted_rrf"}:
        raise ValueError(f"Método de recuperación inválido: {method}")
    bm25_field = retrieval_params.get("bm25_field", "text")
    prefetch_limit = args.prefetch_limit or int(retrieval_params.get("prefetch_limit", 30))
    rrf_k = args.rrf_k or int(retrieval_params.get("rrf_k", 2))
    rrf_weights = args.rrf_weights or retrieval_params.get("rrf_weights", [3.0, 1.0])
    if not isinstance(bm25_field, str) or not bm25_field.strip():
        raise ValueError("bm25_field debe ser el nombre del campo de texto indexado.")
    if prefetch_limit < 1 or rrf_k < 1:
        raise ValueError("prefetch_limit y rrf_k deben ser positivos.")
    if len(rrf_weights) != 2 or any(float(w) <= 0 for w in rrf_weights):
        raise ValueError("rrf_weights debe tener dos números positivos [Dense, BM25].")
    top_k = int(retrieval_params["top_k"])
    if method in {"rrf", "weighted_rrf"} and prefetch_limit < top_k:
        raise ValueError("Para fusionar, prefetch_limit debe ser >= top_k.")

    retrieval_params["method"] = method
    retrieval_params["bm25_field"] = bm25_field
    retrieval_params["prefetch_limit"] = prefetch_limit
    retrieval_params["rrf_k"] = rrf_k
    retrieval_params["rrf_weights"] = [float(w) for w in rrf_weights]
    retrieval_options = {
        "method": method,
        "bm25_field": bm25_field,
        "prefetch_limit": prefetch_limit,
        "rrf_k": rrf_k,
        "rrf_weights": retrieval_params["rrf_weights"],
    }

    output_file = create_output_file(
        retrieval_config=retrieval_config,
        retrieval_params=retrieval_params,
        collection_name=collection_name,
    )

    # top_k ya está validado antes de construir el nombre de salida.

    strategies = retrieval_params[
        "strategies"
    ]

    document_type = retrieval_params[
        "document_type"
    ]

    format = retrieval_params[
        "format"
    ]

    specialty = retrieval_params.get("specialty")

    ground_truth_file = retrieval_params[
        "ground_truth_file"
    ]

    query_file = (
        GROUND_TRUTH_DIR
        / ground_truth_file
    )


    print(
        f"Colección Qdrant: {collection_name}"
    )

    print(f"Método: {method}; top_k final: {top_k}")
    if method in {"rrf", "weighted_rrf"}:
        effective_weights = [1.0, 1.0] if method == "rrf" else retrieval_options["rrf_weights"]
        print(f"Prefetch por rama: {prefetch_limit}; RRF k: {rrf_k}; pesos [Dense, BM25]: {effective_weights}")

    print(
        f"Tipo de documento: {document_type}"
    )

    print(
        f"Formato: {format}"
    )

    print(
        f"Especialidad: {specialty}"
    )

    print(
        f"Estrategias: {strategies}"
    )

    print(
        f"Ground truth: {ground_truth_file}"
    )
    # --------------------------------------------------------
    # 2. Cargar preguntas
    # --------------------------------------------------------

    queries = load_queries(query_file)

    print(
        f"Preguntas cargadas: {len(queries)}"
    )

    # --------------------------------------------------------
    # 3. Cargar embeddings solo cuando sean necesarios
    # --------------------------------------------------------
    start_time = time.perf_counter()
    if method == "bm25":
        print("BM25: no se cargan embeddings ni GPU.")
        query_vectors: List[List[float] | None] = [None] * len(queries)
    else:
        embeddings = create_embeddings(retrieval_config, batch_size=args.batch_size)
        model_name = retrieval_config["embeddings"]["model_name"]
        is_e5 = model_name.startswith("intfloat/multilingual-e5")
        query_vectors = []
        for start in range(0, len(queries), args.batch_size):
            batch_queries = queries[start:start + args.batch_size]
            texts = [item["question"] for item in batch_queries]
            if is_e5:
                texts = ["query: " + text for text in texts]
            # Se mantiene embed_documents(), igual que en tu retrieval previo.
            batch_vectors = embeddings.embed_documents(texts)
            if len(batch_vectors) != len(batch_queries):
                raise RuntimeError("Número de embeddings distinto al de preguntas.")
            query_vectors.extend(batch_vectors)
            print(f"Embeddings de preguntas: {len(query_vectors)}/{len(queries)}")

    if len(query_vectors) != len(queries):
        raise RuntimeError("Número de vectores de preguntas incorrecto.")

    # --------------------------------------------------------
    # 4. Conectar con Qdrant
    # --------------------------------------------------------
    client = connect_to_qdrant(indexing_config)

    # --------------------------------------------------------
    # 5. Ejecutar retrieval
    # --------------------------------------------------------

    total_queries = len(queries) * (
        len(strategies) if strategies is not None else 1
    )
    worker_count = min(args.workers, max(1, len(queries)))
    print(
        f"Consultas Qdrant: {total_queries}; workers: {worker_count}; "
        f"batch de embeddings: {args.batch_size}"
    )

    work_items = (
        (
            client,
            query_item,
            query_vector,
            collection_name,
            top_k,
            strategies,
            document_type,
            format,
            specialty,
            retrieval_options,
        )
        for query_item, query_vector in zip(queries, query_vectors)
    )
    all_results = []
    try:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="qdrant-retrieval",
        ) as executor:
            for completed, result in enumerate(
                executor.map(lambda item: retrieve_question(*item), work_items),
                start=1,
            ):
                all_results.append(result)
                if completed % 100 == 0 or completed == len(queries):
                    elapsed = time.perf_counter() - start_time
                    completed_queries = completed * (
                        len(strategies) if strategies is not None else 1
                    )
                    print(
                        f"Preguntas recuperadas: {completed}/{len(queries)} "
                        f"({completed_queries}/{total_queries} consultas); "
                        f"tiempo transcurrido: {elapsed:.1f} s"
                    )
    finally:
        client.close()

    print(
        f"Tiempo total embeddings + retrieval: "
        f"{time.perf_counter() - start_time:.1f} s"
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