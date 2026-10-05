# ============================================================
# CHUNKING MARKDOWN
# ============================================================

"""
Ejecuta todas las estrategias de chunking para documentos Markdown.

Este script:
1. Carga los documentos Markdown desde data/raw/markdown.
2. Aplica todos los chunkers disponibles (rule-based, recursivos,
   semánticos, dinámicos y basados en LLM).
3. Genera un archivo JSON por cada documento con:
   - los chunks producidos,
   - los metadatos del chunker,
   - las estadísticas de tokens,
   - la configuración utilizada.
4. Guarda los resultados en output/chunks/markdown/chunk_results.
5. Guarda las métricas en output/chunks/markdown/metrics.
6. Genera un summary.csv con una fila por estrategia.
"""

# ============================================================
# IMPORTS GENERALES
# ============================================================

import json
import os
import time
import argparse
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import nltk
import pandas as pd
import psutil
import torch

# Utilidades
from chunking.utilities import (
    calculate_stats,
    create_tokenizer,
    create_token_counter,
    load_config,
    save_json,
)

# ============================================================
# MÉTODOS RULE-BASED
# ============================================================

from chunking.chunkers.rulebased.fixed_character_chunking import FixedCharacterChunker
from chunking.chunkers.rulebased.fixed_token_chunking import FixedTokenChunker
from chunking.chunkers.rulebased.length_aware_chunking import LengthAwareChunker
from chunking.chunkers.rulebased.overlapping_token_chunking import OverlappingTokenChunker
from chunking.chunkers.rulebased.paragraph_based_chunking import ParagraphBasedChunker
from chunking.chunkers.rulebased.paragraph_group_chunking import ParagraphGroupChunker
from chunking.chunkers.rulebased.sentence_based_chunking import SentenceBasedChunker
from chunking.chunkers.rulebased.sentence_group_chunking import SentenceGroupChunker
from chunking.chunkers.rulebased.sliding_window_token_chunking import SlidingWindowTokenChunker

# ============================================================
# MÉTODOS RECURSIVOS
# ============================================================

from chunking.chunkers.recursive.recursive_chunking import RecursiveChunker
from chunking.chunkers.recursive.recursive_token_chunking import RecursiveTokenChunker
from chunking.chunkers.recursive.parent_child_chunking import ParentChildChunker

# ============================================================
# MÉTODOS SEMÁNTICOS
# ============================================================

from chunking.chunkers.semantic.semantic_boundary_chunking import SemanticBoundaryChunker
from chunking.chunkers.semantic.semantic_embedding_chunking import SemanticEmbeddingChunker
from chunking.chunkers.semantic.semantic_similarity_threshold_chunking import SemanticSimilarityThresholdChunker
from chunking.chunkers.semantic.topic_based_chunking import TopicBasedChunker

# ============================================================
# MÉTODOS DINÁMICOS
# ============================================================

from chunking.chunkers.dynamic.content_density_adaptive_chunking import ContentDensityAdaptiveChunker
from chunking.chunkers.dynamic.dynamic_token_size_chunking import DynamicTokenSizeChunker
from chunking.chunkers.dynamic.semantic_variance_adaptive_chunking import SemanticVarianceAdaptiveChunker

# ============================================================
# MÉTODOS BASADOS EN LLM
# ============================================================

from chunking.chunkers.llm.llm_boundary_detection_chunking import LLMBoundaryDetectionChunker
from chunking.chunkers.llm.llm_segment_then_chunking import LLMSegmentThenChunker

# ============================================================
# RUTAS
# ============================================================

# Carpeta donde se encuentra el archivo run_MD.py: 
# main/chunking
CHUNKING_DIR = Path(__file__).resolve().parent

# Carpeta raíz del proyecto:
# main
BASE_DIR = CHUNKING_DIR.parent

# Archivo JSON con la configuración de los chunkers:
# main/configs
CHUNKING_CONFIG_FILE = BASE_DIR / "configs" / "chunker_config.json"

# Archivo JSON con la configuración del tokenizador:
TOKENIZER_CONFIG_FILE = BASE_DIR / "configs" / "tokenizer_config.json"

# Carpeta que contiene los scripts de análisis:
# main/scripts
SCRIPTS_DIR = BASE_DIR / "scripts"

# Carpeta que contiene los documentos Markdown de entrada
# main/data/raw/markdown
INPUT_DIR = BASE_DIR / "data" / "raw" / "markdown"

# Carpeta general de salida
OUTPUT_DIR = BASE_DIR / "output" / "chunks" / "markdown"

# Carpeta donde se guardarán los chunks generados
CHUNKS_DIR = OUTPUT_DIR / "chunk_results"

# Carpeta donde se guardarán las métricas de cada método
METRICS_DIR = OUTPUT_DIR / "metrics"

# Archivo resumen con una fila por método
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

# ============================================================
# DICCIONARIO DE CLASES DE CHUNKING
# ============================================================

CHUNKER_CLASSES = {
    # Rule-based
    "FixedCharacterChunker": FixedCharacterChunker,
    "FixedTokenChunker": FixedTokenChunker,
    "LengthAwareChunker": LengthAwareChunker,
    "OverlappingTokenChunker": OverlappingTokenChunker,
    "ParagraphBasedChunker": ParagraphBasedChunker,
    "ParagraphGroupChunker": ParagraphGroupChunker,
    "SentenceBasedChunker": SentenceBasedChunker,
    "SentenceGroupChunker": SentenceGroupChunker,
    "SlidingWindowTokenChunker": SlidingWindowTokenChunker,

    # Recursive
    "RecursiveChunker": RecursiveChunker,
    "RecursiveTokenChunker": RecursiveTokenChunker,
    "ParentChildChunker": ParentChildChunker,

    # Semantic
    "SemanticBoundaryChunker": SemanticBoundaryChunker,
    "SemanticEmbeddingChunker": SemanticEmbeddingChunker,
    "SemanticSimilarityThresholdChunker": SemanticSimilarityThresholdChunker,
    "TopicBasedChunker": TopicBasedChunker,

    # Dynamic
    "ContentDensityAdaptiveChunker": ContentDensityAdaptiveChunker,
    "DynamicTokenSizeChunker": DynamicTokenSizeChunker,
    "SemanticVarianceAdaptiveChunker": SemanticVarianceAdaptiveChunker,

    # LLM-based
    "LLMBoundaryDetectionChunker": LLMBoundaryDetectionChunker,
    "LLMSegmentThenChunker": LLMSegmentThenChunker
}

# ============================================================
# ABREVIATURAS DE LOS MÉTODOS
# ============================================================

CHUNKER_ABBREVIATIONS = {
    # Rule-based
    "FixedCharacterChunker": "FCC",
    "FixedTokenChunker": "FC",
    "OverlappingTokenChunker": "OFC",
    "SlidingWindowTokenChunker": "SWC",
    "LengthAwareChunker": "LAC",
    "SentenceBasedChunker": "SBC",
    "SentenceGroupChunker": "SGC",
    "ParagraphBasedChunker": "PBC",
    "ParagraphGroupChunker": "PGC",

    # Recursive
    "RecursiveChunker": "RC",
    "RecursiveTokenChunker": "RTF",
    "ParentChildChunker": "PCC",

    # Semantic
    "SemanticEmbeddingChunker": "SEBC",
    "SemanticSimilarityThresholdChunker": "SSTC",
    "TopicBasedChunker": "TBC",
    "SemanticBoundaryChunker": "SBDC",

    # Dynamic
    "DynamicTokenSizeChunker": "DFC",
    "ContentDensityAdaptiveChunker": "CDAC",
    "SemanticVarianceAdaptiveChunker": "SVAC",

    # LLM
    "LLMBoundaryDetectionChunker": "LBDC",
    "LLMSegmentThenChunker": "LSTC",
}

# ============================================================
# PARALELIZACIÓN DE DOCUMENTOS Y GPU EN EMBEDDINGS
# ============================================================

GPU_EMBEDDING_CHUNKERS = {
    "SemanticEmbeddingChunker",
    "SemanticSimilarityThresholdChunker",
    "TopicBasedChunker",
    "SemanticBoundaryChunker",
    "SemanticVarianceAdaptiveChunker",
}

NLTK_CHUNKERS = {
    "SentenceBasedChunker",
    "SentenceGroupChunker",
    "LengthAwareChunker",
    "SemanticSimilarityThresholdChunker",
    "TopicBasedChunker",
    "SemanticVarianceAdaptiveChunker",
    "ContentDensityAdaptiveChunker",
    "LLMBoundaryDetectionChunker",
    "LLMSegmentThenChunker",
}

LLM_CHUNKERS = {
    "LLMBoundaryDetectionChunker",
    "LLMSegmentThenChunker",
}

_WORKER_CONTEXT = threading.local()


def _initialize_document_worker(
    chunker_type: str,
    params: Dict[str, Any],
    tokenizer_model_name: str,
    device: str,
    measure_ram: bool,
) -> None:
    """Crea tokenizer y chunker privados para el hilo que procesa documentos."""

    try:
        tokenizer = create_tokenizer(tokenizer_model_name)
        token_counter = create_token_counter(tokenizer)
        chunker_params = params.copy()
        constructor_params = inspect.signature(
            CHUNKER_CLASSES[chunker_type].__init__
        ).parameters

        if "token_counter" in constructor_params:
            chunker_params["token_counter"] = token_counter
        if "tokenizer" in constructor_params:
            chunker_params["tokenizer"] = tokenizer
        if chunker_type in GPU_EMBEDDING_CHUNKERS:
            chunker_params["device"] = device

        _WORKER_CONTEXT.chunker = CHUNKER_CLASSES[chunker_type](
            **chunker_params
        )
        _WORKER_CONTEXT.initialization_error = None
        _WORKER_CONTEXT.measure_ram = measure_ram
    except Exception as error:
        _WORKER_CONTEXT.initialization_error = str(error)


def _chunk_document(
    document: Dict[str, Any],
) -> Dict[str, Any]:
    """Procesa un documento usando el chunker local del worker."""

    initialization_error = getattr(
        _WORKER_CONTEXT,
        "initialization_error",
        None,
    )
    if initialization_error is not None:
        return {"initialization_error": initialization_error}

    process = psutil.Process(os.getpid()) if _WORKER_CONTEXT.measure_ram else None
    ram_before_mb = (
        process.memory_info().rss / (1024 ** 2)
        if process is not None
        else None
    )
    start_time = time.perf_counter()

    try:
        chunks = list(
            _WORKER_CONTEXT.chunker.chunk(
                text=document["text"],
                doc_id=document["doc_id"],
            )
        )
    except Exception as error:
        return {
            "error": str(error),
            "execution_time_seconds": time.perf_counter() - start_time,
            "ram_before_mb": ram_before_mb,
            "ram_after_mb": (
                process.memory_info().rss / (1024 ** 2)
                if process is not None
                else None
            ),
        }

    return {
        "chunks": chunks,
        "execution_time_seconds": time.perf_counter() - start_time,
        "ram_before_mb": ram_before_mb,
        "ram_after_mb": (
            process.memory_info().rss / (1024 ** 2)
            if process is not None
            else None
        ),
    }


def _ensure_nltk_data() -> None:
    """Descarga una sola vez el recurso usado por los tokenizadores NLTK."""

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError(
                "No se ha podido descargar el recurso NLTK 'punkt_tab'."
            )
        nltk.data.find("tokenizers/punkt_tab")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero positivo."
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero positivo."
        )
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero no negativo."
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero no negativo."
        )
    return parsed


def _resolve_device(gpu_id: int | None, needs_gpu: bool) -> str:
    if gpu_id is None or not needs_gpu:
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Se ha solicitado una GPU, pero CUDA no está disponible."
        )
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(
            f"GPU ID {gpu_id} no válido: hay {device_count} GPU(s) CUDA visibles."
        )
    return f"cuda:{gpu_id}"


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(
    workers_override: int | None = None,
    gpu_id_override: int | None = None,
) -> None:
    """
    Ejecuta todas las estrategias activadas sobre los documentos
    Markdown disponibles en la carpeta de entrada.

    Los chunks y las métricas se guardan por separado para cada
    combinación de método y documento.
    """

    print("=" * 70)
    print("INICIO DE LA EJECUCIÓN")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Comprobar carpeta y archivos de entrada
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"No se ha encontrado la carpeta de entrada: "
            f"{INPUT_DIR}"
        )

    # Busca archivos dentro de las carpetas paper/guideline/ficha_tecnica dentro de markdown
    markdown_files = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in {
                ".md",
                ".markdown",
            }
        )
    )

    if not markdown_files:
        raise FileNotFoundError(
            f"No se han encontrado archivos Markdown en: "
            f"{INPUT_DIR}"
        )

    print(
        f"Documentos Markdown encontrados: "
        f"{len(markdown_files)}"
    )

    # --------------------------------------------------------
    # 2. Cargar los documentos
    # --------------------------------------------------------

    documents = []

    for markdown_path in markdown_files:
        try:
            text = markdown_path.read_text(
                encoding="utf-8"
            ).strip()

        except OSError as error:
            print(
                f"[ERROR] No se ha podido leer "
                f"{markdown_path.name}: {error}"
            )
            continue

        if not text:
            print(
                f"[AVISO] El documento "
                f"{markdown_path.name} está vacío."
            )
            continue



        relative_path = markdown_path.relative_to(
            INPUT_DIR
        )

        # La primera carpeta indica la tipología del documento: paper, guideline o ficha_tecnica
        document_type = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "unknown"
        )

        # Construir un identificador único a partir de la ruta
        doc_id = "_".join(
            relative_path.with_suffix("").parts
        )

        documents.append(
            {
                "doc_id": doc_id,
                "document_type": document_type,
                "file_name": markdown_path.name,
                "source_path": str(relative_path),
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            "No hay documentos válidos para procesar."
        )

    doc_ids = [document["doc_id"] for document in documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(
            "Hay documentos con doc_id duplicado; sus archivos de salida "
            "se sobrescribirían al procesarlos."
        )

    print(
        f"Documentos válidos: {len(documents)}"
    )

    # --------------------------------------------------------
    # 3. Cargar la configuración
    # --------------------------------------------------------

    chunker_config = load_config(CHUNKING_CONFIG_FILE)

    tokenizer_config = load_config(TOKENIZER_CONFIG_FILE)
    execution_config = chunker_config.get("execution", {})
    requested_workers = (
        workers_override
        if workers_override is not None
        else int(execution_config.get("workers", 1))
    )
    if requested_workers < 1:
        raise ValueError("El número de workers debe ser al menos 1.")
    available_cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(requested_workers, available_cpus, len(documents))

    configured_gpu_id = execution_config.get("gpu_id")
    gpu_id = (
        gpu_id_override
        if gpu_id_override is not None
        else configured_gpu_id
    )
    if gpu_id is not None and (
        not isinstance(gpu_id, int)
        or isinstance(gpu_id, bool)
        or gpu_id < 0
    ):
        raise ValueError("gpu_id debe ser un entero no negativo o null.")

    # Modelo cuyo tokenizador se utilizará de forma común para contar los tokens
    tokenizer_model_name = tokenizer_config["tokenizer"]["model_name"]

    if workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(1)
    tokenizer = create_tokenizer(
        tokenizer_model_name
    )

    token_counter = create_token_counter(tokenizer)

    # Contar los tokens de cada documento original
    for document in documents:
        document["token_count"] = token_counter(
            document["text"]
        )

    strategies = chunker_config["strategies"]

    enabled_strategies = [
        strategy
        for strategy in strategies
        if strategy.get("enabled", True)
    ]

    if not enabled_strategies:
        raise ValueError(
            "No hay ninguna estrategia activada "
            "en el archivo de configuración."
        )

    needs_gpu = any(
        strategy.get("spec", {}).get("type") in GPU_EMBEDDING_CHUNKERS
        for strategy in enabled_strategies
    )
    device = _resolve_device(gpu_id, needs_gpu)

    print(
        f"Estrategias activadas: "
        f"{len(enabled_strategies)}"
    )
    print(
        f"Workers de documentos: {workers}; "
        f"GPU para chunkers semánticos: {device}"
    )
    if device != "cpu" and workers > 1:
        print(
            "Cada worker de las estrategias semánticas carga su propia "
            "instancia del modelo en la GPU seleccionada."
        )

    if workers > 1 and any(
        strategy.get("spec", {}).get("type") in NLTK_CHUNKERS
        and strategy.get("spec", {}).get("type") not in LLM_CHUNKERS
        for strategy in enabled_strategies
    ):
        _ensure_nltk_data()

    # El CSV tendrá una fila por método y documento
    summary_results = []

    # --------------------------------------------------------
    # 4. Ejecutar las estrategias activadas
    # --------------------------------------------------------

    for strategy in enabled_strategies:

        strategy_name = strategy.get("name")
        spec = strategy.get("spec", {})
        chunker_type = spec.get("type")
        params = spec.get("params", {})
        strategy_workers = (
            1
            if chunker_type in LLM_CHUNKERS
            else workers
        )

        print("\n" + "=" * 70)
        print(f"Estrategia: {strategy_name}")
        print(f"Clase: {chunker_type}")
        print(f"Parámetros: {params}")
        print(f"Workers para esta estrategia: {strategy_workers}")
        print("=" * 70)

        # ----------------------------------------------------
        # Validar la estrategia
        # ----------------------------------------------------

        if not strategy_name:
            print(
                "[ERROR] La estrategia no contiene "
                "el campo 'name'."
            )
            continue

        if not chunker_type:
            print(
                f"[ERROR] La estrategia {strategy_name} "
                f"no contiene el campo 'type'."
            )
            continue

        if not isinstance(params, dict):
            print(
                f"[ERROR] Los parámetros de "
                f"{strategy_name} deben ser un diccionario."
            )
            continue

        if chunker_type not in CHUNKER_CLASSES:
            print(
                f"[ERROR] Tipo de chunker no reconocido: "
                f"{chunker_type}"
            )
            continue

        if chunker_type not in CHUNKER_ABBREVIATIONS:
            print(
                f"[ERROR] No se ha definido una abreviatura "
                f"para la clase {chunker_type}."
            )
            continue

        abbreviation = CHUNKER_ABBREVIATIONS[
            chunker_type
        ]

        completed_documents = 0
        strategy_chunk_count = 0
        initialization_error_recorded = False

        def record_document_result(
            document: Dict[str, Any],
            result: Dict[str, Any],
        ) -> None:
            nonlocal completed_documents, strategy_chunk_count
            nonlocal initialization_error_recorded

            print(f"\nProcesando: {document['file_name']}")
            if "initialization_error" in result:
                error_message = result["initialization_error"]
                print(
                    f"[ERROR] No se ha podido crear "
                    f"{strategy_name}: {error_message}"
                )
                if not initialization_error_recorded:
                    save_json(
                        data={
                            "strategy": strategy_name,
                            "type": chunker_type,
                            "abbreviation": abbreviation,
                            "params": params,
                            "status": "initialization_error",
                            "error": error_message,
                        },
                        output_path=(
                            METRICS_DIR
                            / f"{abbreviation}_initialization_error.json"
                        ),
                    )
                    initialization_error_recorded = True
                return

            if "error" in result:
                print(
                    f"[ERROR] Ha fallado {strategy_name} "
                    f"con {document['file_name']}: {result['error']}"
                )
                document_error = {
                    "strategy": strategy_name,
                    "type": chunker_type,
                    "abbreviation": abbreviation,
                    "params": params,
                    "status": "error",
                    "doc_id": document["doc_id"],
                    "input_file": document["file_name"],
                    "error": result["error"],
                }
                save_json(
                    data=document_error,
                    output_path=(
                        METRICS_DIR
                        / (
                            f"{abbreviation}_"
                            f"{document['doc_id']}_error.json"
                        )
                    ),
                )
                return

            document_chunks = result["chunks"]
            execution_time_seconds = result["execution_time_seconds"]
            ram_before_mb = result["ram_before_mb"]
            ram_after_mb = result["ram_after_mb"]

            # ------------------------------------------------
            # Serializar los chunks
            # ------------------------------------------------

            serialized_chunks = []

            for chunk in document_chunks:

                # Pydantic v2
                if hasattr(chunk, "model_dump"):
                    chunk_data = chunk.model_dump()

                # Pydantic v1
                elif hasattr(chunk, "dict"):
                    chunk_data = chunk.dict()

                # Fallback
                else:
                    chunk_data = {
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                    }

                serialized_chunks.append(
                    chunk_data
                )

            # ------------------------------------------------
            # 6. Guardar chunks por documento
            # ------------------------------------------------

            document_result = {
                "strategy": strategy_name,
                "type": chunker_type,
                "abbreviation": abbreviation,
                "params": params,
                "doc_id": document["doc_id"],
                "document_type": document["document_type"],
                "input_file": document["file_name"],
                "source_path": document["source_path"],
                "tokenizer_model": tokenizer_model_name,
                "original_token_count": document["token_count"],
                "number_of_chunks": len(
                    document_chunks
                ),
                "chunks": serialized_chunks,
            }

            chunks_file_name = (
                f"{abbreviation}_"
                f"{document['doc_id']}.json"
            )

            chunks_output_path = (
                CHUNKS_DIR / chunks_file_name
            )

            save_json(
                data=document_result,
                output_path=chunks_output_path,
            )

            # ------------------------------------------------
            # 7. Calcular métricas por documento
            # ------------------------------------------------

            document_stats = calculate_stats(
                chunks=document_chunks,
                original_token_count=(
                    document["token_count"]
                ),
                execution_time_seconds=(
                    execution_time_seconds
                ),
                ram_before_mb=ram_before_mb or 0.0,
                ram_after_mb=ram_after_mb or 0.0,
                token_counter=token_counter,
            )
            if ram_before_mb is None or ram_after_mb is None:
                document_stats["ram_before_mb"] = None
                document_stats["ram_after_mb"] = None
                document_stats["ram_increase_mb"] = None

            document_metrics_result = {
                "strategy": strategy_name,
                "type": chunker_type,
                "abbreviation": abbreviation,
                "params": params,
                "status": "completed",
                "doc_id": document["doc_id"],
                "document_type": document["document_type"],
                "input_file": document["file_name"],
                "source_path": document["source_path"],
                "tokenizer_model": tokenizer_model_name,
                **document_stats,
            }

            metrics_file_name = (
                f"{abbreviation}_"
                f"{document['doc_id']}_metrics.json"
            )

            metrics_output_path = (
                METRICS_DIR / metrics_file_name
            )

            save_json(
                data=document_metrics_result,
                output_path=metrics_output_path,
            )

            # Una fila del CSV por método y documento
            summary_results.append(
                document_metrics_result
            )

            completed_documents += 1
            strategy_chunk_count += len(
                document_chunks
            )

            print(
                f"[OK] Chunks: "
                f"{chunks_output_path.name}"
            )
            print(
                f"[OK] Métricas: "
                f"{metrics_output_path.name}"
            )
            print(
                f"[OK] Número de chunks: "
                f"{len(document_chunks)}"
            )
            print(
                f"[OK] Tiempo: "
                f"{execution_time_seconds:.6f} segundos"
            )
            if ram_before_mb is not None and ram_after_mb is not None:
                print(
                    f"[OK] Incremento de RAM: "
                    f"{ram_after_mb - ram_before_mb:.2f} MB"
                )
            else:
                print("[INFO] RAM por documento no medida en modo paralelo.")

        if strategy_workers == 1:
            _initialize_document_worker(
                chunker_type,
                params,
                tokenizer_model_name,
                device,
                True,
            )
            initialization_error = _WORKER_CONTEXT.initialization_error
            if initialization_error is not None:
                record_document_result(
                    documents[0],
                    {"initialization_error": initialization_error},
                )
                continue
            for document in documents:
                record_document_result(document, _chunk_document(document))
        else:
            with ThreadPoolExecutor(
                max_workers=strategy_workers,
                initializer=_initialize_document_worker,
                initargs=(
                    chunker_type,
                    params,
                    tokenizer_model_name,
                    device,
                    False,
                ),
                thread_name_prefix="markdown-chunker",
            ) as executor:
                for document, result in zip(
                    documents,
                    executor.map(_chunk_document, documents),
                ):
                    record_document_result(document, result)

        # ----------------------------------------------------
        # Resumen de la estrategia
        # ----------------------------------------------------

        print(
            f"\n[OK] Estrategia completada: "
            f"{strategy_name}"
        )
        print(
            f"[OK] Documentos procesados: "
            f"{completed_documents}/{len(documents)}"
        )
        print(
            f"[OK] Chunks totales generados: "
            f"{strategy_chunk_count}"
        )

    # --------------------------------------------------------
    # 8. Crear el CSV resumen
    # --------------------------------------------------------

    if summary_results:
        summary_df = pd.DataFrame(
            summary_results
        )

        # Guardar los parámetros como texto JSON
        if "params" in summary_df.columns:
            summary_df["params"] = (
                summary_df["params"].apply(
                    lambda value: json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                )
            )

        # Ordenar por documento y método
        summary_df = summary_df.sort_values(
            by=[
                "doc_id",
                "strategy",
            ]
        ).reset_index(drop=True)

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 70)
        print(
            f"Resumen guardado en: "
            f"{SUMMARY_FILE}"
        )
        print(
            f"Filas del resumen: "
            f"{len(summary_df)}"
        )

    else:
        print(
            "\nNo se ha podido completar "
            "ninguna combinación de método y documento."
        )

    print("=" * 70)
    print("FIN DE LA EJECUCIÓN")
    print("=" * 70)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta las estrategias de chunking Markdown."
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="Máximo de documentos procesados en paralelo (configurable).",
    )
    parser.add_argument(
        "--gpu-id",
        type=_nonnegative_int,
        default=None,
        help="ID de la única GPU CUDA usada por las estrategias semánticas.",
    )
    arguments = parser.parse_args()
    main(
        workers_override=arguments.workers,
        gpu_id_override=arguments.gpu_id,
    )