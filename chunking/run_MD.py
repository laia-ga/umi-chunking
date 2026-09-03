# ============================================================
# IMPORTS GENERALES
# ============================================================

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List
from transformers import AutoTokenizer 
import numpy as np
import pandas as pd
import psutil

# ============================================================
# CLASES BASE
# ============================================================

from chunkers.base import Chunk

# ============================================================
# MÉTODOS RULE-BASED
# ============================================================

from chunkers.rulebased.fixed_character_chunking import FixedCharacterChunker
from chunkers.rulebased.fixed_token_chunking import FixedTokenChunker
from chunkers.rulebased.length_aware_chunking import LengthAwareChunker
from chunkers.rulebased.overlapping_token_chunking import OverlappingTokenChunker
from chunkers.rulebased.paragraph_based_chunking import ParagraphBasedChunker
from chunkers.rulebased.paragraph_group_chunking import ParagraphGroupChunker
from chunkers.rulebased.sentence_based_chunking import SentenceBasedChunker
from chunkers.rulebased.sentence_group_chunking import SentenceGroupChunker
from chunkers.rulebased.sliding_window_token_chunking import SlidingWindowTokenChunker

# ============================================================
# MÉTODOS RECURSIVOS
# ============================================================

from chunkers.recursive.recursive_chunking import RecursiveChunker
from chunkers.recursive.recursive_token_chunking import RecursiveTokenChunker
from chunkers.recursive.parent_child_chunking import ParentChildChunker

# ============================================================
# MÉTODOS SEMÁNTICOS
# ============================================================

from chunkers.semantic.semantic_boundary_chunking import SemanticBoundaryChunker
from chunkers.semantic.semantic_embedding_chunking import SemanticEmbeddingChunker
from chunkers.semantic.semantic_similarity_threshold_chunking import SemanticSimilarityThresholdChunker
from chunkers.semantic.topic_based_chunking import TopicBasedChunker

# ============================================================
# MÉTODOS DINÁMICOS
# ============================================================

from chunkers.dynamic.content_density_adaptive_chunking import ContentDensityAdaptiveChunker
from chunkers.dynamic.dynamic_token_size_chunking import DynamicTokenSizeChunker
from chunkers.dynamic.semantic_variance_adaptive_chunking import SemanticVarianceAdaptiveChunker

# ============================================================
# MÉTODOS BASADOS EN LLM
# ============================================================

from chunkers.llm.llm_boundary_detection_chunking import LLMBoundaryDetectionChunker
from chunkers.llm.llm_segment_then_chunking import LLMSegmentThenChunker

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
CONFIG_FILE = BASE_DIR / "configs" / "chunker_config.json"

# Archivo JSON con la configuración del tokenizador utilizado para el conteo:
# main/configs
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
# FUNCIÓN PARA CARGAR LA CONFIGURACIÓN
# ============================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Carga y valida el archivo JSON de configuración.

    Parameters
    ----------
    config_path:
        Ruta al archivo chunker_config.json.

    Returns
    -------
    Dict[str, Any]
        Configuración completa cargada desde el JSON.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo de configuración: "
            f"{config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"La ruta de configuración no corresponde a un archivo: "
            f"{config_path}"
        )

    try:
        with config_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"El archivo de configuración no contiene un JSON válido. "
            f"Error en la línea {error.lineno}, "
            f"columna {error.colno}: {error.msg}"
        ) from error

    if not isinstance(config, dict):
        raise ValueError(
            "La configuración debe ser un objeto JSON."
        )

    strategies = config.get("strategies")

    if not isinstance(strategies, list):
        raise ValueError(
            "La configuración debe contener una lista llamada "
            "'strategies'."
        )

    return config

# ============================================================
# FUNCIÓN PARA GUARDAR ARCHIVOS JSON
# ============================================================

def save_json(
    data: Any,
    output_path: Path,
) -> None:
    """
    Guarda datos de Python en un archivo JSON.

    Parameters
    ----------
    data:
        Datos que se quieren guardar. Pueden ser un diccionario,
        una lista u otro objeto compatible con JSON.

    output_path:
        Ruta completa del archivo JSON de salida.
    """

    # Crear la carpeta de destino si todavía no existe
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with output_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except TypeError as error:
        raise TypeError(
            f"No se han podido guardar los datos en "
            f"{output_path}. Algún objeto no es compatible "
            f"con el formato JSON."
        ) from error

    except OSError as error:
        raise OSError(
            f"No se ha podido escribir el archivo: "
            f"{output_path}"
        ) from error


# ============================================================
# FUNCIÓN PARA CREAR EL CONTADOR DE TOKENS
# ============================================================

def create_token_counter(
    model_name: str,
) -> Callable[[str], int]:
    """
    Crea un contador utilizando el tokenizer del modelo
    de embeddings.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    def count_tokens(text: str) -> int:
        if not text or not text.strip():
            return 0

        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )

    return count_tokens


# ============================================================
# FUNCIÓN PARA CALCULAR ESTADÍSTICAS
# ============================================================

def calculate_stats(
    chunks: List[Chunk],
    original_token_count: int,
    execution_time_seconds: float,
    ram_before_mb: float,
    ram_after_mb: float,
    token_counter: Callable[[str], int],
) -> Dict[str, Any]:
    """
    Calcula estadísticas sobre el número de tokens de los chunks.

    Parameters
    ----------
    chunks:
        Lista de chunks producidos por una estrategia.

    original_token_count:
        Número total de tokens de los documentos originales.

    execution_time_seconds:
        Tiempo empleado por la estrategia.

    ram_before_mb:
        Memoria RAM utilizada antes de ejecutar la estrategia.

    ram_after_mb:
        Memoria RAM utilizada después de ejecutar la estrategia.

    token_counter:
        Función utilizada para contar tokens con el tokenizer
        del modelo de embeddings.

    Returns
    -------
    Dict[str, Any]
        Diccionario con las estadísticas calculadas.
    """

    # Conservar únicamente chunks con texto
    valid_chunks = [
        chunk
        for chunk in chunks
        if chunk.text and chunk.text.strip()
    ]

    # Contar los tokens de cada chunk
    lengths = [
        token_counter(chunk.text.strip())
        for chunk in valid_chunks
    ]

    # Número de chunks sin texto
    empty_chunks = len(chunks) - len(valid_chunks)

    # Si no hay ningún chunk con texto, devolver valores vacíos
    if not lengths:
        return {
            "number_of_chunks": 0,
            "empty_chunks": empty_chunks,
            "tokens_min": 0,
            "tokens_p25": 0,
            "tokens_median": 0,
            "tokens_mean": 0,
            "tokens_p75": 0,
            "tokens_max": 0,
            "tokens_std": 0,
            "tokens_total": 0,
            "tokens_original": original_token_count,
            "tokens_ratio": 0,
            "execution_time_seconds": round(
                execution_time_seconds,
                6,
            ),
            "ram_before_mb": round(
                ram_before_mb,
                2,
            ),
            "ram_after_mb": round(
                ram_after_mb,
                2,
            ),
            "ram_increase_mb": round(
                ram_after_mb - ram_before_mb,
                2,
            ),
        }

    # Número total de tokens de todos los chunks
    total_tokens = sum(lengths)

    return {
        "number_of_chunks": len(valid_chunks),
        "empty_chunks": empty_chunks,

        "tokens_min": min(lengths),

        "tokens_p25": round(
            float(np.percentile(lengths, 25)),
            2,
        ),

        "tokens_median": round(
            float(np.median(lengths)),
            2,
        ),

        "tokens_mean": round(
            float(np.mean(lengths)),
            2,
        ),

        "tokens_p75": round(
            float(np.percentile(lengths, 75)),
            2,
        ),

        "tokens_max": max(lengths),

        "tokens_std": round(
            float(np.std(lengths)),
            2,
        ),

        "tokens_total": total_tokens,

        "tokens_original": original_token_count,

        "tokens_ratio": round(
            total_tokens / original_token_count,
            3,
        )
        if original_token_count > 0
        else 0,

        "execution_time_seconds": round(
            execution_time_seconds,
            6,
        ),

        "ram_before_mb": round(
            ram_before_mb,
            2,
        ),

        "ram_after_mb": round(
            ram_after_mb,
            2,
        ),

        "ram_increase_mb": round(
            ram_after_mb - ram_before_mb,
            2,
        ),
    }

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
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

    print(
        f"Documentos válidos: {len(documents)}"
    )

    # --------------------------------------------------------
    # 3. Cargar la configuración
    # --------------------------------------------------------

    config = load_config(CONFIG_FILE)

    embedding_model_name = config["embedding_model"]["name"]

    token_counter = create_token_counter(embedding_model_name)

    # Contar los tokens de cada documento original
    for document in documents:
        document["character_count"] = token_counter(
            document["text"]
        )

    strategies = config["strategies"]

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

    print(
        f"Estrategias activadas: "
        f"{len(enabled_strategies)}"
    )

    # El CSV tendrá una fila por método y documento
    summary_results = []

    # Proceso actual para medir RAM
    process = psutil.Process(os.getpid())

    # --------------------------------------------------------
    # 4. Ejecutar las estrategias activadas
    # --------------------------------------------------------

    for strategy in enabled_strategies:

        strategy_name = strategy.get("name")
        spec = strategy.get("spec", {})
        chunker_type = spec.get("type")
        params = spec.get("params", {})

        print("\n" + "=" * 70)
        print(f"Estrategia: {strategy_name}")
        print(f"Clase: {chunker_type}")
        print(f"Parámetros: {params}")
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

        chunker_class = CHUNKER_CLASSES[
            chunker_type
        ]

        abbreviation = CHUNKER_ABBREVIATIONS[
            chunker_type
        ]

        # ----------------------------------------------------
        # Crear el chunker una sola vez por estrategia
        # ----------------------------------------------------

        try:
            chunker = chunker_class(**params)

        except Exception as error:
            print(
                f"[ERROR] No se ha podido crear "
                f"{strategy_name}: {error}"
            )

            error_result = {
                "strategy": strategy_name,
                "type": chunker_type,
                "abbreviation": abbreviation,
                "params": params,
                "status": "initialization_error",
                "error": str(error),
            }

            save_json(
                data=error_result,
                output_path=(
                    METRICS_DIR
                    / f"{abbreviation}_initialization_error.json"
                ),
            )

            continue

        completed_documents = 0
        strategy_chunk_count = 0

        # ----------------------------------------------------
        # 5. Procesar cada documento por separado
        # ----------------------------------------------------

        for document in documents:

            print(
                f"\nProcesando: {document['file_name']}"
            )

            # ------------------------------------------------
            # Medición inicial por documento
            # ------------------------------------------------

            ram_before_mb = (
                process.memory_info().rss
                / (1024 ** 2)
            )

            start_time = time.perf_counter()

            try:
                document_chunks = list(
                    chunker.chunk(
                        text=document["text"],
                        doc_id=document["doc_id"],
                    )
                )

            except Exception as error:
                print(
                    f"[ERROR] Ha fallado {strategy_name} "
                    f"con {document['file_name']}: {error}"
                )

                document_error = {
                    "strategy": strategy_name,
                    "type": chunker_type,
                    "abbreviation": abbreviation,
                    "params": params,
                    "status": "error",
                    "doc_id": document["doc_id"],
                    "input_file": document["file_name"],
                    "error": str(error),
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

                continue

            execution_time_seconds = (
                time.perf_counter() - start_time
            )

            ram_after_mb = (
                process.memory_info().rss
                / (1024 ** 2)
            )

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
                "embedding_model": embedding_model_name,
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
                ram_before_mb=ram_before_mb,
                ram_after_mb=ram_after_mb,
                token_counter=token_counter,
            )

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
                "embedding_model": embedding_model_name,
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
            print(
                f"[OK] Incremento de RAM: "
                f"{ram_after_mb - ram_before_mb:.2f} MB"
            )

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
    main()