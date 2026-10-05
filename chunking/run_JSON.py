# ============================================================
# CHUNKING JSON
# ============================================================

"""
Ejecuta el chunking jerárquico para documentos JSON estructurados.

Este script:
1. Carga los documentos JSON desde data/raw/json.
2. Detecta automáticamente la tipología documental (paper, guideline,
   ficha técnica) según la carpeta de origen.
3. Selecciona el document plan correspondiente.
4. Aplica el HierarchicalJSONChunker, que:
   - respeta la estructura del JSON,
   - evita cortes peligrosos,
   - genera chunks semánticos y trazables,
   - añade metadatos como nivel, grupo, rutas JSON y límites de tokens.
5. Guarda los chunks en output/chunks/json/chunk_results.
6. Guarda las métricas en output/chunks/json/metrics.
7. Genera un summary.csv con una fila por documento.
"""


# ============================================================
# IMPORTS GENERALES
# ============================================================

import json
import os
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import psutil

# Utilidades
from chunking.utilities import (
    calculate_stats,
    create_tokenizer,
    create_token_counter,
    load_config,
    save_json,
)

# ============================================================
# JSON-CHUNKER
# ============================================================

from chunking.chunkers.json.JSON_chunker import (
    HierarchicalJSONChunker,
)

# Document plans
from chunking.chunkers.json.document_plan_ficha_tecnica import DOCUMENT_PLAN_FC
from chunking.chunkers.json.document_plan_guideline import DOCUMENT_PLAN_GL
from chunking.chunkers.json.document_plan_paper import DOCUMENT_PLAN_PAPER

# ============================================================
# RUTAS
# ============================================================

# Carpeta donde se encuentra el archivo run_JSON.py:
# main/chunking
CHUNKING_DIR = Path(__file__).resolve().parent

# Carpeta raíz del proyecto:
# main
BASE_DIR = CHUNKING_DIR.parent

# Archivo JSON con la configuración del tokenizador utilizado para el conteo:
# main/configs
TOKENIZER_CONFIG_FILE = BASE_DIR / "configs" / "tokenizer_config.json"

# Carpeta que contiene los scripts de análisis:
# main/scripts
SCRIPTS_DIR = BASE_DIR / "scripts"

# Carpeta que contiene los documentos JSON de entrada
# main/data/raw/json
INPUT_DIR = BASE_DIR / "data" / "raw" / "json"

# Carpeta general de salida
OUTPUT_DIR = BASE_DIR / "output" / "chunks" / "json"

# Carpeta donde se guardarán los chunks generados
CHUNKS_DIR = OUTPUT_DIR / "chunk_results"

# Carpeta donde se guardarán las métricas de cada método
METRICS_DIR = OUTPUT_DIR / "metrics"

# Archivo resumen con una fila por método
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

# ============================================================
# SELECCIÓN DEL DOCUMENT PLAN
# ============================================================

DOCUMENT_PLANS = {
    "paper": DOCUMENT_PLAN_PAPER,
    "guideline": DOCUMENT_PLAN_GL,
    "ficha_tecnica": DOCUMENT_PLAN_FC,
}


def select_document_plan(document_type: str):
    try:
        return DOCUMENT_PLANS[document_type]
    except KeyError as error:
        available_types = ", ".join(DOCUMENT_PLANS)
        raise ValueError(
            f"Tipo de documento no soportado: {document_type!r}. "
            f"Tipos disponibles: {available_types}"
        ) from error

# INVIDISIBLE LIST PATHS

INDIVISIBLE_LIST_PATHS = {

    "paper": ["authors", "keywords"],

    "guideline": ["toc", "references", "abbreviations", "metadata.authors"],

    "ficha_tecnica": ["excipients","national_codes","national_codes_queried"],
}

# ============================================================
# PARALELIZACIÓN DE DOCUMENTOS
# ============================================================

_WORKER_CONTEXT = threading.local()


def _initialize_worker(
    model_name: str,
    add_special_tokens: bool,
) -> None:
    tokenizer = create_tokenizer(model_name)
    _WORKER_CONTEXT.token_counter = create_token_counter(
        tokenizer,
        add_special_tokens=add_special_tokens,
    )


def _process_document(
    doc: Dict[str, Any],
    target_tokens: int,
    max_tokens: int,
    measure_ram: bool,
) -> Dict[str, Any]:
    token_counter = _WORKER_CONTEXT.token_counter
    original_token_count = token_counter(doc["full_text"])
    chunker = HierarchicalJSONChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        token_counter=token_counter,
        document_plan=select_document_plan(doc["document_type"]),
        root_field=None,
        indivisible_list_paths=INDIVISIBLE_LIST_PATHS,
    )

    process = psutil.Process(os.getpid()) if measure_ram else None
    ram_before = (
        process.memory_info().rss / (1024 ** 2)
        if process is not None
        else None
    )
    start = time.perf_counter()
    chunks = chunker.chunk_document(doc["data"])
    execution_time = time.perf_counter() - start
    ram_after = (
        process.memory_info().rss / (1024 ** 2)
        if process is not None
        else None
    )

    stats = calculate_stats(
        chunks=chunks,
        original_token_count=original_token_count,
        execution_time_seconds=execution_time,
        ram_before_mb=ram_before if ram_before is not None else 0.0,
        ram_after_mb=ram_after if ram_after is not None else 0.0,
        token_counter=token_counter,
    )
    if ram_before is None or ram_after is None:
        stats["ram_before_mb"] = None
        stats["ram_after_mb"] = None
        stats["ram_increase_mb"] = None

    return {
        "chunks": chunks,
        "original_token_count": original_token_count,
        "stats": stats,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El número de workers debe ser un entero positivo."
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "El número de workers debe ser al menos 1."
        )
    return parsed

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(workers_override: int | None = None) -> None:

    print("=" * 70)
    print("INICIO DE LA EJECUCIÓN (JSON)")
    print("=" * 70)

    # 1. Cargar JSON
    json_files = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".json"
    )

    documents = []
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rel = path.relative_to(INPUT_DIR)
        doc_type = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        doc_id = "_".join(rel.with_suffix("").parts)

        full_text = json.dumps(data, ensure_ascii=False)

        documents.append({
            "doc_id": doc_id,
            "document_type": doc_type,
            "file_name": path.name,
            "source_path": str(rel),
            "data": data,
            "full_text": full_text,
        })

    # 2. Config + token counter
    tokenizer_cfg = load_config(TOKENIZER_CONFIG_FILE)
    model_name = tokenizer_cfg["tokenizer"]["model_name"]
    add_special_tokens = tokenizer_cfg["tokenizer"].get(
        "add_special_tokens",
        True,
    )
    target_tokens = tokenizer_cfg["tokenizer"]["target_tokens"]
    max_tokens = tokenizer_cfg["tokenizer"]["max_tokens"]

    # Estrategia única para JSON
    strategy_name = "JSONChunking"
    abbreviation = "JC"

    summary_results = []
    requested_workers = (
        workers_override
        if workers_override is not None
        else 1
    )
    if requested_workers < 1:
        raise ValueError("El número de workers debe ser al menos 1.")
    available_cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(requested_workers, available_cpus, max(1, len(documents)))
    print(f"Workers para chunking JSON: {workers}")

    # 4. Ejecutar chunking JSON
    if workers == 1:
        _initialize_worker(model_name, add_special_tokens)
        document_results = (
            _process_document(doc, target_tokens, max_tokens, True)
            for doc in documents
        )
        results = zip(documents, document_results)
        for doc, processed in results:
            _save_document_result(
                doc,
                processed,
                model_name,
                target_tokens,
                max_tokens,
                strategy_name,
                abbreviation,
                summary_results,
            )
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(model_name, add_special_tokens),
            thread_name_prefix="json-chunker",
        ) as executor:
            for doc, processed in zip(
                documents,
                executor.map(
                    lambda document: _process_document(
                        document,
                        target_tokens,
                        max_tokens,
                        False,
                    ),
                    documents,
                ),
            ):
                _save_document_result(
                    doc,
                    processed,
                    model_name,
                    target_tokens,
                    max_tokens,
                    strategy_name,
                    abbreviation,
                    summary_results,
                )

    # 5. Summary CSV
    if summary_results:
        df = pd.DataFrame(summary_results)
        SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("FIN DE LA EJECUCIÓN (JSON)")
    print("=" * 70)


def _save_document_result(
    doc: Dict[str, Any],
    processed: Dict[str, Any],
    model_name: str,
    target_tokens: int,
    max_tokens: int,
    strategy_name: str,
    abbreviation: str,
    summary_results: list[Dict[str, Any]],
) -> None:
    print(f"\nProcesando: {doc['file_name']}")
    chunks = processed["chunks"]
    serialized = [asdict(chunk) for chunk in chunks]
    params = {
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
    }

    result = {
        "strategy": strategy_name,
        "type": "HierarchicalJSONChunker",
        "abbreviation": abbreviation,
        "doc_id": doc["doc_id"],
        "document_type": doc["document_type"],
        "input_file": doc["file_name"],
        "source_path": doc["source_path"],
        "tokenizer_model": model_name,
        "params": params,
        "original_token_count": processed["original_token_count"],
        "number_of_chunks": len(chunks),
        "chunks": serialized,
    }

    save_json(result, CHUNKS_DIR / f"{abbreviation}_{doc['doc_id']}.json")

    metrics = {
        "strategy": strategy_name,
        "type": "HierarchicalJSONChunker",
        "abbreviation": abbreviation,
        "status": "completed",
        "doc_id": doc["doc_id"],
        "document_type": doc["document_type"],
        "input_file": doc["file_name"],
        "source_path": doc["source_path"],
        "tokenizer_model": model_name,
        "params": params,
        **processed["stats"],
    }

    save_json(
        metrics,
        METRICS_DIR / f"{abbreviation}_{doc['doc_id']}_metrics.json",
    )
    summary_results.append(metrics)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta el chunking jerárquico de documentos JSON."
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="Máximo de documentos procesados en paralelo (por defecto, 1).",
    )
    arguments = parser.parse_args()
    main(workers_override=arguments.workers)
