# ============================================================
# IMPORTS GENERALES
# ============================================================

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import psutil

# Utilidades
from utilities import (
    calculate_stats,
    create_tokenizer,
    create_token_counter,
    load_config,
    save_json,
)

# ============================================================
# JSON-CHUNKER
# ============================================================

from chunkers.json.JSON_chunker import (
    HierarchicalJSONChunker,
)

# Document plans
from chunkers.json.document_plan_ficha_tecnica import DOCUMENT_PLAN_FC
from chunkers.json.document_plan_guideline import DOCUMENT_PLAN_GL
from chunkers.json.document_plan_paper import DOCUMENT_PLAN_PAPER

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
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:

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

    tokenizer = create_tokenizer(model_name)
    token_counter = create_token_counter(
        tokenizer,
        add_special_tokens=add_special_tokens,
    )

    for doc in documents:
        doc["token_count"] = token_counter(doc["full_text"])

    # Estrategia única para JSON
    strategy_name = "JSONChunking"
    abbreviation = "JC"

    summary_results = []
    process = psutil.Process(os.getpid())

    # 4. Ejecutar chunking JSON
    for doc in documents:

        print(f"\nProcesando: {doc['file_name']}")

        plan = select_document_plan(doc["document_type"])

        chunker = HierarchicalJSONChunker(
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
            document_plan=plan,
            root_field=None,
            indivisible_list_paths=INDIVISIBLE_LIST_PATHS,
        )

        ram_before = process.memory_info().rss / (1024**2)
        start = time.perf_counter()

        chunks = chunker.chunk_document(doc["data"])

        exec_time = time.perf_counter() - start
        ram_after = process.memory_info().rss / (1024**2)

        # Serializar los dataclasses Chunk generados por el chunker JSON
        serialized = [asdict(chunk) for chunk in chunks]

        # Guardar chunks
        result = {
            "strategy": strategy_name,
            "type": "HierarchicalJSONChunker",
            "abbreviation": abbreviation,
            "doc_id": doc["doc_id"],
            "document_type": doc["document_type"],
            "input_file": doc["file_name"],
            "source_path": doc["source_path"],
            "tokenizer_model": model_name,
            "params": {
                "target_tokens": target_tokens,
                "max_tokens": max_tokens,
            },
            "original_token_count": doc["token_count"],
            "number_of_chunks": len(chunks),
            "chunks": serialized,
        }

        save_json(result, CHUNKS_DIR / f"{abbreviation}_{doc['doc_id']}.json")

        # Métricas
        stats = calculate_stats(
            chunks=chunks,
            original_token_count=doc["token_count"],
            execution_time_seconds=exec_time,
            ram_before_mb=ram_before,
            ram_after_mb=ram_after,
            token_counter=token_counter,
        )

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
            "params": {
                "target_tokens": target_tokens,
                "max_tokens": max_tokens,
            },
            **stats,
        }

        save_json(metrics, METRICS_DIR / f"{abbreviation}_{doc['doc_id']}_metrics.json")

        summary_results.append(metrics)

    # 5. Summary CSV
    if summary_results:
        df = pd.DataFrame(summary_results)
        SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("FIN DE LA EJECUCIÓN (JSON)")
    print("=" * 70)

# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
