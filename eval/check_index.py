################################################################################
# CHECK_INDEX.PY
#
# Comprueba si el índice de Qdrant es coherente con los chunks normalizados:
#   1. ¿Hay chunk_id repetidos entre estrategias? (se sobrescribirían en Qdrant,
#      porque el id del punto se calcula a partir del chunk_id)
#   2. ¿Comparten abreviatura dos estrategias distintas?
#   3. Nº de chunks normalizados vs. nº de puntos en Qdrant, por estrategia.
#
# Uso (desde la raíz del proyecto, con el .venv activado):
#   python eval/check_index.py
#   python eval/check_index.py --normalized output/indexing/all_bge_m3/normalized_chunks.jsonl \
#                              --collection all_bge_m3
################################################################################

import argparse
import json
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized",
                        default="output/indexing/all_bge_m3/normalized_chunks.jsonl")
    parser.add_argument("--indexing-config", default="configs/indexing_config.json")
    parser.add_argument("--collection", default=None,
                        help="Nombre de la colección (por defecto, el del config).")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()

    path = BASE_DIR / args.normalized
    print(f"Leyendo {path}")
    df = pd.read_json(path, lines=True)
    print(f"Chunks normalizados: {len(df)}\n")

    # ---- 1. chunk_id repetidos --------------------------------------------
    dup_mask = df.duplicated("chunk_id", keep=False)
    n_dup = int(df.duplicated("chunk_id").sum())
    print(f"[1] chunk_id repetidos (filas que se sobrescribirían): {n_dup}")
    if n_dup:
        per_strategy = df[dup_mask].groupby("strategy").size().sort_values(ascending=False)
        print("    Filas implicadas por estrategia:")
        print(per_strategy.to_string())
    print()

    # ---- 2. abreviaturas compartidas --------------------------------------
    abbr = df.groupby("strategy_abbreviation")["strategy"].nunique()
    shared = abbr[abbr > 1]
    print(f"[2] Abreviaturas compartidas por varias estrategias: {len(shared)}")
    for a in shared.index:
        names = sorted(df.loc[df.strategy_abbreviation == a, "strategy"].unique())
        print(f"    {a}: {names}")
    print()

    # ---- 3. normalizados vs Qdrant ----------------------------------------
    normalized = df.groupby("strategy").size().rename("normalizados")
    if args.skip_qdrant:
        print(normalized.to_string())
        return
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        cfg = json.load(open(BASE_DIR / args.indexing_config, encoding="utf-8"))["qdrant"]
        collection = args.collection or cfg["collection_name"]
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", cfg["host"]),
            port=int(os.getenv("QDRANT_PORT", cfg["port"])),
            timeout=60,
        )
        total = client.count(collection_name=collection, exact=True).count
        print(f"[3] Colección '{collection}': {total} puntos "
              f"(chunks normalizados: {len(df)})")
        rows = []
        for strategy, n in normalized.items():
            count = client.count(
                collection_name=collection,
                count_filter=models.Filter(must=[
                    models.FieldCondition(key="strategy",
                                          match=models.MatchValue(value=strategy))
                ]),
                exact=True,
            ).count
            rows.append((strategy, n, count, count - n))
        out = pd.DataFrame(rows, columns=["strategy", "normalizados", "en_qdrant", "diferencia"])
        print(out.sort_values("diferencia").to_string(index=False))
    except Exception as error:
        print(f"[3] No se pudo consultar Qdrant: {error}")
        print(normalized.to_string())


if __name__ == "__main__":
    main()