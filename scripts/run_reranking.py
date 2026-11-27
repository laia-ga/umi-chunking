
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURACIÓN
# ============================================================

def load_reranking_config(config_file):
    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config["reranking"]


# ============================================================
# RERANKING DE CHUNKS
# ============================================================

def rerank_hits(question, hits, reranker, batch_size, candidate_k):

    if not hits:
        return [], 0.0

    # Orden original del retrieval
    ordered_hits = sorted(
        hits,
        key=lambda hit: int(hit["rank"]),
    )

    # Seleccionar los primeros candidate_k
    selected_hits = ordered_hits[:candidate_k]

    if any(
        not isinstance(hit.get("text"), str)
        or not hit["text"].strip()
        for hit in selected_hits
    ):
        raise ValueError("Hay chunks sin texto para reranking.")

    pairs = [
        (question, hit["text"])
        for hit in selected_hits
    ]

    start = time.perf_counter()

    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    scores = np.asarray(scores)

    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]

    if scores.ndim != 1 or len(scores) != len(selected_hits):
        raise ValueError(
            "El reranker debe devolver un score por chunk."
        )

    reordered = []

    for hit, score in zip(selected_hits, scores):
        item = copy.deepcopy(hit)
        item["retrieval_rank"] = hit["rank"]
        item["retrieval_score"] = hit.get("score")
        item["rerank_score"] = float(score)
        reordered.append(item)

    reordered.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    for rank, item in enumerate(reordered, start=1):
        item["rank"] = rank

    return reordered, elapsed


# ============================================================
# RERANKING POR PREGUNTA
# ============================================================

def rerank_question(
    result,
    reranker,
    batch_size,
    candidate_k,
):

    question = result["question"]
    total_seconds = 0.0

    if "strategies" in result:
        reranked_strategies = {}

        for strategy, hits in result["strategies"].items():
            reordered, elapsed = rerank_hits(
                question,
                hits,
                reranker,
                batch_size,
                candidate_k,
            )

            reranked_strategies[strategy] = reordered
            total_seconds += elapsed

        return {"strategies": reranked_strategies}, total_seconds

    if "retrieved_chunks" in result:
        reordered, elapsed = rerank_hits(
            question,
            result["retrieved_chunks"],
            reranker,
            batch_size,
            candidate_k,
        )

        return {"retrieved_chunks": reordered}, elapsed

    raise ValueError("Estructura de retrieval desconocida.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reranking_config.json"),
    )

    args = parser.parse_args()

    config_file = args.config.resolve()
    config = load_reranking_config(config_file)

    # Se asume que el JSON está en PROJECT_ROOT/configs/
    project_root = config_file.parent.parent

    input_file = project_root / config["input"]
    output_base = project_root / config["output"]

    model_name = config["model"]
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    candidate_k = int(config["candidate_k"])
    output_k = sorted(set(config["output_k"]))

    if batch_size < 1 or max_length < 1 or candidate_k < 1:
        raise ValueError(
            "batch_size, max_length y candidate_k deben ser positivos."
        )

    if not output_k or any(
        not isinstance(k, int) or isinstance(k, bool)
        or k < 1 or k > candidate_k
        for k in output_k
    ):
        raise ValueError(
            "output_k debe contener enteros entre 1 y candidate_k."
        )

    configured_device = config.get("device", "auto")

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if configured_device == "auto" else configured_device

    print(f"Modelo: {model_name}")
    print(f"Dispositivo: {device}")
    print(f"Candidatos: {candidate_k}")
    print(f"Top-k finales: {output_k}")

    # --------------------------------------------------------
    # 1. Cargar modelo
    # --------------------------------------------------------

    reranker = CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
    )

    # --------------------------------------------------------
    # 2. Cargar retrieval
    # --------------------------------------------------------

    with input_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError("Se esperaba una lista de preguntas.")

    print(f"Preguntas: {len(results)}")

    # --------------------------------------------------------
    # 3. Preparar salidas
    # --------------------------------------------------------

    outputs = {k: [] for k in output_k}
    total_seconds = 0.0

    # --------------------------------------------------------
    # 4. Reranking
    # --------------------------------------------------------

    for index, result in enumerate(results, start=1):

        reranked_data, elapsed = rerank_question(
            result,
            reranker,
            batch_size,
            candidate_k,
        )

        total_seconds += elapsed

        for k in output_k:
            output_result = copy.deepcopy(result)

            if "strategies" in reranked_data:
                output_result["strategies"] = {
                    strategy: copy.deepcopy(hits[:k])
                    for strategy, hits
                    in reranked_data["strategies"].items()
                }
            else:
                output_result["retrieved_chunks"] = (
                    copy.deepcopy(
                        reranked_data["retrieved_chunks"][:k]
                    )
                )

            output_result["reranker_model"] = model_name
            output_result["reranking_candidates"] = candidate_k
            output_result["reranking_top_k"] = k
            output_result["reranking_seconds"] = elapsed

            outputs[k].append(output_result)

        if index % 10 == 0 or index == len(results):
            print(
                f"Preguntas procesadas: {index}/{len(results)}"
            )

    # --------------------------------------------------------
    # 5. Guardar resultados
    # --------------------------------------------------------

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for k, output_results in outputs.items():

        output_file = output_base.with_name(
            f"{output_base.stem}_c{candidate_k}_top{k}.json"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output_results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Guardado: {output_file}")

    print("=" * 60)
    print("RERANKING FINALIZADO")
    print(f"Tiempo de inferencia: {total_seconds:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
