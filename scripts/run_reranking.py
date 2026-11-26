
import argparse
import copy
import json
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder


def rerank_hits(question, hits, reranker, batch_size):
    if not hits:
        return [], 0.0

    if any(not isinstance(h.get("text"), str) or not h["text"].strip()
           for h in hits):
        raise ValueError("Hay chunks sin texto para reranking.")

    pairs = [(question, hit["text"]) for hit in hits]

    start = time.perf_counter()
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    reordered = []

    for hit, score in zip(hits, scores):
        item = copy.deepcopy(hit)
        item["retrieval_rank"] = hit["rank"]
        item["retrieval_score"] = hit.get("score")
        item["rerank_score"] = float(score)
        reordered.append(item)

    reordered.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    for new_rank, item in enumerate(reordered, start=1):
        item["rank"] = new_rank

    return reordered, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    reranker = CrossEncoder(
        args.model,
        device=device,
        max_length=args.max_length,
    )

    with args.input.open(encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError("Se esperaba una lista de preguntas.")

    for result in results:
        question = result["question"]
        total_seconds = 0.0

        # Mantener el esquema original por estrategias
        if "strategies" in result:
            for strategy, hits in result["strategies"].items():
                reordered, elapsed = rerank_hits(
                    question, hits, reranker, args.batch_size
                )
                result["strategies"][strategy] = reordered
                total_seconds += elapsed

        elif "retrieved_chunks" in result:
            reordered, elapsed = rerank_hits(
                question,
                result["retrieved_chunks"],
                reranker,
                args.batch_size,
            )
            result["retrieved_chunks"] = reordered
            total_seconds += elapsed
        else:
            raise ValueError("Estructura de retrieval desconocida.")

        result["reranker_model"] = args.model
        result["reranking_seconds"] = total_seconds

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
