import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent

JUDGE_DIR = BASE_DIR / "output" / "judge"
OUTPUT_DIR = BASE_DIR / "output" / "metrics"


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Calcula métricas de evaluación a partir "
            "del CSV generado por run_judge.py."
        )
    )

    parser.add_argument(
        "judge_file",
        help="Archivo CSV generado por run_judge.py",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número máximo de chunks evaluados. Default: 5",
    )

    return parser.parse_args()


# ============================================================
# DCG / nDCG
# ============================================================

def dcg(scores):

    """
    Discounted Cumulative Gain.

    Usa directamente las relevancias graduadas:
    0 = no relevante
    1 = parcialmente relevante
    2 = altamente relevante
    """

    score = 0.0

    for i, relevance in enumerate(scores):

        gain = (2 ** relevance) - 1
        discount = np.log2(i + 2)

        score += gain / discount

    return score


def ndcg_at_k(scores, k):

    """
    nDCG@k.

    Compara el ranking real con el ranking ideal
    de los mismos chunks juzgados.
    """

    scores = list(scores)[:k]

    if not scores:
        return 0.0

    actual_dcg = dcg(scores)

    ideal_scores = sorted(
        scores,
        reverse=True,
    )

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# AVERAGE PRECISION
# ============================================================

def average_precision_at_k(scores, k):

    """
    AP@k.

    Para AP convertimos la relevancia a binaria:

    score 0 -> no relevante
    score 1 -> relevante
    score 2 -> relevante
    """

    scores = list(scores)[:k]

    binary = [
        1 if score >= 1 else 0
        for score in scores
    ]

    total_relevant = sum(binary)

    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0

    for rank, relevance in enumerate(
        binary,
        start=1,
    ):

        if relevance == 1:

            relevant_found += 1

            precision_at_rank = (
                relevant_found / rank
            )

            precision_sum += precision_at_rank

    return precision_sum / total_relevant


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def precision_recall_f1_at_k(
    all_scores,
    k,
):

    """
    Calcula Precision@k, Recall@k y F1@k.

    Un chunk se considera relevante si:
    relevance_score >= 1.

    IMPORTANTE:
    El recall se calcula respecto a todos los chunks
    relevantes presentes en el conjunto juzgado.
    """

    all_scores = list(all_scores)

    binary_all = [
        1 if score >= 1 else 0
        for score in all_scores
    ]

    binary_k = binary_all[:k]

    relevant_retrieved = sum(binary_k)

    total_relevant = sum(binary_all)

    # Precision
    if len(binary_k) == 0:
        precision = 0.0
    else:
        precision = (
            relevant_retrieved
            / len(binary_k)
        )

    # Recall
    if total_relevant == 0:
        recall = 0.0
    else:
        recall = (
            relevant_retrieved
            / total_relevant
        )

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


# ============================================================
# MÉTRICAS POR PREGUNTA
# ============================================================

def calculate_query_metrics(
    query_df,
    k_values,
):

    # Asegurar orden correcto del ranking
    query_df = query_df.sort_values(
        "rank"
    )

    scores = (
        query_df["relevance_score"]
        .astype(int)
        .tolist()
    )

    result = {
        "query_id": query_df.iloc[0]["query_id"],
        "query": query_df.iloc[0]["query"],
        "n_chunks": len(query_df),
        "n_highly_relevant": sum(
            score == 2
            for score in scores
        ),
        "n_partially_relevant": sum(
            score == 1
            for score in scores
        ),
        "n_irrelevant": sum(
            score == 0
            for score in scores
        ),
    }

    for k in k_values:

        actual_k = min(
            k,
            len(scores),
        )

        result[f"nDCG@{k}"] = ndcg_at_k(
            scores,
            actual_k,
        )

        result[f"AP@{k}"] = (
            average_precision_at_k(
                scores,
                actual_k,
            )
        )

        precision, recall, f1 = (
            precision_recall_f1_at_k(
                scores,
                actual_k,
            )
        )

        result[f"Precision@{k}"] = precision
        result[f"Recall@{k}"] = recall
        result[f"F1@{k}"] = f1

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    judge_file = (
        JUDGE_DIR
        / args.judge_file
    )

    if not judge_file.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n"
            f"{judge_file}"
        )

    # --------------------------------------------------------
    # Cargar resultados del judge
    # --------------------------------------------------------

    df = pd.read_csv(
        judge_file
    )

    required_columns = {
        "query_id",
        "query",
        "rank",
        "relevance_score",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Faltan columnas necesarias: "
            + ", ".join(missing)
        )

    # Eliminar evaluaciones fallidas
    df_valid = df[
        df["relevance_score"].notna()
    ].copy()

    df_valid["relevance_score"] = (
        df_valid["relevance_score"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Valores de K
    # --------------------------------------------------------

    default_k = [1, 3, 5]

    k_values = [
        k
        for k in default_k
        if k <= args.k
    ]

    if args.k not in k_values:
        k_values.append(args.k)

    k_values = sorted(
        set(k_values)
    )

    # --------------------------------------------------------
    # Métricas por pregunta
    # --------------------------------------------------------

    query_results = []

    for query_id, query_df in df_valid.groupby(
        "query_id",
        sort=False,
    ):

        metrics = calculate_query_metrics(
            query_df,
            k_values,
        )

        query_results.append(
            metrics
        )

    per_query_df = pd.DataFrame(
        query_results
    )

    # --------------------------------------------------------
    # Resumen global
    # --------------------------------------------------------

    summary = {
        "n_queries": len(per_query_df),
    }

    metric_columns = [
        column
        for column in per_query_df.columns
        if (
            column.startswith("nDCG@")
            or column.startswith("AP@")
            or column.startswith("Precision@")
            or column.startswith("Recall@")
            or column.startswith("F1@")
        )
    ]

    for column in metric_columns:

        mean_value = (
            per_query_df[column].mean()
        )

        # AP promedio = MAP
        if column.startswith("AP@"):

            k = column.split("@")[1]

            summary[f"MAP@{k}"] = (
                mean_value
            )

        else:

            summary[
                f"Mean_{column}"
            ] = mean_value

    summary_df = pd.DataFrame(
        [summary]
    )

    # --------------------------------------------------------
    # Nombre del Excel
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = judge_file.stem

    if suffix.startswith("judge_"):
        suffix = suffix[len("judge_"):]

    output_file = (
        OUTPUT_DIR
        / f"metrics_{suffix}.xlsx"
    )

    # --------------------------------------------------------
    # Guardar Excel
    # --------------------------------------------------------

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        per_query_df.to_excel(
            writer,
            sheet_name="per_query",
            index=False,
        )

        df_valid.to_excel(
            writer,
            sheet_name="chunks",
            index=False,
        )

    print("=" * 70)
    print("MÉTRICAS CALCULADAS")
    print("=" * 70)

    print(
        f"\nPreguntas evaluadas: "
        f"{len(per_query_df)}"
    )

    print(
        f"K evaluados: {k_values}"
    )

    print(
        f"\nResultados guardados en:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()