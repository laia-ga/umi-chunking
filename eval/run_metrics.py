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
            "Calcula métricas de retrieval a partir "
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

    Mantiene la relevancia graduada del LLM judge:

    0 = no relevante
    1 = parcialmente relevante
    2 = altamente relevante
    """

    value = 0.0

    for i, relevance in enumerate(scores):

        gain = (2 ** relevance) - 1
        discount = np.log2(i + 2)

        value += gain / discount

    return value


def ndcg_at_k(scores, k):

    """
    nDCG@k.

    Compara el orden obtenido por el retrieval
    con el orden ideal de esos chunks.
    """

    scores = list(scores)[:k]

    if not scores:
        return 0.0

    actual_dcg = dcg(scores)

    ideal_scores = sorted(
        scores,
        reverse=True,
    )

    ideal_dcg = dcg(
        ideal_scores
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# PRECISION
# ============================================================

def precision_at_k(scores, k):

    """
    Precision@k.

    Consideramos relevante:
        score 1
        score 2

    Consideramos no relevante:
        score 0
    """

    scores = list(scores)[:k]

    if not scores:
        return 0.0

    relevant = sum(
        score >= 1
        for score in scores
    )

    return relevant / len(scores)


# ============================================================
# HIT RATE
# ============================================================

def hit_at_k(scores, k):

    """
    Hit@k.

    Devuelve 1 si existe al menos un chunk
    relevante entre los primeros k resultados.
    En caso contrario devuelve 0.
    """

    scores = list(scores)[:k]

    if any(score >= 1 for score in scores):
        return 1

    return 0


# ============================================================
# AVERAGE PRECISION
# ============================================================

def average_precision_at_k(scores, k):

    """
    AP@k.

    Para AP convertimos las relevancias a binarias:

        0 -> no relevante
        1 -> relevante
        2 -> relevante

    MAP@k será posteriormente la media del AP@k
    de todas las preguntas.
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

            precision_sum += (
                relevant_found / rank
            )

    return precision_sum / total_relevant


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(scores):

    """
    Reciprocal Rank.

    Busca el primer chunk relevante.

    Ejemplos:

    [2, 0, 0, 0, 0] -> 1
    [0, 2, 0, 0, 0] -> 1/2
    [0, 0, 1, 0, 0] -> 1/3
    [0, 0, 0, 0, 0] -> 0

    MRR será la media del RR de todas las preguntas.
    """

    for rank, score in enumerate(
        scores,
        start=1,
    ):

        if score >= 1:
            return 1 / rank

    return 0.0


# ============================================================
# MÉTRICAS POR PREGUNTA
# ============================================================

def calculate_query_metrics(
    query_df,
    k_values,
):

    # Ordenar los chunks según el ranking
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

        "RR": reciprocal_rank(
            scores
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

        result[f"Precision@{k}"] = precision_at_k(
            scores,
            actual_k,
        )

        result[f"AP@{k}"] = average_precision_at_k(
            scores,
            actual_k,
        )

        result[f"Hit@{k}"] = hit_at_k(
            scores,
            actual_k,
        )

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
    # Cargar CSV del judge
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

    # --------------------------------------------------------
    # Eliminar evaluaciones fallidas
    # --------------------------------------------------------

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
    # RESUMEN GLOBAL
    # --------------------------------------------------------

    summary = {
        "n_queries": len(per_query_df),

        # Mean Reciprocal Rank
        "MRR": per_query_df["RR"].mean(),
    }

    for k in k_values:

        summary[f"Mean_nDCG@{k}"] = (
            per_query_df[f"nDCG@{k}"].mean()
        )

        summary[f"Mean_Precision@{k}"] = (
            per_query_df[
                f"Precision@{k}"
            ].mean()
        )

        # Mean Average Precision
        summary[f"MAP@{k}"] = (
            per_query_df[f"AP@{k}"].mean()
        )

        # Media de Hit@k = Hit Rate@k
        summary[f"HitRate@{k}"] = (
            per_query_df[f"Hit@{k}"].mean()
        )

    summary_df = pd.DataFrame(
        [summary]
    )

    # --------------------------------------------------------
    # NOMBRE DEL EXCEL
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
    # GUARDAR EXCEL
    # --------------------------------------------------------

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        # Resumen global del método
        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        # Métricas individuales por pregunta
        per_query_df.to_excel(
            writer,
            sheet_name="per_query",
            index=False,
        )

        # Todos los chunks evaluados
        df_valid.to_excel(
            writer,
            sheet_name="chunks",
            index=False,
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

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

    print("\nResumen:")

    for key, value in summary.items():

        if key == "n_queries":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

    print(
        f"\nResultados guardados en:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()