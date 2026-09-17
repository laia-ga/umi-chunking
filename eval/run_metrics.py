import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# MÉTRICAS A EVALUAR
# ============================================================

# Mean nDCG@5: mide la calidad global del ranking de los 5
# chunks recuperados, teniendo en cuenta tanto el grado de 
# relevancia como la posición

# HitRate@5: porcentaje de preguntas en el que se consigue 
# recuperar al menos un chunk con información suficiente
# (score = 2) entre los 5 chunks recuperados

# MRR: media de los inversos de las posiciones de los chunks 
# relevantes (score = 2) entre los chunks recuperados

# %score2: porcentaje de chunks relevantes recuperados (score = 2)

# %score1: porcentaje de chunks útiles pero insuficientes recuperados (score = 1)

# %score0: porcentaje de chunks irrelevantes recuperados (score = 0)


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
            "Calcula métricas de retrieval por combinación "
            "de strategy y document_type a partir del CSV "
            "generado por run_judge.py."
        )
    )

    parser.add_argument(
        "judge_file",
        help="Archivo CSV generado por run_judge.py",
    )

    return parser.parse_args()


# ============================================================
# DCG
# ============================================================

def dcg(scores):

    """
    Discounted Cumulative Gain.

    Relevancia graduada:

        0 = irrelevante
        1 = útil pero insuficiente
        2 = suficiente para responder

    Los chunks más relevantes reciben mayor ganancia
    y los chunks situados en posiciones posteriores
    reciben una penalización.
    """

    value = 0.0

    for i, relevance in enumerate(scores):

        gain = (2 ** relevance) - 1
        discount = np.log2(i + 2)

        value += gain / discount

    return value


# ============================================================
# nDCG@5
# ============================================================

def ndcg_at_5(scores):

    """
    nDCG@5.

    Mide la calidad global del ranking de los 5 chunks
    recuperados, teniendo en cuenta tanto el grado de
    relevancia como la posición.

    El ranking obtenido se compara con el mejor orden
    posible de esos mismos chunks.
    """

    scores = list(scores)[:5]

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
# HIT RATE@5
# ============================================================

def hit_at_5(scores):

    """
    HitRate@5.

    Devuelve 1 si entre los 5 chunks recuperados existe
    al menos un chunk con información suficiente para
    responder la pregunta (score = 2).

    En caso contrario devuelve 0.
    """

    scores = list(scores)[:5]

    if any(score == 2 for score in scores):
        return 1

    return 0


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(scores):

    """
    Reciprocal Rank.

    Busca la posición del primer chunk con información
    suficiente para responder la pregunta (score = 2).

    Ejemplos:

        [2, 0, 0, 0, 0] -> 1
        [0, 2, 0, 0, 0] -> 1/2
        [0, 0, 2, 0, 0] -> 1/3
        [0, 0, 0, 0, 2] -> 1/5
        [0, 0, 0, 0, 0] -> 0

    MRR será posteriormente la media del Reciprocal Rank
    de todas las preguntas.
    """

    scores = list(scores)[:5]

    for rank, score in enumerate(
        scores,
        start=1,
    ):

        if score == 2:
            return 1 / rank

    return 0.0


# ============================================================
# MÉTRICAS POR PREGUNTA
# ============================================================

def calculate_question_metrics(
    question_df,
):

    """
    Calcula nDCG@5, Hit@5 y RR para una pregunta.

    Los chunks se ordenan primero por su rank original.
    """

    question_df = question_df.sort_values(
        "rank"
    )

    scores = (
        question_df["relevance_score"]
        .astype(int)
        .tolist()
    )

    # Solo utilizar los primeros 5 chunks
    scores = scores[:5]

    return {
        "question_id":
            question_df.iloc[0]["question_id"],

        "nDCG@5":
            ndcg_at_5(scores),

        "Hit@5":
            hit_at_5(scores),

        "RR":
            reciprocal_rank(scores),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    judge_file = (
        JUDGE_DIR
        / args.judge_file
    )

    # --------------------------------------------------------
    # Comprobar que existe el archivo
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Comprobar columnas necesarias
    # --------------------------------------------------------

    required_columns = {
        "question_id",
        "question",
        "strategy",
        "document_type",
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
            + ", ".join(sorted(missing))
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
    # Comprobar scores
    # --------------------------------------------------------

    valid_scores = {
        0,
        1,
        2,
    }

    unexpected_scores = (
        set(df_valid["relevance_score"].unique())
        - valid_scores
    )

    if unexpected_scores:

        raise ValueError(
            "Se han encontrado relevance_score no válidos: "
            f"{sorted(unexpected_scores)}"
        )


    # ========================================================
    # MÉTRICAS POR STRATEGY × DOCUMENT_TYPE
    # ========================================================

    summary_results = []

    grouped = df_valid.groupby(
        [
            "strategy",
            "document_type",
        ],
        dropna=False,
        sort=False,
    )

    for (
        strategy,
        document_type,
    ), group_df in grouped:


        # ----------------------------------------------------
        # Métricas individuales de cada pregunta
        # ----------------------------------------------------

        question_results = []

        for question_id, question_df in group_df.groupby(
            "question_id",
            sort=False,
        ):

            metrics = calculate_question_metrics(
                question_df
            )

            question_results.append(
                metrics
            )


        per_question_df = pd.DataFrame(
            question_results
        )


        # ----------------------------------------------------
        # Chunks considerados para los porcentajes
        # ----------------------------------------------------

        top5_chunks = (
            group_df
            .sort_values(
                [
                    "question_id",
                    "rank",
                ]
            )
            .groupby(
                "question_id",
                sort=False,
            )
            .head(5)
        )


        # ----------------------------------------------------
        # Número total de chunks
        # ----------------------------------------------------

        n_chunks = len(
            top5_chunks
        )


        # ----------------------------------------------------
        # Porcentaje de cada score
        # ----------------------------------------------------

        if n_chunks > 0:

            pct_score2 = (
                (
                    top5_chunks["relevance_score"]
                    == 2
                ).mean()
                * 100
            )

            pct_score1 = (
                (
                    top5_chunks["relevance_score"]
                    == 1
                ).mean()
                * 100
            )

            pct_score0 = (
                (
                    top5_chunks["relevance_score"]
                    == 0
                ).mean()
                * 100
            )

        else:

            pct_score2 = 0.0
            pct_score1 = 0.0
            pct_score0 = 0.0


        # ----------------------------------------------------
        # Resumen de la combinación
        # ----------------------------------------------------

        summary = {
            "strategy":
                strategy,

            "document_type":
                document_type,

            "n_questions":
                len(per_question_df),

            "n_chunks":
                n_chunks,

            "Mean_nDCG@5":
                per_question_df[
                    "nDCG@5"
                ].mean(),

            "HitRate@5":
                per_question_df[
                    "Hit@5"
                ].mean(),

            "MRR":
                per_question_df[
                    "RR"
                ].mean(),

            "%score2":
                pct_score2,

            "%score1":
                pct_score1,

            "%score0":
                pct_score0,
        }

        summary_results.append(
            summary
        )


    # ========================================================
    # CREAR DATAFRAME FINAL
    # ========================================================

    summary_df = pd.DataFrame(
        summary_results
    )


    # --------------------------------------------------------
    # Ordenar resultados
    # --------------------------------------------------------

    summary_df = summary_df.sort_values(
        [
            "strategy",
            "document_type",
        ]
    )


    # ========================================================
    # ARCHIVO DE SALIDA
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = judge_file.stem

    if suffix.startswith("judge_"):

        suffix = suffix[
            len("judge_"):
        ]

    output_file = (
        OUTPUT_DIR
        / f"metrics_{suffix}.xlsx"
    )


    # ========================================================
    # GUARDAR EXCEL
    # ========================================================

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )


    # ========================================================
    # RESULTADO
    # ========================================================

    print("=" * 70)
    print("MÉTRICAS CALCULADAS")
    print("=" * 70)

    print(
        f"\nCombinaciones strategy × document_type: "
        f"{len(summary_df)}"
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nResultados guardados en:\n"
        f"{output_file}"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()