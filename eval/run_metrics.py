
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MÉTRICAS A EVALUAR
# ============================================================

# Mean nDCG@5: mide la calidad de los 5 chunks recuperados
# frente a una situación ideal fija (5 chunks con score = 2),
# teniendo en cuenta tanto la relevancia como la posición.
#
# HitRate@5: proporción de preguntas para las que se recupera
# al menos un chunk con score = 2 entre los primeros cinco.
#
# MRR@5: media del inverso de la posición del primer chunk
# con score = 2 entre los cinco primeros.
#
# %score2, %score1, %score0: distribución de puntuaciones
# entre los chunks recuperados, considerando los primeros cinco.
#
# Las preguntas sin chunks recuperados cuentan como 0.
# Las evaluaciones fallidas del juez se tratan como score = 0.


# ============================================================
# RUTAS
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent

EVAL_CONFIG_FILE = BASE_DIR / "configs" / "eval_config.json"

JUDGE_DIR = BASE_DIR / "output" / "judge"
OUTPUT_DIR = BASE_DIR / "output" / "metrics"


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Calcula métricas de retrieval por estrategia "
            "a partir del CSV generado por run_judge.py."
        )
    )

    parser.add_argument(
        "judge_file",
        nargs="?",
        default=None,
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

    Ganancia = 2^relevancia - 1
    Descuento = log2(posición + 1)
    """

    value = 0.0

    for i, relevance in enumerate(scores):

        gain = (2 ** relevance) - 1
        discount = np.log2(i + 2)

        value += gain / discount

    return value


# ============================================================
# nDCG@5 (IDEAL FIJO)
# ============================================================

IDEAL_SCORES = [2, 2, 2, 2, 2]
IDEAL_DCG = dcg(IDEAL_SCORES)


def ndcg_at_5(scores):

    """
    nDCG@5 con ideal fijo.

    Solo el ranking [2, 2, 2, 2, 2] obtiene 1.0.
    Las posiciones ausentes equivalen a relevancia 0.
    """

    scores = list(scores)[:5]

    return dcg(scores) / IDEAL_DCG


# ============================================================
# HIT RATE@5
# ============================================================

def hit_at_5(scores):

    """
    Devuelve 1 si existe al menos un chunk con score 2
    entre los cinco primeros; en caso contrario, 0.
    """

    scores = list(scores)[:5]

    return int(any(score == 2 for score in scores))


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(scores):

    """
    Inverso de la posición del primer chunk con score 2.
    Devuelve 0 si no aparece entre los cinco primeros.
    """

    scores = list(scores)[:5]

    for rank, score in enumerate(scores, start=1):

        if score == 2:
            return 1 / rank

    return 0.0


# ============================================================
# MÉTRICAS POR PREGUNTA
# ============================================================

def calculate_question_metrics(question_df):

    """
    Calcula nDCG@5, Hit@5 y RR@5 para una pregunta.

    Los chunks se ordenan según su rank original.
    """

    question_df = question_df.sort_values("rank")

    scores = (
        question_df["relevance_score"]
        .astype(int)
        .tolist()
    )

    scores = scores[:5]

    return {
        "question_id": question_df.iloc[0]["question_id"],
        "nDCG@5": ndcg_at_5(scores),
        "Hit@5": hit_at_5(scores),
        "MRR@5": reciprocal_rank(scores),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # 1. Cargar configuración de evaluación
    # --------------------------------------------------------
    with EVAL_CONFIG_FILE.open("r", encoding="utf-8") as file:
        eval_config = json.load(file)

    fold = eval_config["fold"]
    suffix = fold.removeprefix("all_")
    judge_name = eval_config.get("metrics_judge", "local")
    output_folder = eval_config[f"{judge_name}_judge"]["output_folder"]

    # --------------------------------------------------------
    # 2. Localizar CSV del juez
    # --------------------------------------------------------

    judge_filename = args.judge_file or f"{judge_name}_judge_{suffix}.csv"
    judge_file = JUDGE_DIR / output_folder / judge_filename

    if not judge_file.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n{judge_file}"
        )

    # --------------------------------------------------------
    # 3. Cargar resultados del juez
    # --------------------------------------------------------

    df = pd.read_csv(judge_file)

    required_columns = {
        "question_id",
        "question",
        "strategy",
        "rank",
        "relevance_score",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            "Faltan columnas necesarias: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError(
            "El CSV del juez no contiene evaluaciones."
        )

    # --------------------------------------------------------
    # 4. Preguntas esperadas
    # --------------------------------------------------------

    # Todas las preguntas presentes en el CSV, sin distinguir
    # por tipo de documento ni por estrategia.

    expected_questions = df["question_id"].dropna().unique()

    # --------------------------------------------------------
    # 5. Evaluaciones fallidas -> score 0
    # --------------------------------------------------------

    df_valid = df.copy()

    df_valid["judge_failed"] = (
        df_valid["relevance_score"].isna()
    )

    n_failed_total = int(
        df_valid["judge_failed"].sum()
    )

    df_valid["relevance_score"] = (
        df_valid["relevance_score"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # 6. Comprobar puntuaciones
    # --------------------------------------------------------

    valid_scores = {0, 1, 2}

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
    # 7. MÉTRICAS POR STRATEGY
    # ========================================================

    summary_results = []

    grouped = df_valid.groupby(
        "strategy",
        dropna=False,
        sort=False,
    )

    for strategy, group_df in grouped:

        # ----------------------------------------------------
        # Métricas individuales por pregunta
        # ----------------------------------------------------

        question_results = []

        for question_id, question_df in group_df.groupby(
            "question_id",
            sort=False,
        ):

            question_results.append(
                calculate_question_metrics(question_df)
            )

        per_question_df = pd.DataFrame(
            question_results,
            columns=[
                "question_id",
                "nDCG@5",
                "Hit@5",
                "MRR@5",
            ],
        )

        # ----------------------------------------------------
        # Preguntas sin chunks recuperados -> métricas 0
        # ----------------------------------------------------

        retrieved = set(
            per_question_df["question_id"]
        )

        missing_q = [
            question_id
            for question_id in expected_questions
            if question_id not in retrieved
        ]

        if missing_q:

            empty = pd.DataFrame([
                {
                    "question_id": question_id,
                    "nDCG@5": 0.0,
                    "Hit@5": 0,
                    "MRR@5": 0.0,
                }
                for question_id in missing_q
            ])

            per_question_df = pd.concat(
                [per_question_df, empty],
                ignore_index=True,
            )

        # ----------------------------------------------------
        # Primeros cinco chunks por pregunta
        # ----------------------------------------------------

        top5_chunks = (
            group_df
            .sort_values(["question_id", "rank"])
            .groupby("question_id", sort=False)
            .head(5)
        )

        n_chunks = len(top5_chunks)

        n_failed = int(
            top5_chunks["judge_failed"].sum()
        )

        # ----------------------------------------------------
        # Distribución de puntuaciones
        # ----------------------------------------------------

        if n_chunks > 0:

            scores = top5_chunks["relevance_score"]

            pct_score2 = (scores == 2).mean() * 100
            pct_score1 = (scores == 1).mean() * 100
            pct_score0 = (scores == 0).mean() * 100

        else:

            pct_score2 = 0.0
            pct_score1 = 0.0
            pct_score0 = 0.0

        # ----------------------------------------------------
        # Resumen de la estrategia
        # ----------------------------------------------------

        summary_results.append({
            "strategy": strategy,
            "n_questions": len(per_question_df),
            "n_missing": len(missing_q),
            "n_chunks": n_chunks,
            "Mean_nDCG@5": per_question_df["nDCG@5"].mean(),
            "HitRate@5": per_question_df["Hit@5"].mean(),
            "MRR@5": per_question_df["MRR@5"].mean(),
            "%score2": pct_score2,
            "%score1": pct_score1,
            "%score0": pct_score0,
        })

    # ========================================================
    # 8. CREAR DATAFRAME FINAL
    # ========================================================

    summary_df = pd.DataFrame(summary_results)

    summary_df = summary_df.sort_values(
        "Mean_nDCG@5",
        ascending=False,
    )

    # ========================================================
    # 9. COMBINACIONES AUSENTES
    # ========================================================

    # Ya no hay combinaciones strategy x document_type.
    # Cada estrategia aparece una única vez en summary.

    # ========================================================
    # 10. ARCHIVO DE SALIDA
    # ========================================================

    collection_output_dir = OUTPUT_DIR / output_folder

    collection_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = judge_file.stem

    if suffix.startswith("judge_"):
        suffix = suffix[len("judge_"):]

    output_file = (
        collection_output_dir
        / f"metrics_{suffix}.xlsx"
    )

    # ========================================================
    # 11. GUARDAR EXCEL
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
    # 12. RESULTADO
    # ========================================================

    print("=" * 70)
    print("MÉTRICAS CALCULADAS")
    print("=" * 70)

    print(
        f"\nEstrategias evaluadas: {len(summary_df)}"
    )

    print(
        f"Preguntas esperadas: {len(expected_questions)}"
    )

    print(
        "Evaluaciones fallidas del judge "
        f"(tratadas como 0): {n_failed_total}"
    )

    print()
    print(summary_df.to_string(index=False))

    print(
        f"\nResultados guardados en:\n{output_file}"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
