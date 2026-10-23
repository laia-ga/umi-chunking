
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MÉTRICAS A EVALUAR
# ============================================================

# nDCG, HitRate y MRR se calculan en los cortes 5 y 10.
# El ideal de nDCG ordena las relevancias juzgadas disponibles
# para cada pregunta; HitRate y MRR consideran suficiente solo score 2.
# La distribución de scores se calcula sobre los diez primeros chunks.


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
# nDCG
# ============================================================

def ndcg_at_k(scores, k):

    """
    Calcula nDCG@k usando como ideal las relevancias juzgadas
    disponibles para esta pregunta.

    Si no hay ningún chunk relevante, nDCG es 0.
    """

    scores = list(scores)
    ranked_scores = scores[:k]
    ideal_scores = sorted(scores, reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0
    return dcg(ranked_scores) / ideal_dcg


# ============================================================
# HIT RATE
# ============================================================

def hit_at_k(scores, k):

    """
    Devuelve 1 si existe al menos un chunk con score 2
    entre los primeros k; en caso contrario, 0.
    """

    return int(any(score == 2 for score in list(scores)[:k]))


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank_at_k(scores, k):

    """
    Inverso de la posición del primer chunk con score 2.
    Devuelve 0 si no aparece entre los primeros k.
    """

    for rank, score in enumerate(list(scores)[:k], start=1):

        if score == 2:
            return 1 / rank

    return 0.0


# ============================================================
# MÉTRICAS POR PREGUNTA
# ============================================================

def calculate_question_metrics(question_df):

    """
    Calcula nDCG, HitRate y MRR en @5 y @10 para una pregunta.

    Los chunks se ordenan según su rank original.
    """

    question_df = question_df.sort_values("rank")

    scores = (
        question_df["relevance_score"]
        .astype(int)
        .tolist()
    )

    metrics = {
        "question_id": question_df.iloc[0]["question_id"],
    }
    for cutoff in (5, 10):
        metrics[f"nDCG@{cutoff}"] = ndcg_at_k(scores, cutoff)
        metrics[f"Hit@{cutoff}"] = hit_at_k(scores, cutoff)
        metrics[f"MRR@{cutoff}"] = reciprocal_rank_at_k(scores, cutoff)
    return metrics


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
    # 5. Validar evaluaciones del judge
    # --------------------------------------------------------

    df_valid = df.copy()
    if "judge_error" in df_valid.columns:
        judge_errors = (
            df_valid["judge_error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        if judge_errors.any():
            examples = df_valid.loc[
                judge_errors,
                ["question_id", "rank", "judge_error"],
            ].head(5)
            raise RuntimeError(
                f"El CSV contiene {int(judge_errors.sum())} errores del judge. "
                "Repite esas evaluaciones antes de calcular métricas.\n"
                f"Ejemplos:\n{examples.to_string(index=False)}"
            )

    numeric_scores = pd.to_numeric(
        df_valid["relevance_score"],
        errors="coerce",
    )
    if numeric_scores.isna().any():
        bad_rows = df_valid.loc[
            numeric_scores.isna(),
            ["question_id", "rank", "relevance_score"],
        ].head(5)
        raise RuntimeError(
            "El CSV contiene puntuaciones vacías o no numéricas. "
            "Repite esas evaluaciones antes de calcular métricas.\n"
            f"Ejemplos:\n{bad_rows.to_string(index=False)}"
        )
    if not np.equal(numeric_scores, np.floor(numeric_scores)).all():
        raise ValueError("Las puntuaciones de relevancia deben ser enteros.")
    df_valid["relevance_score"] = numeric_scores.astype(int)

    numeric_ranks = pd.to_numeric(df_valid["rank"], errors="coerce")
    if (
        numeric_ranks.isna().any()
        or (numeric_ranks < 1).any()
        or not np.equal(numeric_ranks, np.floor(numeric_ranks)).all()
    ):
        raise ValueError("Los rangos deben ser enteros positivos.")
    df_valid["rank"] = numeric_ranks.astype(int)

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
                "nDCG@10",
                "Hit@10",
                "MRR@10",
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
                    "nDCG@10": 0.0,
                    "Hit@10": 0,
                    "MRR@10": 0.0,
                }
                for question_id in missing_q
            ])

            per_question_df = pd.concat(
                [per_question_df, empty],
                ignore_index=True,
            )

        # ----------------------------------------------------
        # Primeros diez chunks por pregunta
        # ----------------------------------------------------

        top10_chunks = (
            group_df
            .sort_values(["question_id", "rank"])
            .groupby("question_id", sort=False)
            .head(10)
        )

        n_chunks = len(top10_chunks)

        # ----------------------------------------------------
        # Distribución de puntuaciones en @10
        # ----------------------------------------------------

        if n_chunks > 0:

            scores = top10_chunks["relevance_score"]

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
            "Mean_nDCG@10": per_question_df["nDCG@10"].mean(),
            "HitRate@10": per_question_df["Hit@10"].mean(),
            "MRR@10": per_question_df["MRR@10"].mean(),
            "%score2@10": pct_score2,
            "%score1@10": pct_score1,
            "%score0@10": pct_score0,
        })

    # ========================================================
    # 8. CREAR DATAFRAME FINAL
    # ========================================================

    summary_df = pd.DataFrame(summary_results)

    summary_df = summary_df.sort_values(
        "Mean_nDCG@10",
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
        "Evaluaciones fallidas del judge: 0 "
        "(las métricas se detienen si quedan errores pendientes)"
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
