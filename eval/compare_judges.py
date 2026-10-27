################################################################################
# COMPARE_JUDGES.PY
#
# Compara dos CSV generados por run_judge_qa_local.py sobre la MISMA muestra
# (mismo --sample-questions y --seed) sin necesidad de etiquetas humanas.
#
# Muestra:
#   - % de acuerdo y kappa ponderado entre los dos jueces
#   - matriz de confusión (filas = juez A, columnas = juez B)
#   - distribución de scores de cada juez
#   - cobertura léxica de la respuesta de referencia dentro del chunk, por score
#     (señal objetiva y barata: un score 2 debería tener mucha más cobertura
#     que un score 0)
#   - un CSV con los casos de desacuerdo para revisarlos rápidamente
#
# Uso:
#   python compare_judges.py juez_A.csv juez_B.csv --names selene qwen3
################################################################################

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_tokens(text) -> set:
    """Palabras en minúscula, sin tildes; se descartan las cortas (salvo cifras)."""
    if not isinstance(text, str):
        return set()
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {t for t in re.findall(r"\w+", text) if len(t) >= 4 or t.isdigit()}


def gold_coverage(gold, chunk) -> float:
    """Fracción de palabras de la respuesta de referencia presentes en el chunk."""
    gold_tokens = normalize_tokens(gold)
    if not gold_tokens:
        return np.nan
    return len(gold_tokens & normalize_tokens(chunk)) / len(gold_tokens)


def weighted_kappa(a, b, n_labels=3) -> float:
    """Kappa ponderado cuadrático para etiquetas 0..n_labels-1."""
    observed = np.zeros((n_labels, n_labels))
    for x, y in zip(a, b):
        observed[x, y] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.array(
        [[(i - j) ** 2 for j in range(n_labels)] for i in range(n_labels)]
    ) / (n_labels - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return 1 - (weights * observed).sum() / denominator


def load_valid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["judge_error"] = df["judge_error"].fillna("").astype(str).str.strip()
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")
    df["strategy"] = df["strategy"].fillna("")
    n_total = len(df)
    df = df[(df["judge_error"] == "") & df["relevance_score"].isin([0, 1, 2])].copy()
    df["relevance_score"] = df["relevance_score"].astype(int)
    print(f"{path.name}: {len(df)} válidas de {n_total}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("judge_a")
    parser.add_argument("judge_b")
    parser.add_argument("--names", nargs=2, default=["A", "B"])
    parser.add_argument("--out", default="judge_disagreements.csv")
    parser.add_argument("--n-disagreements", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    name_a, name_b = args.names
    key = ["question_id", "strategy", "rank"]

    a = load_valid(Path(args.judge_a))
    b = load_valid(Path(args.judge_b))

    merged = a.merge(
        b[key + ["relevance_score"]],
        on=key,
        suffixes=(f"_{name_a}", f"_{name_b}"),
    )
    col_a, col_b = f"relevance_score_{name_a}", f"relevance_score_{name_b}"
    print(f"\nPares comparables (válidos en ambos): {len(merged)}")
    if merged.empty:
        raise SystemExit("No hay pares en común: ¿usaste la misma muestra y semilla?")

    # ------------------------------------------------------------------
    # Acuerdo
    # ------------------------------------------------------------------
    agreement = (merged[col_a] == merged[col_b]).mean() * 100
    kappa = weighted_kappa(merged[col_a].tolist(), merged[col_b].tolist())
    print(f"Acuerdo exacto: {agreement:.1f} %")
    print(f"Kappa ponderado: {kappa:.3f}")

    print(f"\nMatriz de confusión (filas = {name_a}, columnas = {name_b}):")
    print(pd.crosstab(merged[col_a], merged[col_b]))

    print("\nDistribución de scores (%):")
    dist = pd.DataFrame({
        name_a: merged[col_a].value_counts(normalize=True).sort_index() * 100,
        name_b: merged[col_b].value_counts(normalize=True).sort_index() * 100,
    }).fillna(0).round(1)
    print(dist)

    # ------------------------------------------------------------------
    # Cobertura léxica de la respuesta de referencia
    # ------------------------------------------------------------------
    merged["gold_coverage"] = [
        gold_coverage(g, c)
        for g, c in zip(merged["gold_answer"], merged["chunk_text"])
    ]
    print("\nCobertura media de la respuesta de referencia en el chunk, por score:")
    coverage = pd.DataFrame({
        name_a: merged.groupby(col_a)["gold_coverage"].mean(),
        name_b: merged.groupby(col_b)["gold_coverage"].mean(),
    }).round(3)
    print(coverage)
    print("(Un buen juez da cobertura claramente mayor al score 2 que al 0.)")

    # ------------------------------------------------------------------
    # Desacuerdos para revisión rápida
    # ------------------------------------------------------------------
    diff = (merged[col_a] - merged[col_b]).abs()
    disagreements = merged[diff > 0].copy()
    disagreements["diff"] = diff[diff > 0]
    # primero los desacuerdos más grandes (0 vs 2), luego al azar
    big = disagreements[disagreements["diff"] == 2]
    small = disagreements[disagreements["diff"] == 1]
    n = args.n_disagreements
    picked = pd.concat([
        big.sample(min(len(big), n // 2), random_state=args.seed),
        small.sample(min(len(small), n - min(len(big), n // 2)), random_state=args.seed),
    ])
    picked = picked.assign(chunk_text=picked["chunk_text"].astype(str).str.slice(0, 1500))
    picked[
        ["question", "gold_answer", "chunk_text", col_a, col_b, "gold_coverage", "strategy", "rank"]
    ].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nDesacuerdos totales: {len(disagreements)} "
          f"(de ellos 0 vs 2: {len(big)})")
    print(f"Muestra de {len(picked)} desacuerdos guardada en: {args.out}")


if __name__ == "__main__":
    main()
