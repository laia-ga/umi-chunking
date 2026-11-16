################################################################################
# COMPARE_JUDGE_SCALES.PY
#
# Compara dos jueces con escalas distintas (Selene 0-2 y Prometheus 1-5) sobre
# los MISMOS pares pregunta-estrategia-rango.
#
# Qué hace:
#   1. Une los dos CSV por (question_id, strategy, rank) y descarta errores.
#   2. Acuerdo a nivel de chunk: tabla cruzada, Spearman, kappa ponderado con
#      tres mapeos de Prometheus a 0/1/2 (dos fijos y uno que iguala la
#      distribución de Selene).
#   3. Métricas por estrategia (Hit@5, Hit@10, MRR@10) con varios criterios de
#      "relevante" (Selene >=2, >=1; Prometheus >=3, >=4, >=5 y los umbrales
#      que igualan a Selene) y comparación de los rankings (Spearman).
#   4. Bootstrap por preguntas: probabilidad de que cada estrategia sea la mejor
#      y su intervalo de confianza al 95 % (Hit@10).
#   5. Árbitro independiente: ¿la respuesta de referencia aparece escrita en el
#      chunk? (solo para respuestas de >= 4 caracteres; no detecta paráfrasis
#      ni traducciones, úsalo como indicio).
#   6. Exporta 2 CSV con pares donde los jueces discrepan: uno "ciego" (sin
#      notas) para que alguien lo etiquete y otro con la clave.
#
# Uso (desde la raíz del proyecto, con el .venv activado):
#   python eval/compare_judge_scales.py \
#       --selene output/judge/all_bge_m3_atlaai/local_judge_bge_m3_selene.csv \
#       --prometheus output/judge/all_bge_m3_prometheus/local_judge_bge_m3_prometheus.csv
################################################################################

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "judge" / "comparison_judges"
KEY = ["question_id", "strategy", "rank"]


# ------------------------------------------------------------------------------
# Carga
# ------------------------------------------------------------------------------
def load(path, low, high, name):
    df = pd.read_csv(path)
    total = len(df)
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")
    has_error = df["judge_error"].fillna("").astype(str).str.strip() != ""
    valid = (~has_error) & df["relevance_score"].between(low, high)
    print(f"{name}: {total} filas; {int(valid.sum())} válidas "
          f"({total - int(valid.sum())} descartadas por error o fuera de {low}-{high})")
    df = df[valid].copy()
    df["relevance_score"] = df["relevance_score"].astype(int)
    df["strategy"] = df["strategy"].fillna("")
    return df


# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------
def weighted_kappa(a, b, n_labels=3):
    observed = np.zeros((n_labels, n_labels))
    for x, y in zip(a, b):
        observed[x, y] += 1
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = np.array([[(i - j) ** 2 for j in range(n_labels)]
                        for i in range(n_labels)]) / (n_labels - 1) ** 2
    denominator = (weights * expected).sum()
    return float("nan") if denominator == 0 else 1 - (weights * observed).sum() / denominator


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def share_at_least(series, t):
    return float((series >= t).mean())


def match_threshold(prom, target_share, candidates=range(1, 6)):
    """Umbral t de Prometheus tal que share(prom >= t) se acerque a target_share."""
    return min(candidates, key=lambda t: abs(share_at_least(prom, t) - target_share))


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def min_rank_matrix(df, mask, questions, strategies):
    ranks = (df[mask].groupby(["question_id", "strategy"])["rank"].min()
             .unstack("strategy"))
    ranks = ranks.reindex(index=questions, columns=strategies)
    return ranks.fillna(np.inf).to_numpy()


def hits_with_nan(matrix, valid, k):
    """1/0 por (pregunta, estrategia); NaN donde la pareja se excluye."""
    return np.where(valid, (matrix <= k).astype(float), np.nan)


def metrics_from_matrix(matrix, valid):
    finite = np.isfinite(matrix)
    rr = np.where(finite, 1.0 / np.where(finite, matrix, 1.0), 0.0)
    rr = np.where(valid, rr, np.nan)
    return (
        np.nanmean(hits_with_nan(matrix, valid, 5), axis=0),   # Hit@5
        np.nanmean(hits_with_nan(matrix, valid, 10), axis=0),  # Hit@10
        np.nanmean(rr, axis=0),                                # MRR@10
    )


def bootstrap_best(hits, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_q, n_s = hits.shape
    wins = np.zeros(n_s)
    means = []
    for start in range(0, n_boot, 100):
        size = min(100, n_boot - start)
        idx = rng.integers(0, n_q, size=(size, n_q))
        sample_means = np.nanmean(hits[idx], axis=1)  # (size, n_s)
        wins += np.bincount(sample_means.argmax(axis=1), minlength=n_s)
        means.append(sample_means)
    means = np.vstack(means)
    return wins / n_boot, np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


# ------------------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selene", required=True)
    parser.add_argument("--prometheus", required=True)
    parser.add_argument("--n-disagreements", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    sel = load(args.selene, 0, 2, "Selene")
    pro = load(args.prometheus, 1, 5, "Prometheus")

    merged = sel.merge(
        pro[KEY + ["relevance_score"]], on=KEY, suffixes=("_sel", "_pro"))
    merged = merged.rename(columns={"relevance_score_sel": "sel", "relevance_score_pro": "pro"})

    # Una pareja (pregunta, estrategia) solo cuenta si sus filas están todas
    # validadas por ambos jueces (como hace run_metrics.py, por estrategia).
    strategies = sorted(merged.strategy.unique())
    questions = np.array(sorted(merged.question_id.unique()))
    counts = (merged.groupby(["question_id", "strategy"]).size()
              .unstack("strategy").reindex(index=questions, columns=strategies).fillna(0))
    expected = int(counts.to_numpy().max())
    valid = (counts.to_numpy() == expected)
    excluded = (~valid).sum(axis=0)
    print(f"\nPares válidos en ambos jueces: {len(merged)}; preguntas: {len(questions)}; "
          f"estrategias: {len(strategies)}; filas esperadas por pregunta y estrategia: {expected}")
    print(f"Parejas (pregunta, estrategia) excluidas por errores en algún juez: "
          f"{int(excluded.sum())} (máx. por estrategia: {int(excluded.max())})")

    # ---------------------------------------------------------------- acuerdo
    print("\n" + "=" * 78 + "\n1. ACUERDO A NIVEL DE CHUNK\n" + "=" * 78)
    print("\nFilas = nota de Selene; columnas = nota de Prometheus (% por fila):")
    table = pd.crosstab(merged["sel"], merged["pro"])
    print(table.to_string())
    print((pd.crosstab(merged["sel"], merged["pro"], normalize="index") * 100).round(1).to_string())
    print(f"\nSpearman Selene vs Prometheus (chunk a chunk): {spearman(merged['sel'], merged['pro']):.3f}")

    sel_share_2 = share_at_least(merged["sel"], 2)
    sel_share_1 = share_at_least(merged["sel"], 1)
    t2 = match_threshold(merged["pro"], sel_share_2)
    t1 = (match_threshold(merged["pro"], sel_share_1, range(1, t2))
          if t2 > 1 else 1)  # el umbral del 1 siempre por debajo del del 2
    mappings = {
        "A: 1-2->0, 3->1, 4-5->2": lambda p: np.select([p <= 2, p == 3], [0, 1], 2),
        "B: 1-3->0, 4->1, 5->2": lambda p: np.select([p <= 3, p == 4], [0, 1], 2),
        f"C: igualar a Selene (>={t1}->1, >={t2}->2)":
            lambda p: np.select([p < t1, p < t2], [0, 1], 2),
    }
    print(f"\nPrometheus, % por nota: "
          f"{(merged['pro'].value_counts(normalize=True).sort_index() * 100).round(1).to_dict()}")
    print(f"Selene, % por nota:     "
          f"{(merged['sel'].value_counts(normalize=True).sort_index() * 100).round(1).to_dict()}")
    print("\nKappa ponderado Selene vs Prometheus mapeado a 0/1/2:")
    for label, fn in mappings.items():
        mapped = fn(merged["pro"].to_numpy())
        agree = (mapped == merged["sel"].to_numpy()).mean() * 100
        print(f"  {label:45s} kappa={weighted_kappa(merged['sel'].tolist(), mapped.tolist()):.3f}"
              f"  acuerdo exacto={agree:.1f}%  (%2={np.mean(mapped == 2) * 100:.1f})")

    # ------------------------------------------------- métricas por estrategia
    print("\n" + "=" * 78 + "\n2. MÉTRICAS POR ESTRATEGIA CON DISTINTOS CRITERIOS\n" + "=" * 78)
    criteria = {
        "Selene>=2": merged["sel"] >= 2,
        "Selene>=1": merged["sel"] >= 1,
        "Prom>=5": merged["pro"] >= 5,
        "Prom>=4": merged["pro"] >= 4,
        "Prom>=3": merged["pro"] >= 3,
    }
    criteria[f"Prom>={t2}(=Selene2)"] = merged["pro"] >= t2
    criteria[f"Prom>={t1}(=Selene>=1)"] = merged["pro"] >= t1

    hit10 = {}
    summary = pd.DataFrame(index=strategies)
    for name, mask in criteria.items():
        matrix = min_rank_matrix(merged, mask, questions, strategies)
        h5, h10, mrr = metrics_from_matrix(matrix, valid)
        hit10[name] = (h10, matrix)
        summary[f"{name}|Hit@10"] = h10
        summary[f"{name}|MRR@10"] = mrr
        summary[f"{name}|Hit@5"] = h5

    hit_table = pd.DataFrame({name: values[0] for name, values in hit10.items()},
                             index=strategies)
    hit_table = hit_table.sort_values("Selene>=2", ascending=False)
    print("\nHit@10 por estrategia (ordenado por Selene>=2):")
    print(hit_table.round(3).to_string())

    # ---------------------------------------------------- árbitro independiente
    gold_norm = merged["gold_answer"].map(normalize_text)
    chunk_norm = merged["chunk_text"].map(normalize_text)
    usable = gold_norm.str.len() >= 4
    contains = pd.Series(
        [g in c if u else False for g, c, u in zip(gold_norm, chunk_norm, usable)],
        index=merged.index)
    usable_questions = np.array(sorted(merged.loc[usable, "question_id"].unique()))
    sub = merged[merged.question_id.isin(usable_questions)]
    proxy_matrix = min_rank_matrix(sub, contains.loc[sub.index], usable_questions, strategies)
    proxy_valid = valid[np.isin(questions, usable_questions)]
    proxy_h10 = metrics_from_matrix(proxy_matrix, proxy_valid)[1]
    hit_table["Respuesta_en_chunk"] = proxy_h10
    print(f"\nÁrbitro 'respuesta de referencia escrita en el chunk' "
          f"({len(usable_questions)} preguntas con respuesta >= 4 caracteres):")
    print(hit_table[["Selene>=2", "Prom>=4", "Respuesta_en_chunk"]].round(3).to_string())

    cells = {
        "Selene=2 y Prom<=2 (Selene dice suficiente, Prometheus no)":
            (merged["sel"] == 2) & (merged["pro"] <= 2),
        "Selene=0 y Prom>=4 (Prometheus dice suficiente, Selene no)":
            (merged["sel"] == 0) & (merged["pro"] >= 4),
        "Ambos altos (Selene=2 y Prom>=4)": (merged["sel"] == 2) & (merged["pro"] >= 4),
        "Ambos bajos (Selene=0 y Prom<=2)": (merged["sel"] == 0) & (merged["pro"] <= 2),
    }
    print("\nÁrbitro a nivel de chunk: % de chunks cuyo texto contiene la respuesta de referencia")
    for label, mask in cells.items():
        selected = mask & usable
        rate = contains[selected].mean() * 100 if selected.any() else float("nan")
        print(f"  {label:62s} n={int(selected.sum()):7d}  contiene la respuesta: {rate:5.1f}%")
    print(f"  (media general: {contains[usable].mean() * 100:.1f}%)")

    # ----------------------------------------------- comparación de rankings
    print("\n" + "=" * 78 + "\n3. ¿SE PARECEN LOS RANKINGS DE ESTRATEGIAS?\n" + "=" * 78)
    reference = hit_table["Selene>=2"]
    print(f"{'criterio':28s} {'Spearman vs Selene>=2':>22s} {'Spearman vs respuesta':>22s}  "
          f"{'mejor estrategia':42s} {'posición JSON':>13s}")
    for column in hit_table.columns:
        ranking = hit_table[column].sort_values(ascending=False)
        position = (list(ranking.index).index("JSONChunking") + 1
                    if "JSONChunking" in ranking.index else -1)
        print(f"{column:28s} {spearman(reference, hit_table[column]):22.3f} "
              f"{spearman(hit_table['Respuesta_en_chunk'], hit_table[column]):22.3f}  "
              f"{ranking.index[0]:42s} {position:13d}")

    # ------------------------------------------------------------- bootstrap
    print("\n" + "=" * 78 + "\n4. BOOTSTRAP (Hit@10): ¿QUIÉN ES LA MEJOR Y CON QUÉ SEGURIDAD?\n" + "=" * 78)
    for name in ["Selene>=2", "Prom>=4", f"Prom>={t2}(=Selene2)"]:
        hits = hits_with_nan(hit10[name][1], valid, 10).astype(np.float32)
        p_best, low, high = bootstrap_best(hits, args.n_boot, args.seed)
        result = pd.DataFrame({"Hit@10": np.nanmean(hits, axis=0), "IC95_inf": low, "IC95_sup": high,
                               "P(es la mejor)": p_best}, index=strategies)
        print(f"\nCriterio {name}:")
        print(result.sort_values("Hit@10", ascending=False).head(6).round(3).to_string())

    # -------------------------------------------------------- nota por estrategia
    means = merged.groupby("strategy").agg(sel_media=("sel", "mean"), pro_media=("pro", "mean"))
    print("\nNota media por estrategia:")
    print(means.sort_values("sel_media", ascending=False).round(3).to_string())

    # -------------------------------- guardar resultados en output/judge/comparison_judges
    summary.round(4).to_csv(OUT_DIR / "strategy_metrics_by_criterion.csv", encoding="utf-8-sig")
    hit_table.round(4).to_csv(OUT_DIR / "hit10_by_criterion.csv", encoding="utf-8-sig")

    merged["sel_norm"] = merged["sel"] / 2
    merged["pro_norm"] = (merged["pro"] - 1) / 4
    merged["diff"] = merged["sel_norm"] - merged["pro_norm"]
    half = args.n_disagreements // 2
    rng_state = args.seed
    selene_higher = merged[merged["diff"] >= 0.5]
    prom_higher = merged[merged["diff"] <= -0.5]
    picked = pd.concat([
        selene_higher.sample(min(half, len(selene_higher)), random_state=rng_state),
        prom_higher.sample(min(half, len(prom_higher)), random_state=rng_state),
    ]).sample(frac=1, random_state=rng_state).reset_index(drop=True)
    picked["pair_id"] = np.arange(1, len(picked) + 1)
    picked["chunk_text_trim"] = picked["chunk_text"].astype(str).str.slice(0, 1500)
    blind = picked[["pair_id", "question", "gold_answer", "chunk_text_trim"]].rename(
        columns={"chunk_text_trim": "chunk_text"})
    blind["etiqueta_0_1_2"] = ""
    blind.to_csv(OUT_DIR / "disagreements_blind.csv", index=False, encoding="utf-8-sig")
    picked[["pair_id", "question_id", "strategy", "rank", "sel", "pro"]].to_csv(
        OUT_DIR / "disagreements_key.csv", index=False, encoding="utf-8-sig")

    print(f"\nArchivos guardados en {OUT_DIR}:")
    print("  strategy_metrics_by_criterion.csv, hit10_by_criterion.csv")
    print(f"  disagreements_blind.csv ({len(blind)} pares para etiquetar a ciegas) y "
          f"disagreements_key.csv (clave)")


if __name__ == "__main__":
    main()