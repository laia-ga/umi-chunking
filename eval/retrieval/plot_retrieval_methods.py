"""Compara retrieval Dense, BM25, RRF y Weighted RRF para JSON Chunking.

Guardar en eval/plot_retrieval_methods.py (o scripts/) y ejecutar desde la raíz:
    python eval/plot_retrieval_methods.py
    python eval/plot_retrieval_methods.py --input-dir output/metrics
    python eval/plot_retrieval_methods.py --strategy JSONChunking

Lee la hoja 'summary' de cinco Excel generados por run_metrics.py.
Las métricas Mean_nDCG, HitRate y MRR están disponibles en @3, @5, @7, @10.
Los porcentajes de score 0/1/2 del Excel original SOLO corresponden a @10.

Salida (PNG y CSV) en output/graphics/aggregation/retrieval_methods/.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Servidores y Slurm sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
INPUT_DIR = BASE_DIR / "output" / "metrics"
OUTPUT_DIR = BASE_DIR / "output" / "graphics" / "retrieval_methods"

# Orden de visualización fijo. Solo cambiar rutas si cambia el nombre de los Excel.
FILES = {
    "Dense": (
        "qwen_uro_JSON_dense/"
        "metrics_local_judge_qwen3_embedding_4b_dense_top30_urology.xlsx"
    ),
    "BM25": (
        "qwen_uro_JSON_bm25/"
        "metrics_local_judge_qwen3_embedding_4b_bm25_top30_urology.xlsx"
    ),
    "RRF": (
        "qwen_uro_JSON_rrf/"
        "metrics_local_judge_qwen3_embedding_4b_rrf_n30_rrfk2_top30_urology.xlsx"
    ),
    "Weighted RRF 3:1": (
        "qwen_uro_JSON_wrrf_3_1/"
        "metrics_local_judge_qwen3_embedding_4b_weighted_rrf_n30_rrfk2_d3p0_b1p0_top30_urology.xlsx"
    ),
    "Weighted RRF 1:3": (
        "qwen_uro_JSON_wrrf_1_3/"
        "metrics_local_judge_qwen3_embedding_4b_weighted_rrf_n30_rrfk2_d1p0_b3p0_top30_urology.xlsx"
    ),
}

K_VALUES = (3, 5, 7, 10)
METRICS = {
    "Mean_nDCG": "Mean nDCG",
    "HitRate": "HitRate",
    "MRR": "MRR",
}
METHOD_COLORS = {
    "Dense": "#4477AA",
    "BM25": "#CC6677",
    "RRF": "#228833",
    "Weighted RRF 3:1": "#AA7744",
    "Weighted RRF 1:3": "#8855AA",
}
SCORE_COLORS = {0: "#AAB4C2", 1: "#E6B85C", 2: "#399B85"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--strategy", default="JSONChunking",
                   help="Estrategia de la hoja summary (por defecto: JSONChunking).")
    p.add_argument("--dpi", type=int, default=250)
    args = p.parse_args()
    if args.dpi < 50:
        p.error("--dpi debe ser al menos 50")
    return args


def select_strategy(table, path, strategy):
    """Evita promediar accidentalmente filas de distintas estrategias."""
    if table.empty:
        raise ValueError(f"Hoja summary vacía: {path}")
    if "strategy" not in table.columns:
        if len(table) == 1:
            warnings.warn(f"{path.name}: no existe 'strategy'; se utiliza su única fila")
            return table.iloc[0]
        raise ValueError(f"{path}: falta columna 'strategy' y hay {len(table)} filas")

    normalized = table["strategy"].astype(str).str.replace(r"[^a-zA-Z0-9]", "", regex=True).str.lower()
    expected = "".join(c for c in strategy.lower() if c.isalnum())
    matches = table.loc[normalized.eq(expected)]
    if len(matches) != 1:
        if len(table) == 1 and len(matches) == 0:
            warnings.warn(
                f"{path.name}: estrategia '{table.iloc[0]['strategy']}' distinta de '{strategy}'. "
                "Se utiliza la única fila; verifica que corresponde a JSON Chunking."
            )
            return table.iloc[0]
        raise ValueError(
            f"{path}: esperaba una fila para '{strategy}', encontradas {len(matches)}. "
            f"Estrategias disponibles: {table['strategy'].tolist()}"
        )
    return matches.iloc[0]


def numeric_cell(row, column, path, *, low=None, high=None):
    if column not in row.index:
        raise ValueError(f"Falta la columna '{column}' en {path}. "
                         "Comprueba que el Excel procede de run_metrics.py actualizado.")
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        raise ValueError(f"Valor inválido en {column} de {path}: {row[column]!r}")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{column} fuera del rango [{low}, {high}] en {path}: {value}")
    return value


def load_all(input_dir, strategy):
    summary_rows, metric_rows, percent_rows = [], [], []
    for method, relative_path in FILES.items():
        path = input_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra el Excel de {method}: {path}")
        table = pd.read_excel(path, sheet_name="summary")
        table.columns = table.columns.astype(str).str.strip()
        row = select_strategy(table, path, strategy)
        record = {"method": method, "strategy": str(row.get("strategy", strategy))}
        for metric in METRICS:
            for k in K_VALUES:
                field = f"{metric}@{k}"
                value = numeric_cell(row, field, path, low=0, high=1)
                record[field] = value
                count_key = f"n_questions_evaluated@{k}"
                count = pd.to_numeric(row.get(count_key, np.nan), errors="coerce")
                metric_rows.append({
                    "method": method,
                    "metric": metric,
                    "k": k,
                    "value": value,
                    "n_questions_evaluated": count,
                })
        count_key = "n_chunks"
        n_chunks = pd.to_numeric(row.get(count_key, np.nan), errors="coerce")
        record["n_chunks_valid_at_10"] = n_chunks
        for score in (0, 1, 2):
            key = f"%score{score}@10"
            value = numeric_cell(row, key, path, low=0, high=100)
            record[key] = value
            percent_rows.append({
                "method": method, "score": score, "percentage": value,
                "k": 10, "n_chunks_valid": n_chunks,
            })
        total_pct = sum(record[f"%score{s}@10"] for s in (0, 1, 2))
        if np.isfinite(n_chunks) and n_chunks <= 0:
            warnings.warn(f"{method}: n_chunks=0; los porcentajes @10 no son interpretables")
        elif abs(total_pct - 100) > 0.3:
            warnings.warn(f"{method}: los porcentajes @10 suman {total_pct:.2f}% en lugar de 100%")
        summary_rows.append(record)
        print(f"Leído: {method} — {path}")

    summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(metric_rows)
    percentages = pd.DataFrame(percent_rows)

    for k in K_VALUES:
        subset = metrics.loc[(metrics["k"] == k) & (metrics["metric"] == "Mean_nDCG")]
        counts = subset.set_index("method")["n_questions_evaluated"].dropna()
        if counts.nunique() > 1:
            warnings.warn(
                f"k={k}: distinto número de preguntas evaluadas por método: {counts.to_dict()}. "
                "Tenlo en cuenta al comparar medias."
            )
    return summary, metrics, percentages


def compact_ylim(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * 0.12, 0.025)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def plot_curves(metrics, output, dpi):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), sharex=True)
    markers = ("o", "s", "^", "D", "v")
    for ax, (metric, display) in zip(axs, METRICS.items()):
        part = metrics[metrics["metric"] == metric]
        for i, method in enumerate(FILES):
            subset = part[part["method"] == method].sort_values("k")
            ax.plot(subset["k"], subset["value"], label=method,
                    marker=markers[i], linewidth=2.4, markersize=6.5,
                    color=METHOD_COLORS[method])
        ax.set_title(display, fontsize=15, fontweight="bold", pad=12)
        ax.set_xticks(K_VALUES)
        ax.set_xlim(2.5, 10.5)
        ax.set_ylim(*compact_ylim(part["value"]))
        ax.set_xlabel("Número de chunks recuperados (k)")
        ax.set_ylabel("Puntuación media")
        ax.grid(alpha=0.24, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Retrieval Qwen3 + JSON Chunking | rendimiento según k", fontsize=18, fontweight="bold", y=0.97)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.035), fontsize=10.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.24, wspace=0.26)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {output}")


def plot_heatmaps(summary, output, dpi):
    methods = list(FILES)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    cmap = plt.get_cmap("YlGnBu")
    image = None
    for ax, (metric, display) in zip(axs, METRICS.items()):
        matrix = summary.set_index("method").reindex(methods)[[f"{metric}@{k}" for k in K_VALUES]].to_numpy(dtype=float)
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(display, fontsize=15, fontweight="bold", pad=13)
        ax.set_xticks(range(len(K_VALUES)), [f"k={k}" for k in K_VALUES])
        ax.set_yticks(range(len(methods)), methods, fontsize=9.5)
        for r in range(len(methods)):
            for c in range(len(K_VALUES)):
                val = matrix[r, c]
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        color="white" if val > 0.58 else "#172b3b", fontsize=9.4,
                        fontweight="medium")
        ax.set_xticks(np.arange(-.5, len(K_VALUES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.spines[:].set_visible(False)
    fig.suptitle("Comparación de retrieval por métrica y k", fontsize=17, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.11, top=0.85, wspace=0.43)
    color_ax = fig.add_axes([0.93, 0.18, 0.014, 0.57])
    fig.colorbar(image, cax=color_ax, label="Puntuación (0–1)")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {output}")


def plot_percentages(percentages, output, dpi):
    methods = list(FILES)
    data = percentages.pivot(index="method", columns="score", values="percentage").reindex(methods)
    counts = percentages.loc[percentages["score"] == 0].set_index("method")["n_chunks_valid"]

    fig, ax = plt.subplots(figsize=(12, 6.8))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for score, name in ((0, "0 · Irrelevante"), (1, "1 · Parcialmente relevante"), (2, "2 · Suficiente")):
        vals = data[score].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.60, color=SCORE_COLORS[score],
                label=name, edgecolor="white", linewidth=1.2)
        for row, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v/2, row, f"{v:.1f}%", ha="center", va="center", fontsize=10,
                        fontweight="bold", color="#1F2937")
        left += vals

    ax.set_yticks(y, methods, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 107)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    for i, method in enumerate(methods):
        n = counts.get(method, np.nan)
        if pd.notna(n):
            ax.text(101, i, f"n={int(n)}", ha="left", va="center", fontsize=9, color="#51606c")
    ax.set_xlabel("Porcentaje de chunks evaluados entre los 10 primeros de cada pregunta")
    ax.set_title("Distribución de relevancia AtlaAI | top-10", fontsize=17, fontweight="bold", pad=18)
    ax.grid(axis="x", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.legend(title="Puntuación del juez", ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), fontsize=10)
    fig.text(0.5, 0.015, "Distribución sobre chunks con juicio válido; los errores del juez quedan excluidos.",
             ha="center", fontsize=9, color="#596474")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.83, bottom=0.24)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {output}")


def main():
    args = parse_args()
    summary, metrics, percentages = load_all(args.input_dir, args.strategy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "retrieval_methods_summary.csv", index=False)
    metrics.to_csv(args.output_dir / "retrieval_methods_metrics_by_k.csv", index=False)
    percentages.to_csv(args.output_dir / "retrieval_methods_score_percentages_at_10.csv", index=False)
    plot_curves(metrics, args.output_dir / "retrieval_metrics_curves_4k.png", args.dpi)
    plot_heatmaps(summary, args.output_dir / "retrieval_metrics_heatmaps_4k.png", args.dpi)
    plot_percentages(percentages, args.output_dir / "retrieval_score_distribution_at_10.png", args.dpi)
    print("\nComparación completada. Todos los resultados están en:", args.output_dir)


if __name__ == "__main__":
    main()
