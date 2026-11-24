"""Gráficas agregadas de retrieval para k=3, 5, 7 y 10.

Por defecto crea una figura (líneas + heatmap) por modelo y valor de k.
También puede reproducir las figuras originales que comparan los modelos
para cada métrica y cada k.

Ejemplos (desde la raíz del repositorio):
    python eval/run_graphics_aggregation_k.py
    python eval/run_graphics_aggregation_k.py --mode both
    python eval/run_graphics_aggregation_k.py --mode combined --ks 3 7
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Permite generar las figuras dentro de Slurm, sin pantalla.
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Guardar el script en eval/
INPUT_DIR = BASE_DIR / "output" / "metrics"
OUTPUT_DIR = BASE_DIR / "output" / "graphics" / "aggregation"

FILES = {
    "BGE-M3": "bge_uro_atlaai/metrics_local_judge_bge_m3_urology.xlsx",
    "Qwen3": "qwen_uro_atlaai/metrics_local_judge_qwen3_embedding_4b_urology.xlsx",
    "Multilingual-E5": "e5_uro_atlaai/metrics_local_judge_multilingual_e5_large_urology.xlsx",
    "MedCPT": "medcpt_uro_atlaai/metrics_local_judge_medcpt_query_encoder_urology.xlsx",
    # "Jina": "jina_uro_atlaai/metrics_local_judge_jina_embeddings_v2_base_es_urology.xlsx",
}

K_VALUES = (3, 5, 7, 10)
METRIC_PREFIXES = ("Mean_nDCG", "HitRate", "MRR")
HEATMAP_CMAP = "Spectral"      # rojo = bajo, azul = alto
LINE_CMAP = "Spectral_r"      # una tonalidad diferente por estrategia
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera gráficas para k=3,5,7,10 por embedding model."
    )
    parser.add_argument(
        "--mode", choices=("models", "combined", "both"), default="models",
        help=(
            "models: 4 figuras por modelo (predeterminado); "
            "combined: gráficas originales comparando modelos para cada métrica y k; "
            "both: ambas modalidades."
        ),
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", choices=K_VALUES, default=list(K_VALUES),
        metavar="K",
        help="Valores de k a representar (por defecto: 3 5 7 10).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR,
        help="Carpeta que contiene los subdirectorios de métricas.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Carpeta de salida de las figuras.",
    )
    args = parser.parse_args()
    if any(k not in K_VALUES for k in args.ks):
        parser.error(f"Los valores de k admitidos son {K_VALUES}.")
    args.ks = tuple(dict.fromkeys(args.ks))
    return args


def metric_columns(k):
    return [f"{prefix}@{k}" for prefix in METRIC_PREFIXES]


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


# ============================================================
# 2. CARGA Y VALIDACIÓN
# ============================================================

def load_data(input_dir, ks):
    required_metrics = [name for k in ks for name in metric_columns(k)]
    dataframes = []

    for model, filename in FILES.items():
        filepath = input_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"No se encuentra el archivo: {filepath}")

        df = pd.read_excel(filepath, sheet_name="summary")
        df.columns = df.columns.astype(str).str.strip()
        missing = [col for col in ["strategy", *required_metrics] if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {filepath}: {missing}")

        df = df[["strategy", *required_metrics]].copy()
        if df["strategy"].isna().any():
            raise ValueError(f"Hay estrategias sin nombre en {filepath}.")
        df["strategy"] = df["strategy"].astype(str).str.strip()
        if df["strategy"].eq("").any():
            raise ValueError(f"Hay estrategias vacías en {filepath}.")

        for metric in required_metrics:
            original = df[metric]
            df[metric] = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & df[metric].isna()
            if invalid.any():
                raise ValueError(f"Valores no numéricos en {metric} ({filepath}).")
            outside = df[metric].notna() & ~df[metric].between(0, 1)
            if outside.any():
                raise ValueError(f"Valores de {metric} fuera de [0,1] ({filepath}).")

        df["embedding_model"] = model
        dataframes.append(df)

    all_data = pd.concat(dataframes, ignore_index=True)
    duplicates = all_data.duplicated(["embedding_model", "strategy"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Combinaciones duplicadas de modelo y estrategia:\n"
            + all_data.loc[duplicates, ["embedding_model", "strategy"]].to_string(index=False)
        )

    models = list(FILES)
    strategies = sorted(all_data["strategy"].unique())
    for model in models:
        present = set(all_data.loc[all_data["embedding_model"] == model, "strategy"])
        absent = sorted(set(strategies) - present)
        if absent:
            print(f"AVISO: {model} no contiene estas estrategias: {absent}")

    print(f"Modelos: {len(models)}; estrategias: {len(strategies)}; valores k: {ks}")
    return all_data, models, strategies


def build_colors(strategies):
    cmap = plt.get_cmap(LINE_CMAP)
    n = len(strategies)
    if n == 1:
        values = [0.1]
    else:
        values = np.concatenate([
            np.linspace(0.00, 0.40, n // 2),
            np.linspace(0.60, 1.00, n - n // 2),
        ])
    return {strategy: cmap(value) for strategy, value in zip(strategies, values)}


# ============================================================
# 3. ESTILO COMPARTIDO: LÍNEAS + HEATMAP
# ============================================================

def decorate_heatmap(ax, ordered_strategies, colors, fontsize=9):
    """Muestra indicadores coloreados y nombres de estrategias en el lateral."""
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    x_line_start, x_line_end, x_text = -0.58, -0.53, -0.50
    for i, strategy in enumerate(ordered_strategies):
        y = i + 0.5
        ax.plot(
            [x_line_start, x_line_end], [y, y],
            transform=ax.get_yaxis_transform(),
            color=colors[strategy], linewidth=4,
            solid_capstyle="round", clip_on=False, zorder=10,
        )
        ax.text(
            x_text, y, strategy, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=fontsize,
            color="black", clip_on=False,
        )


def format_axes(ax_lines, ax_heatmap, y_label, x_label):
    ax_lines.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax_lines.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax_lines.set_ylim(-0.03, 1.03)
    ax_lines.set_title("Performance trends", fontsize=14, fontweight="bold", pad=12)
    ax_lines.grid(axis="y", linestyle="--", alpha=0.25)
    ax_lines.set_axisbelow(True)
    ax_lines.spines["top"].set_visible(False)
    ax_lines.spines["right"].set_visible(False)
    ax_heatmap.set_title(
        "Performance by chunking strategy", fontsize=14, fontweight="bold", pad=12
    )
    ax_heatmap.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax_heatmap.set_ylabel("")
    ax_heatmap.tick_params(axis="x", rotation=0, labelsize=10)


def finish_figure(fig, output_file):
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.09, top=0.90, wspace=0.48)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_file}")


# ============================================================
# 4. CUATRO GRÁFICAS POR MODELO (20 EN TOTAL)
# ============================================================

def plot_one_model_one_k(df_all, model, k, colors, output_dir):
    """Una figura por modelo y k: líneas y heatmap de las 3 métricas @k."""
    cols = metric_columns(k)
    df = df_all.loc[df_all["embedding_model"] == model].copy()
    df = df.sort_values(f"Mean_nDCG@{k}", ascending=False, na_position="last")
    strategies = df["strategy"].tolist()
    matrix = df.set_index("strategy")[cols]
    labels = [f"Mean nDCG@{k}", f"HitRate@{k}", f"MRR@{k}"]

    fig_height = max(8, len(strategies) * 0.46 + 2.2)
    fig, (ax_lines, ax_heatmap) = plt.subplots(
        1, 2, figsize=(22, fig_height),
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.48},
    )
    fig.suptitle(f"{model} – retrieval metrics @ {k}", fontsize=19,
                 fontweight="bold", y=0.98)

    x = np.arange(len(cols))
    for i, strategy in enumerate(strategies):
        y = matrix.loc[strategy].to_numpy(dtype=float)
        ax_lines.plot(
            x, y, color=colors[strategy],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=5, linewidth=1.8, alpha=0.9,
            markeredgecolor="white", markeredgewidth=0.5,
        )
    ax_lines.set_xticks(x)
    ax_lines.set_xticklabels(labels, fontsize=11)
    format_axes(ax_lines, ax_heatmap, "Score", "Retrieval metric")

    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Score", "shrink": 0.75},
        annot_kws={"fontsize": 9}, ax=ax_heatmap,
    )
    ax_heatmap.set_xticklabels(labels, rotation=0)
    decorate_heatmap(ax_heatmap, strategies, colors)

    filename = f"retrieval_{slug(model)}_k{k}.png"
    finish_figure(fig, output_dir / "by_model" / slug(model) / filename)


# ============================================================
# 5. FIGURAS ORIGINALES: COMPARACIONES ENTRE MODELOS
# ============================================================

def plot_one_metric_one_k(df_all, models, metric, colors, output_dir):
    """Figura original para una métrica@k: líneas y heatmap entre modelos."""
    pivot = df_all.pivot(index="strategy", columns="embedding_model", values=metric)
    pivot = pivot.reindex(columns=models)
    ordered_strategies = (
        pivot.mean(axis=1).sort_values(ascending=False, na_position="last").index.tolist()
    )
    pivot = pivot.loc[ordered_strategies]

    fig_height = max(11, len(ordered_strategies) * 0.46 + 2.2)
    fig, (ax_lines, ax_heatmap) = plt.subplots(
        1, 2, figsize=(22, fig_height),
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.48},
    )
    fig.suptitle(metric, fontsize=19, fontweight="bold", y=0.98)

    x = np.arange(len(models))
    for i, strategy in enumerate(ordered_strategies):
        y = pivot.loc[strategy].to_numpy(dtype=float)
        ax_lines.plot(
            x, y, color=colors[strategy],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=5, linewidth=1.8, alpha=0.9,
            markeredgecolor="white", markeredgewidth=0.5,
        )
    ax_lines.set_xticks(x)
    ax_lines.set_xticklabels(models, fontsize=11)
    format_axes(ax_lines, ax_heatmap, metric, "Embedding model")

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": metric, "shrink": 0.75},
        annot_kws={"fontsize": 9}, ax=ax_heatmap,
    )
    decorate_heatmap(ax_heatmap, ordered_strategies, colors)
    filename = metric.replace("@", "_at_") + "_combined.png"
    finish_figure(fig, output_dir / "comparison" / filename)


def main():
    args = parse_args()
    sns.set_theme(style="white")
    df_all, models, strategies = load_data(args.input_dir, args.ks)
    colors = build_colors(strategies)

    count = 0
    if args.mode in ("models", "both"):
        for model in models:
            for k in args.ks:
                plot_one_model_one_k(df_all, model, k, colors, args.output_dir)
                count += 1

    if args.mode in ("combined", "both"):
        for k in args.ks:
            for metric in metric_columns(k):
                plot_one_metric_one_k(df_all, models, metric, colors, args.output_dir)
                count += 1

    print(f"\nGráficas generadas correctamente: {count}")
    print(f"Carpeta de salida: {args.output_dir}")


if __name__ == "__main__":
    main()
