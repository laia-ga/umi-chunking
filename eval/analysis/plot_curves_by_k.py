"""Curvas de rendimiento en función de k, separadas por embedding model.

Ejemplos (guardar este archivo en eval/ del repositorio):
    python eval/plot_curves_by_k.py
    python eval/plot_curves_by_k.py --metric HitRate
    python eval/plot_curves_by_k.py --metric MRR --top-n 7

Genera una figura con 5 paneles (uno por embeddings) y un sexto panel
con una leyenda común de las estrategias de chunking.
Cada línea es una estrategia de chunking; el eje X es k=3, 5, 7, 10.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Compatible con sesiones Slurm/headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent # if SCRIPT_DIR.name == "eval" else SCRIPT_DIR
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
METRIC_NAMES = {"nDCG": "Mean_nDCG", "HitRate": "HitRate", "MRR": "MRR"}
STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curvas Mean_nDCG, HitRate o MRR para k=3,5,7,10, por modelo de embeddings."
    )
    parser.add_argument("--metric", choices=METRIC_NAMES, default="nDCG",
                        help="Métrica representada (por defecto: nDCG).")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=0,
                        help="0: todas las estrategias (predeterminado). "
                             "N>0: solo las N mejores por media sobre k en cada modelo.")
    args = parser.parse_args()
    if args.top_n < 0:
        parser.error("--top-n debe ser 0 o positivo.")
    return args


def load_metrics(input_dir, prefix):
    required = [f"{prefix}@{k}" for k in K_VALUES]
    frames = []
    for model, relative_path in FILES.items():
        path = input_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra el Excel de {model}: {path}")
        df = pd.read_excel(path, sheet_name="summary")
        df.columns = df.columns.astype(str).str.strip()
        missing = [c for c in ("strategy", *required) if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {missing}. "
                             "Ejecuta primero la versión actualizada de run_metrics.py")
        df = df[["strategy", *required]].copy()
        if df["strategy"].isna().any():
            raise ValueError(f"Estrategias sin nombre en {path}")
        df["strategy"] = df["strategy"].astype(str).str.strip()
        if df["strategy"].eq("").any() or df["strategy"].duplicated().any():
            raise ValueError(f"Estrategias vacías o duplicadas en {path}")
        for column in required:
            original = df[column]
            df[column] = pd.to_numeric(original, errors="coerce")
            if (original.notna() & df[column].isna()).any():
                raise ValueError(f"Valores no numéricos en {column} ({path})")
            if (df[column].notna() & ~df[column].between(0, 1)).any():
                raise ValueError(f"Valores fuera de [0,1] en {column} ({path})")
        df["embedding_model"] = model
        frames.append(df)
        print(f"Leído: {model} ({len(df)} estrategias)")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def plot_curves(df, prefix, output_file, top_n=0):
    models = list(FILES)
    kcols = [f"{prefix}@{k}" for k in K_VALUES]
    strategies = sorted(df["strategy"].unique())
    if not strategies:
        raise ValueError("No se encontraron estrategias.")

    # Colores/estilos estables para una misma estrategia en todos los modelos.
    palette = sns.color_palette("husl", n_colors=len(strategies))
    style_for = {
        name: (palette[i], STYLES[i % len(STYLES)], MARKERS[i % len(MARKERS)])
        for i, name in enumerate(strategies)
    }

    # ========================================================
    # Preparar datos de los cuatro modelos
    # ========================================================

    plot_data = {}

    for model in models:

        part = df.loc[
            df["embedding_model"] == model
        ].copy()

        if top_n:
            part["_mean_k"] = part[kcols].mean(axis=1)
            part = part.nlargest(top_n, "_mean_k")

        plot_data[model] = part

    # ========================================================
    # Calcular escala vertical global
    # ========================================================

    # Reunir todos los valores que se van a representar
    all_values = np.concatenate([
        part[kcols].to_numpy(dtype=float).ravel()
        for part in plot_data.values()
    ])

    # Eliminar valores no válidos
    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        raise ValueError(
            "No hay valores válidos para representar."
        )

    # Mínimo y máximo de los cuatro modelos
    y_min = float(all_values.min())
    y_max = float(all_values.max())

    # Añadir un margen del 5 %
    padding = max((y_max - y_min) * 0.05, 0.005)

    y_limits = (
        max(0, y_min - padding),
        min(1, y_max + padding)
    )

    print(
        f"Escala vertical común: "
        f"{y_limits[0]:.4f} - {y_limits[1]:.4f}"
    )

    # ========================================================
    # Crear figura 2x2
    # ========================================================

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(20, 14),
        sharex=True,
        sharey=True
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.90,
        bottom=0.27,
        wspace=0.20,
        hspace=0.30
    )

    represented = set()
    for ax, model in zip(axes.flat, models):
        part = plot_data[model]

        for _, item in part.iterrows():
            strategy = item["strategy"]
            color, linestyle, marker = style_for[strategy]
            y = item[kcols].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            ax.plot(
                K_VALUES, y,
                label=strategy, color=color, linestyle=linestyle,
                marker=marker, linewidth=1.7, markersize=5.5,
                alpha=0.85, markeredgecolor="white", markeredgewidth=0.5,
            )
            represented.add(strategy)

        ax.set_title(model, fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(K_VALUES)
        ax.set_xlim(2.5, 10.5)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Top-k (número de chunks recuperados)")
        ax.set_ylabel(prefix)
        ax.grid(True, alpha=0.22)
        ax.spines[["right", "top"]].set_visible(False)

    # La sexta celda se dedica a la leyenda en lugar de duplicarla en cinco paneles.

    from matplotlib.lines import Line2D
    handles = []
    for strategy in strategies:
        if strategy not in represented:
            continue
        color, linestyle, marker = style_for[strategy]
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                              marker=marker, markersize=5, linewidth=1.7,
                              label=strategy))
    fig.legend(
        handles=handles,
        title="Chunking strategies",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        fontsize=9,
        title_fontsize=11,
        ncol=4,
        columnspacing=1.5,
        handlelength=2.5,
    )

    fig.suptitle(
        f"Retrieval performance vs. k  |  {prefix}\n"
        "Una línea por estrategia de chunking; paneles por modelo de embeddings",
        fontsize=18, fontweight="bold", y=0.98,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_file}")


def main():
    args = parse_args()
    prefix = METRIC_NAMES[args.metric]
    df = load_metrics(args.input_dir, prefix)
    suffix = f"_top{args.top_n}" if args.top_n else ""
    output_file = args.output_dir / f"curves_k_{prefix}{suffix}.png"
    plot_curves(df, prefix, output_file, top_n=args.top_n)


if __name__ == "__main__":
    main()
