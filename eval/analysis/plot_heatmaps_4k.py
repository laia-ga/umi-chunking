#!/usr/bin/env python3
"""Comparación de embeddings × chunking × k mediante 4 heatmaps.

Instalar dependencias (en el entorno virtual adecuado):
    pip install pandas openpyxl numpy matplotlib seaborn

Ejemplo (guardar este archivo en eval/ del repositorio):
    python eval/plot_heatmaps_4k.py
    python eval/plot_heatmaps_4k.py --metric HitRate
    python eval/plot_heatmaps_4k.py --metric MRR

Cada panel representa k=3, 5, 7 o 10.
  Eje X: modelo de embeddings.
  Eje Y: estrategia de chunking (MISMO orden en los cuatro paneles).
  Color: valor de Mean_nDCG@k, HitRate@k o MRR@k (escala fija 0-1).
Los modelos se leen de output/metrics/<subcarpeta>/<archivo>.xlsx.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Ejecutable en servidor/Slurm sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Si se guarda en eval/, busca output/ en la raíz del repositorio.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent #if SCRIPT_DIR.name == "eval" else SCRIPT_DIR
INPUT_DIR = BASE_DIR / "output" / "metrics"
OUTPUT_DIR = BASE_DIR / "output" / "graphics" / "aggregation"

# Mantener las rutas EXACTAS de tus Excel de run_metrics.py.
FILES = {
    "BGE-M3": "bge_uro_atlaai/metrics_local_judge_bge_m3_urology.xlsx",
    "Qwen3": "qwen_uro_atlaai/metrics_local_judge_qwen3_embedding_4b_urology.xlsx",
    "Multilingual-E5": "e5_uro_atlaai/metrics_local_judge_multilingual_e5_large_urology.xlsx",
    "MedCPT": "medcpt_uro_atlaai/metrics_local_judge_medcpt_query_encoder_urology.xlsx",
    # "Jina": "jina_uro_atlaai/metrics_local_judge_jina_embeddings_v2_base_es_urology.xlsx",
}
K_VALUES = (3, 5, 7, 10)
METRIC_NAMES = {"nDCG": "Mean_nDCG", "HitRate": "HitRate", "MRR": "MRR"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="4 heatmaps (k=3,5,7,10) comparando modelos de embeddings y chunking."
    )
    parser.add_argument("--metric", choices=METRIC_NAMES, default="nDCG",
                        help="Métrica que colorea las celdas (por defecto: nDCG).")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                        help="Carpeta output/metrics con las subcarpetas por modelo.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Carpeta donde guardar la figura PNG.")
    parser.add_argument("--no-values", action="store_true",
                        help="Oculta los números dentro de las celdas.")
    return parser.parse_args()


def load_metrics(input_dir, metric_prefix):
    required = [f"{metric_prefix}@{k}" for k in K_VALUES]
    records = []

    for model, relative_path in FILES.items():
        path = input_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra el Excel de {model}: {path}")

        df = pd.read_excel(path, sheet_name="summary")
        df.columns = df.columns.astype(str).str.strip()
        missing = [c for c in ["strategy", *required] if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {missing}. "
                             "Ejecuta primero la versión actualizada de run_metrics.py")

        df = df[["strategy", *required]].copy()
        if df["strategy"].isna().any():
            raise ValueError(f"Hay nombres de estrategia vacíos en {path}")
        df["strategy"] = df["strategy"].astype(str).str.strip()
        if df["strategy"].eq("").any() or df["strategy"].duplicated().any():
            raise ValueError(f"Estrategias vacías o repetidas en {path}")

        for k in K_VALUES:
            column = f"{metric_prefix}@{k}"
            original = df[column]
            df[column] = pd.to_numeric(original, errors="coerce")
            bad = original.notna() & df[column].isna()
            if bad.any():
                raise ValueError(f"Valores no numéricos en {column} ({path})")
            if (df[column].notna() & ~df[column].between(0, 1)).any():
                raise ValueError(f"Valores fuera de 0-1 en {column} ({path})")
            for strategy, value in zip(df["strategy"], df[column]):
                records.append({
                    "model": model,
                    "strategy": strategy,
                    "k": k,
                    "value": value,
                })
        print(f"Leído: {model} ({len(df)} estrategias)")

    return pd.DataFrame(records)


def create_heatmaps(data, metric, output_file, show_values=True):
    models = list(FILES)
    # Una única ordenación para los cuatro paneles, por la media en todos los k/modelos.
    strategies = (
        data.groupby("strategy", sort=False)["value"]
        .mean()
        .sort_values(ascending=False, na_position="last")
        .index.tolist()
    )
    if not strategies:
        raise ValueError("No se encontraron estrategias para representar.")

    sns.set_theme(style="white", font_scale=0.95)
    fig_height = max(10, 5 + len(strategies) * 0.40)
    fig, axes = plt.subplots(2, 2, figsize=(20, fig_height))
    cmap = plt.get_cmap("Spectral").copy()  # Rojo = bajo, azul = alto
    cmap.set_bad("#ECEFF1")

    # Máximo global entre todos los modelos, estrategias y valores de k
    global_max = data["value"].max()

    if pd.isna(global_max) or global_max <= 0:
        raise ValueError("No hay valores positivos para definir la escala.")

    for ax, k in zip(axes.flat, K_VALUES):
        matrix = (
            data.loc[data["k"] == k]
            .pivot(index="strategy", columns="model", values="value")
            .reindex(index=strategies, columns=models)
        )
        sns.heatmap(
            matrix,
            cmap=cmap, vmin=0, vmax=global_max,
            linewidths=0.6, linecolor="white",
            annot=show_values, fmt=".3f", annot_kws={"fontsize": 7.5},
            cbar=False, ax=ax,
        )
        ax.set_facecolor("#ECEFF1")
        ax.set_title(f"k = {k}", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Embedding model", fontsize=10)
        ax.set_ylabel("Chunking strategy" if ax in axes[:, 0] else "", fontsize=10)
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        # En cada panel se ven los mismos nombres/orden de estrategias.
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(0)

        # Identifica de forma explícita las combinaciones sin datos.
        for i, j in np.argwhere(matrix.isna().to_numpy()):
            ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                    fontsize=10, color="#666666")

    fig.suptitle(
        f"Retrieval performance: {metric}  |  embeddings × chunking × k",
        fontsize=19, fontweight="bold", y=0.995,
    )
    # Misma escala y barra de color para los cuatro heatmaps.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=global_max), cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.15, right=0.89, top=0.94, bottom=0.06,
                        hspace=0.30, wspace=0.50)
    cax = fig.add_axes([0.915, 0.19, 0.015, 0.60])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label(f"{metric} (0–{global_max:.3f})", fontsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_file}")


def main():
    args = parse_args()
    prefix = METRIC_NAMES[args.metric]
    data = load_metrics(args.input_dir, prefix)
    output_file = args.output_dir / f"heatmaps_4k_{prefix}.png"
    create_heatmaps(data, args.metric, output_file, show_values=not args.no_values)


if __name__ == "__main__":
    main()
