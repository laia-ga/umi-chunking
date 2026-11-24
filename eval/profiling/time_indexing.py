"""Comparación de tiempos de indexing por modelo de embeddings.

Lee:
    output/indexing/<modelo>/logs/run_log.jsonl

Y representa, para cada modelo, una barra apilada con:
    - normalize_seconds
    - embedding_seconds
    - upsert_seconds

normalizados por número de chunks.

Ejemplos:
    python eval/plot_indexing_stage_times_stacked.py
    python eval/plot_indexing_stage_times_stacked.py --per 10000
    python eval/plot_indexing_stage_times_stacked.py --scale log
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_ROOT = BASE_DIR / "output" / "indexing"
OUTPUT_DIR = BASE_DIR / "output" / "graphics" / "aggregation" / "indexing_times"

FILES = {
    "BGE-M3": "bge_uro/logs/run_log.jsonl",
    "Qwen3": "qwen_uro/logs/run_log.jsonl",
    "Multilingual-E5": "e5_uro/logs/run_log.jsonl",
    "MedCPT": "medcpt_uro/logs/run_log.jsonl",
    # "Jina": "jina_uro/logs/run_log.jsonl",
}

STAGES = [
    # ("normalize_seconds", "Normalización"),
    ("embedding_seconds", "Embeddings"),
    ("upsert_seconds", "Upsert"),
]

STAGE_COLORS = {
    # "Normalización": "#5B8FA8",
    "Embeddings": "#5B8FA8",
    "Upsert": "#8E6BBE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Representa tiempos de indexing por fase y modelo con barras apiladas."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Ruta raíz que contiene output/indexing/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--per",
        type=int,
        default=1000,
        help="Normalizar como segundos por N chunks (por defecto: 1000).",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "log"),
        default="linear",
        help="Escala del eje Y.",
    )
    return parser.parse_args()


def load_logs(input_root, per_n_chunks):
    records = []

    for model_label, relative_path in FILES.items():
        path = input_root / relative_path

        if not path.is_file():
            print(f"AVISO: no se encuentra {path}")
            continue

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"AVISO: JSON inválido en {path} (línea {line_number}): {exc}")
                    continue

                total_chunks = pd.to_numeric(data.get("total_chunks"), errors="coerce")
                if not np.isfinite(total_chunks) or total_chunks <= 0:
                    print(f"AVISO: total_chunks no válido en {path} (línea {line_number})")
                    continue

                row = {
                    "model": model_label,
                    "timestamp": data.get("timestamp"),
                    "model_name": data.get("model_name", model_label),
                    "collection_name": data.get("collection_name"),
                    "total_chunks": int(total_chunks),
                }

                for field, stage_label in STAGES:
                    raw_value = pd.to_numeric(data.get(field), errors="coerce")
                    if not np.isfinite(raw_value) or raw_value < 0:
                        raw_value = np.nan

                    row[field] = raw_value
                    row[f"{field}_per_{per_n_chunks}_chunks"] = (
                        raw_value / total_chunks * per_n_chunks
                        if np.isfinite(raw_value) else np.nan
                    )

                records.append(row)

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No se encontraron registros válidos en los run_log.jsonl.")

    return df


def summarize_runs(df, per_n_chunks):
    cols = [f"{field}_per_{per_n_chunks}_chunks" for field, _ in STAGES]

    summary = (
        df.groupby("model", as_index=False)[cols]
        .mean()
    )

    for field, _ in STAGES:
        summary[field] = summary[f"{field}_per_{per_n_chunks}_chunks"]

    summary["total_per_chunks"] = sum(
        summary[f"{field}_per_{per_n_chunks}_chunks"] for field, _ in STAGES
    )

    return summary


def plot_stacked(summary, per_n_chunks, output_file, scale="linear"):
    model_order = [m for m in FILES if m in summary["model"].tolist()]
    plot_df = summary.set_index("model").loc[model_order].reset_index()

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))

    for field, stage_label in STAGES:
        values = plot_df[f"{field}_per_{per_n_chunks}_chunks"].to_numpy(dtype=float)

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=STAGE_COLORS[stage_label],
            edgecolor="white",
            linewidth=0.8,
            label=stage_label,
        )

        # Etiqueta centrada dentro del segmento si cabe
        for xi, v, b in zip(x, values, bottom):
            if np.isfinite(v) and v > 0:
                if v >= 0.08 * np.nanmax(plot_df["total_per_chunks"].to_numpy(dtype=float)):
                    ax.text(
                        xi,
                        b + v / 2,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )

        bottom += np.nan_to_num(values, nan=0.0)

    # Etiqueta del total arriba de cada barra
    totals = plot_df["total_per_chunks"].to_numpy(dtype=float)
    for xi, total in zip(x, totals):
        if np.isfinite(total):
            ax.text(
                xi,
                total * 1.01 if scale == "linear" else total * 1.06,
                f"{total:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], fontsize=11)
    ax.set_xlabel("Modelo de embeddings", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        f"Segundos por {per_n_chunks:,} chunks".replace(",", "."),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Tiempo de indexing por modelo y etapa\n(normalizado por número de chunks)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    if scale == "log":
        ax.set_yscale("log")

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Etapa",
        frameon=False,
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
    )

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()

    if args.per <= 0:
        raise ValueError("--per debe ser un entero positivo.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(args.input_root, args.per)
    summary = summarize_runs(df, args.per)

    raw_csv = args.output_dir / f"indexing_stage_times_raw_per_{args.per}_chunks.csv"
    summary_csv = args.output_dir / f"indexing_stage_times_summary_per_{args.per}_chunks.csv"
    fig_file = args.output_dir / f"indexing_stage_times_stacked_per_{args.per}_chunks.png"

    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_stacked(summary, args.per, fig_file, scale=args.scale)

    print("\nArchivos generados:")
    print(f"  Gráfica: {fig_file}")
    print(f"  Datos crudos: {raw_csv}")
    print(f"  Resumen: {summary_csv}")

    print("\nResumen normalizado:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()