"""Gráficas de retrieval para un único corte k (3, 5, 7 o 10).

Usa la pestaña `summary` generada por run_metrics.py.
La distribución de relevancias permanece en @10 porque el Excel actual
solo incluye %score2@10, %score1@10 y %score0@10.
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

GRAPHICS_DIR = Path(__file__).resolve().parent
BASE_DIR = GRAPHICS_DIR.parent

EVAL_CONFIG_FILE = BASE_DIR / "configs" / "eval_config.json"
METRICS_DIR = BASE_DIR / "output" / "metrics"
OUTPUT_DIR = BASE_DIR / "output" / "graphics"

# Cortes disponibles en el nuevo run_metrics.py
AVAILABLE_K = (3, 5, 7, 10)

# El script de métricas actual calcula los porcentajes solo en @10.
SCORE_COLUMNS = ["%score2@10", "%score1@10", "%score0@10"]
SCORE_LABELS = ["% score 2 @10", "% score 1 @10", "% score 0 @10"]


def metric_columns(k):
    """Columnas de métricas correspondientes al k seleccionado."""
    return [f"Mean_nDCG@{k}", f"HitRate@{k}", f"MRR@{k}"]


def metric_labels(k):
    return [f"Mean nDCG@{k}", f"Hit Rate@{k}", f"MRR@{k}"]


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera un heatmap doble (métricas @k y distribución de scores @10) "
            "y un dotplot de las métricas @k desde el Excel de run_metrics.py."
        )
    )
    parser.add_argument(
        "metrics_file",
        nargs="?",
        default=None,
        help="Archivo .xlsx situado en output/metrics/<output_folder>/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=EVAL_CONFIG_FILE.name,
        help=(
            "Nombre del archivo de configuración dentro de configs/ "
            f"(por defecto: {EVAL_CONFIG_FILE.name})."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=AVAILABLE_K,
        default=10,
        help="Corte de las métricas a representar: 3, 5, 7 o 10 (por defecto: 10).",
    )
    return parser.parse_args()


# ============================================================
# PREPARAR DATOS
# ============================================================

def prepare_data(df, k):
    """Ordena las estrategias por Mean_nDCG@k, de mayor a menor."""
    if df.empty:
        raise ValueError("El archivo de métricas no contiene resultados.")
    return df.sort_values(f"Mean_nDCG@{k}", ascending=False).copy()


def get_title_label(df, metrics_file):
    """Tipo documental único si está presente; si no, nombre del archivo."""
    if "document_type" in df.columns:
        values = df["document_type"].dropna().unique()
        if len(values) == 1:
            return str(values[0]).capitalize()
    return metrics_file.stem


# ============================================================
# HEATMAP
# ============================================================

def draw_heatmap(ax, matrix, column_labels, vmax, value_format,
                 colorbar_label, fig):
    """Dibuja el heatmap, valores numéricos y barra de escala independiente."""
    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_xlabel("")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            if pd.isna(value):
                label, text_color = "NA", "black"
            else:
                label = format(value, value_format)
                text_color = "black" if value < 0.65 * vmax else "white"
            ax.text(
                col, row, label,
                ha="center", va="center", color=text_color, fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def create_heatmap(data, title_label, output_file, k):
    """Izquierda: métricas @k. Derecha: porcentajes de scores @10."""
    matrix_metrics = data.set_index("strategy")[metric_columns(k)]
    matrix_scores = data.set_index("strategy")[SCORE_COLUMNS]

    height = max(6, len(matrix_metrics) * 0.42)
    fig, (ax_metrics, ax_scores) = plt.subplots(
        1, 2,
        figsize=(18, height),
        sharey=True,
        gridspec_kw={"wspace": 0.35},
    )
    draw_heatmap(
        ax=ax_metrics,
        matrix=matrix_metrics,
        column_labels=metric_labels(k),
        vmax=1,
        value_format=".3f",
        colorbar_label="Score",
        fig=fig,
    )
    ax_metrics.set_yticks(range(len(matrix_metrics)))
    ax_metrics.set_yticklabels(matrix_metrics.index)
    ax_metrics.set_ylabel("Strategy")
    ax_metrics.set_title(f"Retrieval metrics @{k}")

    draw_heatmap(
        ax=ax_scores,
        matrix=matrix_scores,
        column_labels=SCORE_LABELS,
        vmax=100,
        value_format=".1f",
        colorbar_label="% of retrieved chunks",
        fig=fig,
    )
    ax_scores.tick_params(axis="y", labelleft=False, left=False)
    ax_scores.set_title("Relevance score distribution @10")

    fig.suptitle(f"Retrieval performance – {title_label} (k={k})", fontsize=14)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DOTPLOT
# ============================================================

def create_dotplot(data, title_label, output_file, k):
    """Dotplot con las tres métricas del k elegido, ordenado por nDCG@k."""
    # Al invertir las filas, la mejor estrategia aparece arriba en el eje Y.
    plot_data = data.iloc[::-1].copy()
    height = max(6, len(plot_data) * 0.42)
    fig, ax = plt.subplots(figsize=(11, height))
    y_positions = list(range(len(plot_data)))
    markers = ["o", "s", "^"]

    for metric, label, marker in zip(metric_columns(k), metric_labels(k), markers):
        ax.scatter(
            plot_data[metric],
            y_positions,
            label=label,
            marker=marker,
            s=65,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["strategy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Strategy")
    ax.set_title(f"Retrieval metrics @{k}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle(f"Retrieval performance – {title_label} (k={k})")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    eval_config_file = BASE_DIR / "configs" / args.config
    if not eval_config_file.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo de configuración:\n{eval_config_file}"
        )

    with eval_config_file.open("r", encoding="utf-8") as file:
        eval_config = json.load(file)

    fold = eval_config["fold"]
    suffix = fold.removeprefix("all_")
    retrieval_file = eval_config.get("retrieval_file")
    if retrieval_file:
        suffix = Path(retrieval_file).stem.removeprefix("retrieval_")
    judge_name = eval_config.get("metrics_judge", "local")
    output_folder = eval_config[f"{judge_name}_judge"]["output_folder"]
    print(f"Configuración: {eval_config_file}")

    metrics_filename = (
        args.metrics_file or f"metrics_{judge_name}_judge_{suffix}.xlsx"
    )
    metrics_file = METRICS_DIR / output_folder / metrics_filename
    if not metrics_file.exists():
        raise FileNotFoundError(f"No existe el archivo:\n{metrics_file}")

    df = pd.read_excel(metrics_file, sheet_name="summary")
    required_columns = {"strategy", *metric_columns(args.k), *SCORE_COLUMNS}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas necesarias en el Excel: "
            + ", ".join(sorted(missing))
            + ". Ejecuta primero la nueva versión de run_metrics.py."
        )

    collection_output_dir = OUTPUT_DIR / output_folder
    collection_output_dir.mkdir(parents=True, exist_ok=True)
    title_label = get_title_label(df, metrics_file)

    suffix = metrics_file.stem
    if suffix.startswith("metrics_"):
        suffix = suffix[len("metrics_"):]

    data = prepare_data(df, args.k)
    heatmap_file = collection_output_dir / f"heatmap_{suffix}_k{args.k}.png"
    dotplot_file = collection_output_dir / f"dotplot_{suffix}_k{args.k}.png"

    print(f"\nGenerando gráficas para: {metrics_file.name} (k={args.k})")
    create_heatmap(data, title_label, heatmap_file, args.k)
    create_dotplot(data, title_label, dotplot_file, args.k)
    print(f"  Heatmap: {heatmap_file}")
    print(f"  Dotplot: {dotplot_file}")
    print("\nGráficas generadas correctamente.")


if __name__ == "__main__":
    main()
