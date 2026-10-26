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


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Genera un heatmap doble (métricas y distribución de "
            "scores) y un dotplot de las métricas "
            "de retrieval a partir de un archivo de métricas "
            "de un único tipo de documento."
        )
    )

    parser.add_argument(
        "metrics_file",
        nargs="?",
        default=None,
        help=(
            "Nombre del archivo .xlsx situado en "
            "output/metrics"
        ),
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

    return parser.parse_args()


# ============================================================
# PREPARAR DATOS
# ============================================================

def prepare_data(
    df,
):

    """
    Ordena las estrategias de mejor a peor según Mean_nDCG@10.
    """

    data = df.copy()

    if data.empty:

        raise ValueError(
            "El archivo de métricas no contiene resultados."
        )

    # --------------------------------------------------------
    # nDCG@10 para ordenar
    # --------------------------------------------------------

    data = data.sort_values(
        "Mean_nDCG@10",
        ascending=False,
    )

    return data


# ============================================================
# TÍTULO
# ============================================================

def get_title_label(
    df,
    metrics_file,
):

    """
    Devuelve la etiqueta que se usa en el título de las
    gráficas: el valor de la columna document_type si existe
    y es único, o el nombre del archivo en caso contrario.
    """

    if "document_type" in df.columns:

        values = df["document_type"].dropna().unique()

        if len(values) == 1:
            return str(values[0]).capitalize()

    return metrics_file.stem


# ============================================================
# HEATMAP
# ============================================================

# Métricas del heatmap izquierdo (escala 0-1)
METRICS = [
    "Mean_nDCG@5",
    "HitRate@5",
    "MRR@5",
    "Mean_nDCG@10",
    "HitRate@10",
    "MRR@10",
]

METRIC_LABELS = [
    "Mean nDCG@5",
    "Hit Rate@5",
    "MRR@5",
    "Mean nDCG@10",
    "Hit Rate@10",
    "MRR@10",
]

# Porcentajes del heatmap derecho (escala 0-100)
SCORE_COLUMNS = [
    "%score2@10",
    "%score1@10",
    "%score0@10",
]

SCORE_LABELS = [
    "% score 2 @10",
    "% score 1 @10",
    "% score 0 @10",
]


def draw_heatmap(
    ax,
    matrix,
    column_labels,
    vmax,
    value_format,
    colorbar_label,
    fig,
):

    """
    Dibuja un heatmap en el eje indicado, con el valor de
    cada celda escrito dentro y su propia barra de escala.

    La escala va de 0 a vmax (1 para las métricas y 100
    para los porcentajes).
    """

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------
    #
    # YlGnBu proporciona colores relativamente suaves
    # manteniendo una escala visual clara.
    # --------------------------------------------------------

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
    )

    # --------------------------------------------------------
    # Eje X
    # --------------------------------------------------------

    ax.set_xticks(
        range(len(column_labels))
    )

    ax.set_xticklabels(
        column_labels
    )

    ax.set_xlabel("")

    # --------------------------------------------------------
    # Valores dentro de cada celda
    # --------------------------------------------------------

    for row in range(matrix.shape[0]):

        for col in range(matrix.shape[1]):

            value = matrix.iloc[
                row,
                col,
            ]

            # Texto negro para fondos claros y
            # blanco para fondos más oscuros
            if value < 0.65 * vmax:
                text_color = "black"
            else:
                text_color = "white"

            ax.text(
                col,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    # --------------------------------------------------------
    # Barra de escala
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        colorbar_label
    )


def create_heatmap(
    data,
    title_label,
    output_file,
):

    """
    Genera una imagen con dos heatmaps alineados
    horizontalmente:

        - Izquierda: métricas de ranking @5 y @10 (escala 0-1).
        - Derecha: distribución de scores de los chunks @10
          (escala 0-100).

    Las estrategias aparecen en el mismo orden en ambos
    (de mejor a peor según Mean_nDCG@10), y sus nombres
    solo se muestran en el heatmap izquierdo.
    """

    matrix_metrics = (
        data
        .set_index("strategy")[METRICS]
    )

    matrix_scores = (
        data
        .set_index("strategy")[SCORE_COLUMNS]
    )

    # Altura dinámica según número de estrategias
    height = max(
        6,
        len(matrix_metrics) * 0.42,
    )

    # --------------------------------------------------------
    # Figura con dos heatmaps
    # --------------------------------------------------------
    #
    # sharey=True garantiza que las filas (estrategias)
    # coinciden exactamente en ambos heatmaps.
    #
    # wspace deja un hueco entre la barra de escala del
    # heatmap izquierdo y el heatmap derecho.
    # --------------------------------------------------------

    fig, (ax_metrics, ax_scores) = plt.subplots(
        1,
        2,
        figsize=(21, height),
        sharey=True,
        gridspec_kw={
            "wspace": 0.35,
        },
    )

    # --------------------------------------------------------
    # Heatmap izquierdo: métricas
    # --------------------------------------------------------

    draw_heatmap(
        ax=ax_metrics,
        matrix=matrix_metrics,
        column_labels=METRIC_LABELS,
        vmax=1,
        value_format=".3f",
        colorbar_label="Score",
        fig=fig,
    )

    ax_metrics.set_yticks(
        range(len(matrix_metrics))
    )

    ax_metrics.set_yticklabels(
        matrix_metrics.index
    )

    ax_metrics.set_ylabel("Strategy")

    ax_metrics.set_title("Retrieval metrics")

    # --------------------------------------------------------
    # Heatmap derecho: porcentajes de score
    # --------------------------------------------------------

    draw_heatmap(
        ax=ax_scores,
        matrix=matrix_scores,
        column_labels=SCORE_LABELS,
        vmax=100,
        value_format=".1f",
        colorbar_label="% of retrieved chunks",
        fig=fig,
    )

    # Ocultar los nombres de las estrategias en el
    # heatmap derecho (ya aparecen en el izquierdo)
    ax_scores.tick_params(
        axis="y",
        labelleft=False,
        left=False,
    )

    ax_scores.set_title("Relevance score distribution")

    # --------------------------------------------------------
    # Título general
    # --------------------------------------------------------

    fig.suptitle(
        f"Retrieval performance – "
        f"{title_label}",
        fontsize=14,
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------
    #
    # No se usa tight_layout porque sobrescribiría el
    # espacio entre heatmaps definido en wspace;
    # bbox_inches="tight" recorta los márgenes al guardar.
    # --------------------------------------------------------

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# DOTPLOT
# ============================================================

def create_dotplot(
    data,
    title_label,
    output_file,
):

    """
    Genera dotplots separados para las métricas @5 y @10.

    Las estrategias están ordenadas de mejor a peor
    según Mean_nDCG@10.
    """

    # Invertir el dataframe para que la estrategia
    # con mejor resultado aparezca arriba
    plot_data = data.iloc[::-1].copy()

    height = max(
        6,
        len(plot_data) * 0.42,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, height),
        sharey=True,
    )

    y_positions = list(
        range(len(plot_data))
    )


    markers = [
        "o",
        "s",
        "^",
    ]

    for ax, cutoff in zip(axes, (5, 10)):
        metrics = [
            f"Mean_nDCG@{cutoff}",
            f"HitRate@{cutoff}",
            f"MRR@{cutoff}",
        ]
        metric_labels = [
            f"Mean nDCG@{cutoff}",
            f"Hit Rate@{cutoff}",
            f"MRR@{cutoff}",
        ]

        for metric, label, marker in zip(
            metrics,
            metric_labels,
            markers,
        ):
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
        ax.set_title(f"Retrieval metrics @{cutoff}")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right", frameon=False)

    axes[0].set_ylabel("Strategy")
    axes[1].tick_params(axis="y", labelleft=False, left=False)
    fig.suptitle(f"Retrieval performance – {title_label}")
    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    eval_config_file = BASE_DIR / "configs" / args.config
    if not eval_config_file.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo de configuración:\n"
            f"{eval_config_file}"
        )

    # Leer configuración de evaluación
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

    # Leer las métricas de la colección correspondiente
    metrics_filename = (
        args.metrics_file
        or f"metrics_{judge_name}_judge_{suffix}.xlsx"
    )
    metrics_file = (
        METRICS_DIR
        / output_folder
        / metrics_filename
    )

    # --------------------------------------------------------
    # Comprobar archivo
    # --------------------------------------------------------

    if not metrics_file.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n"
            f"{metrics_file}"
        )


    # --------------------------------------------------------
    # Leer Excel
    # --------------------------------------------------------

    df = pd.read_excel(
        metrics_file,
        sheet_name="summary",
    )


    # --------------------------------------------------------
    # Comprobar columnas
    # --------------------------------------------------------

    required_columns = {
        "strategy",
        "Mean_nDCG@5",
        "HitRate@5",
        "MRR@5",
        "Mean_nDCG@10",
        "HitRate@10",
        "MRR@10",
        "%score2@10",
        "%score1@10",
        "%score0@10",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Faltan columnas necesarias en el Excel: "
            + ", ".join(sorted(missing))
        )


    # --------------------------------------------------------
    # Crear carpeta de salida
    # --------------------------------------------------------

    # Carpeta de gráficas específica de la colección
    collection_output_dir = OUTPUT_DIR / output_folder

    collection_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # GENERAR GRÁFICAS
    # ========================================================

    title_label = get_title_label(
        df,
        metrics_file,
    )

    # Los nombres de las gráficas se basan en el nombre del
    # archivo de métricas, para no sobrescribir las de otros
    # tipos de documento o modelos
    suffix = metrics_file.stem

    if suffix.startswith("metrics_"):
        suffix = suffix[len("metrics_"):]

    print(
        f"\nGenerando gráficas para: "
        f"{metrics_file.name}"
    )


    # --------------------------------------------------------
    # Preparar y ordenar datos
    # --------------------------------------------------------

    data = prepare_data(
        df,
    )


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap_file = (
        collection_output_dir
        / f"heatmap_{suffix}.png"
    )

    create_heatmap(
        data,
        title_label,
        heatmap_file,
    )


    # --------------------------------------------------------
    # Dotplot
    # --------------------------------------------------------

    dotplot_file = (
        collection_output_dir
        / f"dotplot_{suffix}.png"
    )

    create_dotplot(
        data,
        title_label,
        dotplot_file,
    )


    print(
        f"  Heatmap: {heatmap_file}"
    )

    print(
        f"  Dotplot: {dotplot_file}"
    )


    print(
        "\nGráficas generadas correctamente."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()