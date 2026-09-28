import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

GRAPHICS_DIR = Path(__file__).resolve().parent
BASE_DIR = GRAPHICS_DIR.parent

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
        help=(
            "Nombre del archivo .xlsx situado en "
            "output/metrics"
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
    Ordena las estrategias de mejor a peor según
    Mean_nDCG@5.
    """

    data = df.copy()

    if data.empty:

        raise ValueError(
            "El archivo de métricas no contiene resultados."
        )

    # --------------------------------------------------------
    # nDCG@5 para ordenar
    # --------------------------------------------------------

    data = data.sort_values(
        "Mean_nDCG@5",
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
]

METRIC_LABELS = [
    "Mean nDCG@5",
    "Hit Rate@5",
    "MRR@5",
]

# Porcentajes del heatmap derecho (escala 0-100)
SCORE_COLUMNS = [
    "%score2",
    "%score1",
    "%score0",
]

SCORE_LABELS = [
    "% score 2",
    "% score 1",
    "% score 0",
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

        - Izquierda: Mean nDCG@5, HitRate@5 y MRR@5
          (escala 0-1).
        - Derecha: porcentaje de chunks con score 2, 1 y 0
          (escala 0-100).

    Las estrategias aparecen en el mismo orden en ambos
    (de mejor a peor según Mean_nDCG@5), y sus nombres
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
        figsize=(15, height),
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
    Genera un dotplot con las tres métricas para cada
    estrategia.

    Las estrategias están ordenadas de mejor a peor
    según Mean_nDCG@5.
    """

    metrics = [
        "Mean_nDCG@5",
        "HitRate@5",
        "MRR@5",
    ]

    metric_labels = [
        "Mean nDCG@5",
        "Hit Rate@5",
        "MRR@5",
    ]

    # Invertir el dataframe para que la estrategia
    # con mejor resultado aparezca arriba
    plot_data = data.iloc[::-1].copy()

    height = max(
        6,
        len(plot_data) * 0.42,
    )

    # Algo más de anchura para dejar espacio
    # a la leyenda fuera de la gráfica
    fig, ax = plt.subplots(
        figsize=(11, height)
    )

    y_positions = list(
        range(len(plot_data))
    )


    # --------------------------------------------------------
    # Dibujar cada métrica
    # --------------------------------------------------------

    markers = [
        "o",
        "s",
        "^",
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


    # --------------------------------------------------------
    # Ejes
    # --------------------------------------------------------

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        plot_data["strategy"]
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_ylabel(
        "Strategy"
    )

    ax.set_title(
        f"Retrieval performance – "
        f"{title_label}"
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    # Solo líneas verticales para facilitar la comparación
    # de los valores sin sobrecargar la gráfica
    ax.grid(
        axis="x",
        alpha=0.25,
    )


    # --------------------------------------------------------
    # Leyenda
    # --------------------------------------------------------

    # Se coloca fuera de la zona de representación para
    # garantizar que no tape ningún punto.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False,
    )


    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    # Reservar espacio a la derecha para la leyenda
    fig.tight_layout(
        rect=[0, 0, 0.82, 1]
    )

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

    metrics_file = (
        METRICS_DIR
        / args.metrics_file
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
        "%score2",
        "%score1",
        "%score0",
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

    OUTPUT_DIR.mkdir(
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
        OUTPUT_DIR
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
        OUTPUT_DIR
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