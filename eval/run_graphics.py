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
            "Genera heatmaps y dotplots de las métricas "
            "de retrieval para papers y guidelines."
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
    document_type,
):

    """
    Filtra un tipo de documento y ordena las estrategias
    de mejor a peor según la media de:

        - Mean_nDCG@5
        - HitRate@5
        - MRR

    Esta media se utiliza únicamente para ordenar las
    estrategias y no aparece en las gráficas.
    """

    data = df[
        df["document_type"]
        .astype(str)
        .str.lower()
        == document_type.lower()
    ].copy()

    if data.empty:

        raise ValueError(
            f"No se encontraron resultados para "
            f"document_type = '{document_type}'."
        )

    # --------------------------------------------------------
    # Media de las tres métricas únicamente para ordenar
    # --------------------------------------------------------

    data["_mean_metrics"] = data[
        [
            "Mean_nDCG@5",
            "HitRate@5",
            "MRR",
        ]
    ].mean(
        axis=1
    )

    # Ordenar de mejor a peor
    data = data.sort_values(
        "_mean_metrics",
        ascending=False,
    )

    # Eliminar la columna auxiliar
    data = data.drop(
        columns="_mean_metrics"
    )

    return data


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    data,
    document_type,
    output_file,
):

    """
    Genera un heatmap con las estrategias en filas y
    Mean nDCG@5, HitRate@5 y MRR en columnas.

    Las estrategias están ordenadas de mejor a peor
    según la media de las tres métricas.
    """

    metrics = [
        "Mean_nDCG@5",
        "HitRate@5",
        "MRR",
    ]

    metric_labels = [
        "Mean nDCG@5",
        "Hit Rate@5",
        "MRR",
    ]

    matrix = (
        data
        .set_index("strategy")[metrics]
    )

    # Altura dinámica según número de estrategias
    height = max(
        6,
        len(matrix) * 0.42,
    )

    fig, ax = plt.subplots(
        figsize=(8, height)
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------
    #
    # YlGnBu proporciona colores relativamente suaves
    # manteniendo una escala visual clara.
    #
    # La escala se fija entre 0 y 1 para que todas las
    # métricas y ambos tipos de documento sean comparables.
    # --------------------------------------------------------

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )


    # --------------------------------------------------------
    # Ejes
    # --------------------------------------------------------

    ax.set_xticks(
        range(len(metrics))
    )

    ax.set_xticklabels(
        metric_labels
    )

    ax.set_yticks(
        range(len(matrix))
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlabel("")
    ax.set_ylabel("Strategy")

    ax.set_title(
        f"Retrieval performance – "
        f"{document_type.capitalize()}"
    )


    # --------------------------------------------------------
    # Valores dentro de cada celda
    # --------------------------------------------------------

    for row in range(len(matrix)):

        for col in range(len(metrics)):

            value = matrix.iloc[
                row,
                col,
            ]

            # Texto negro para fondos claros y
            # blanco para fondos más oscuros
            if value < 0.65:
                text_color = "black"
            else:
                text_color = "white"

            ax.text(
                col,
                row,
                f"{value:.3f}",
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
        "Score"
    )


    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    fig.tight_layout()

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
    document_type,
    output_file,
):

    """
    Genera un dotplot con las tres métricas para cada
    estrategia.

    Las estrategias están ordenadas de mejor a peor
    según la media de las tres métricas.
    """

    metrics = [
        "Mean_nDCG@5",
        "HitRate@5",
        "MRR",
    ]

    metric_labels = [
        "Mean nDCG@5",
        "Hit Rate@5",
        "MRR",
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
        f"{document_type.capitalize()}"
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
        "document_type",
        "Mean_nDCG@5",
        "HitRate@5",
        "MRR",
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

    for document_type in [
        "paper",
        "guideline",
    ]:

        print(
            f"\nGenerando gráficas para: "
            f"{document_type}"
        )


        # ----------------------------------------------------
        # Preparar y ordenar datos
        # ----------------------------------------------------

        data = prepare_data(
            df,
            document_type,
        )


        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        heatmap_file = (
            OUTPUT_DIR
            / f"heatmap_{document_type}.png"
        )

        create_heatmap(
            data,
            document_type,
            heatmap_file,
        )


        # ----------------------------------------------------
        # Dotplot
        # ----------------------------------------------------

        dotplot_file = (
            OUTPUT_DIR
            / f"dotplot_{document_type}.png"
        )

        create_dotplot(
            data,
            document_type,
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