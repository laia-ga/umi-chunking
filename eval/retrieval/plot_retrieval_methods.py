
"""Compara Dense + Reranking, Dense, BM25, RRF y Weighted RRF.

Lee los Excel de métricas producidos por run_metrics.py.

Genera:
- Curvas de Mean nDCG, HitRate y MRR por k.
- Heatmaps de las tres métricas.
- Barras de porcentajes de relevancia 0/1/2 a k=10.
- CSV con las métricas y porcentajes.

Orden fijo:
1. Dense + Reranking
2. Dense
3. Weighted RRF 4:1
4. Weighted RRF 3:1
5. RRF
6. Weighted RRF 1:3
7. BM25
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

INPUT_DIR = BASE_DIR / "output" / "metrics"
OUTPUT_DIR = BASE_DIR / "output" / "graphics" / "retrieval_methods"


# ============================================================
# ARCHIVOS DE MÉTRICAS
# ============================================================

# Los seis archivos originales se mantienen.
# El Excel de reranking se indica mediante --reranking-excel
# o se detecta automáticamente si hay una única coincidencia.

FILES = {
    "Dense + Reranking": (
        "qwen_uro_JSON_reranking/"
        "metrics_local_judge_qwen_reranker_dense_c20_top10.xlsx"
    ),

    "Dense": (
        "qwen_uro_JSON_dense/"
        "metrics_local_judge_qwen3_embedding_4b_dense_top30_urology.xlsx"
    ),

    "Weighted RRF 4:1": (
        "qwen_uro_JSON_wrrf_4_1/"
        "metrics_local_judge_qwen3_embedding_4b_weighted_rrf_n30_rrfk2_d4p0_b1p0_top30_urology.xlsx"
    ),

    "Weighted RRF 3:1": (
        "qwen_uro_JSON_wrrf_3_1/"
        "metrics_local_judge_qwen3_embedding_4b_weighted_rrf_n30_rrfk2_d3p0_b1p0_top30_urology.xlsx"
    ),

    "RRF": (
        "qwen_uro_JSON_rrf/"
        "metrics_local_judge_qwen3_embedding_4b_rrf_n30_rrfk2_top30_urology.xlsx"
    ),

    "Weighted RRF 1:3": (
        "qwen_uro_JSON_wrrf_1_3/"
        "metrics_local_judge_qwen3_embedding_4b_weighted_rrf_n30_rrfk2_d1p0_b3p0_top30_urology.xlsx"
    ),

    "BM25": (
        "qwen_uro_JSON_bm25/"
        "metrics_local_judge_qwen3_embedding_4b_bm25_top30_urology.xlsx"
    ),
}


# ============================================================
# ORDEN FIJO DE TODOS LOS GRÁFICOS
# ============================================================

METHOD_ORDER = [
    "Dense + Reranking",
    "Dense",
    "Weighted RRF 4:1",
    "Weighted RRF 3:1",
    "RRF",
    "Weighted RRF 1:3",
    "BM25",
]

K_VALUES = (3, 5, 7, 10)

METRICS = {
    "Mean_nDCG": "Mean nDCG",
    "HitRate": "HitRate",
    "MRR": "MRR",
}

METHOD_COLORS = {
    "Dense + Reranking": "#D45C85",
    "Dense": "#4477AA",
    "Weighted RRF 4:1": "#C9D576",
    "Weighted RRF 3:1": "#AA7744",
    "RRF": "#228833",
    "Weighted RRF 1:3": "#8855AA",
    "BM25": "#CC6677",
}

METHOD_MARKERS = {
    "Dense + Reranking": "*",
    "Dense": "o",
    "Weighted RRF 4:1": "s",
    "Weighted RRF 3:1": "^",
    "RRF": "D",
    "Weighted RRF 1:3": "v",
    "BM25": "P",
}

SCORE_COLORS = {
    0: "#AAB4C2",
    1: "#E6B85C",
    2: "#399B85",
}


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--strategy",
        default="JSONChunking",
        help="Estrategia de la hoja summary.",
    )

    parser.add_argument(
        "--reranking-excel",
        type=Path,
        default=None,
        help=(
            "Ruta al Excel de Dense + Reranking. "
            "Si es relativa, se interpreta desde --input-dir. "
            "Si se omite, se busca automáticamente."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if args.dpi < 50:
        parser.error("--dpi debe ser al menos 50")

    return args


# ============================================================
# LOCALIZAR EXCEL DE RERANKING
# ============================================================

def resolve_reranking_excel(input_dir, specified_path):
    """Localiza el Excel evaluado de Dense + Reranking."""

    if specified_path is not None:
        path = (
            specified_path
            if specified_path.is_absolute()
            else input_dir / specified_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"No existe el Excel de reranking: {path}"
            )

        return path

    # Buscar únicamente Excel asociados a reranking
    # cuyo nombre o ruta indique top10.
    candidates = []

    for path in input_dir.rglob("*.xlsx"):

        relative = str(path.relative_to(input_dir)).lower()

        if (
            "rerank" in relative
            and any(
                token in relative
                for token in ("top10", "top_10", "top-10")
            )
            and not path.name.startswith("~$")
        ):
            candidates.append(path)

    if len(candidates) == 1:
        print(
            "Excel de reranking detectado: "
            f"{candidates[0]}"
        )
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "No se ha encontrado el Excel de Dense + Reranking. "
            "Indica su ruta con --reranking-excel."
        )

    listed = "\n".join(str(p) for p in candidates)

    raise ValueError(
        "Se han encontrado varios Excel de reranking:\n"
        f"{listed}\n"
        "Selecciona uno con --reranking-excel."
    )


# ============================================================
# SELECCIONAR ESTRATEGIA
# ============================================================

def select_strategy(table, path, strategy):
    """Selecciona la fila de la estrategia solicitada."""

    if table.empty:
        raise ValueError(
            f"Hoja summary vacía: {path}"
        )

    if "strategy" not in table.columns:

        if len(table) == 1:
            warnings.warn(
                f"{path.name}: no existe 'strategy'; "
                "se utiliza su única fila."
            )
            return table.iloc[0]

        raise ValueError(
            f"{path}: falta la columna strategy."
        )

    normalized = (
        table["strategy"]
        .astype(str)
        .str.replace(
            r"[^a-zA-Z0-9]",
            "",
            regex=True,
        )
        .str.lower()
    )

    expected = "".join(
        char
        for char in strategy.lower()
        if char.isalnum()
    )

    matches = table.loc[
        normalized.eq(expected)
    ]

    if len(matches) != 1:

        if len(table) == 1 and len(matches) == 0:
            warnings.warn(
                f"{path.name}: la estrategia es "
                f"'{table.iloc[0]['strategy']}', "
                f"distinta de '{strategy}'. "
                "Se utiliza la única fila; comprueba "
                "que corresponde al experimento deseado."
            )
            return table.iloc[0]

        raise ValueError(
            f"{path}: esperaba una fila para "
            f"'{strategy}', encontradas {len(matches)}."
        )

    return matches.iloc[0]


# ============================================================
# VALIDAR VALORES NUMÉRICOS
# ============================================================

def numeric_cell(
    row,
    column,
    path,
    *,
    low=None,
    high=None,
):
    if column not in row.index:
        raise ValueError(
            f"Falta la columna '{column}' en {path}."
        )

    value = pd.to_numeric(
        row[column],
        errors="coerce",
    )

    if pd.isna(value) or not np.isfinite(float(value)):
        raise ValueError(
            f"Valor inválido en {column} de {path}: "
            f"{row[column]!r}"
        )

    value = float(value)

    if (
        (low is not None and value < low)
        or (high is not None and value > high)
    ):
        raise ValueError(
            f"{column} fuera del rango "
            f"[{low}, {high}] en {path}: {value}"
        )

    return value


# ============================================================
# CARGAR TODOS LOS EXPERIMENTOS
# ============================================================

def load_all(input_dir, strategy, reranking_excel):
    """Carga los siete métodos en el orden solicitado."""

    reranking_path = resolve_reranking_excel(
        input_dir,
        reranking_excel,
    )

    summary_rows = []
    metric_rows = []
    percent_rows = []

    for method in METHOD_ORDER:

        if method == "Dense + Reranking":
            path = reranking_path
        else:
            path = input_dir / FILES[method]

        if not path.is_file():
            raise FileNotFoundError(
                f"No se encuentra el Excel de {method}: "
                f"{path}"
            )

        table = pd.read_excel(
            path,
            sheet_name="summary",
        )

        table.columns = (
            table.columns.astype(str).str.strip()
        )

        row = select_strategy(
            table,
            path,
            strategy,
        )

        record = {
            "method": method,
            "strategy": str(
                row.get("strategy", strategy)
            ),
        }

        # --------------------------------------------
        # MÉTRICAS @3, @5, @7 y @10
        # --------------------------------------------

        for metric in METRICS:

            for k in K_VALUES:

                field = f"{metric}@{k}"

                value = numeric_cell(
                    row,
                    field,
                    path,
                    low=0,
                    high=1,
                )

                record[field] = value

                count_key = (
                    f"n_questions_evaluated@{k}"
                )

                count = pd.to_numeric(
                    row.get(count_key, np.nan),
                    errors="coerce",
                )

                metric_rows.append({
                    "method": method,
                    "metric": metric,
                    "k": k,
                    "value": value,
                    "n_questions_evaluated": count,
                })

        # --------------------------------------------
        # PORCENTAJES DE SCORE @10
        # --------------------------------------------

        n_chunks = pd.to_numeric(
            row.get("n_chunks", np.nan),
            errors="coerce",
        )

        record["n_chunks_valid_at_10"] = n_chunks

        for score in (0, 1, 2):

            key = f"%score{score}@10"

            value = numeric_cell(
                row,
                key,
                path,
                low=0,
                high=100,
            )

            record[key] = value

            percent_rows.append({
                "method": method,
                "score": score,
                "percentage": value,
                "k": 10,
                "n_chunks_valid": n_chunks,
            })

        total_pct = sum(
            record[f"%score{s}@10"]
            for s in (0, 1, 2)
        )

        if (
            np.isfinite(n_chunks)
            and n_chunks <= 0
        ):
            warnings.warn(
                f"{method}: n_chunks=0."
            )

        elif abs(total_pct - 100) > 0.3:
            warnings.warn(
                f"{method}: los porcentajes suman "
                f"{total_pct:.2f}% en lugar de 100%."
            )

        summary_rows.append(record)

        print(f"Leído: {method} — {path}")

    # El orden de las filas se conserva.
    summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(metric_rows)
    percentages = pd.DataFrame(percent_rows)

    # Comprobar tamaños muestrales.
    for k in K_VALUES:

        subset = metrics.loc[
            (metrics["k"] == k)
            & (metrics["metric"] == "Mean_nDCG")
        ]

        counts = (
            subset.set_index("method")[
                "n_questions_evaluated"
            ]
            .dropna()
        )

        if counts.nunique() > 1:
            warnings.warn(
                f"k={k}: distinto número de "
                f"preguntas evaluadas por método: "
                f"{counts.to_dict()}"
            )

    return summary, metrics, percentages


# ============================================================
# AJUSTE DEL EJE Y
# ============================================================

def compact_ylim(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    lo = values.min()
    hi = values.max()

    pad = max(
        (hi - lo) * 0.12,
        0.025,
    )

    return (
        max(0.0, lo - pad),
        min(1.0, hi + pad),
    )


# ============================================================
# CURVAS DE MÉTRICAS POR K
# ============================================================

def plot_curves(metrics, output, dpi):

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(19, 6.7),
        sharex=True,
    )

    for ax, (metric, display) in zip(
        axs,
        METRICS.items(),
    ):

        part = metrics.loc[
            metrics["metric"] == metric
        ]

        # Dibujar las curvas siguiendo el orden fijo.
        for method in METHOD_ORDER:

            subset = (
                part.loc[
                    part["method"] == method
                ]
                .sort_values("k")
            )

            ax.plot(
                subset["k"],
                subset["value"],
                label=method,
                marker=METHOD_MARKERS[method],
                linewidth=2.3,
                markersize=7,
                color=METHOD_COLORS[method],
            )

        ax.set_title(
            display,
            fontsize=15,
            fontweight="bold",
            pad=12,
        )

        ax.set_xticks(K_VALUES)
        ax.set_xlim(2.5, 10.5)

        ax.set_ylim(
            *compact_ylim(part["value"])
        )

        ax.set_xlabel(
            "Número de chunks recuperados (k)"
        )

        ax.set_ylabel(
            "Puntuación media"
        )

        ax.grid(
            alpha=0.24,
            linestyle="--",
        )

        ax.spines[
            ["top", "right"]
        ].set_visible(False)

    fig.suptitle(
        "Retrieval Qwen3 + JSON Chunking | "
        "rendimiento según k",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    handles, labels = (
        axs[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.82,
        bottom=0.27,
        wspace=0.26,
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Guardado: {output}")


# ============================================================
# HEATMAPS DE MÉTRICAS
# ============================================================

def plot_heatmaps(summary, output, dpi):

    methods = METHOD_ORDER

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(19.5, 6.7),
    )

    cmap = plt.get_cmap("YlGnBu")

    image = None

    for ax, (metric, display) in zip(
        axs,
        METRICS.items(),
    ):

        matrix = (
            summary.set_index("method")
            .reindex(methods)[
                [
                    f"{metric}@{k}"
                    for k in K_VALUES
                ]
            ]
            .to_numpy(dtype=float)
        )

        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        ax.set_title(
            display,
            fontsize=15,
            fontweight="bold",
            pad=13,
        )

        ax.set_xticks(
            range(len(K_VALUES)),
            [f"k={k}" for k in K_VALUES],
        )

        # Orden de arriba abajo: METHOD_ORDER.
        ax.set_yticks(
            range(len(methods)),
            methods,
            fontsize=9.5,
        )

        # Valores numéricos en las celdas.
        for r in range(len(methods)):

            for c in range(len(K_VALUES)):

                val = matrix[r, c]

                ax.text(
                    c,
                    r,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if val > 0.58
                        else "#172b3b"
                    ),
                    fontsize=9.2,
                    fontweight="medium",
                )

        ax.set_xticks(
            np.arange(
                -0.5,
                len(K_VALUES),
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                len(methods),
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=1.8,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        ax.spines[:].set_visible(False)

    fig.suptitle(
        "Comparación de retrieval por métrica y k",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.91,
        bottom=0.11,
        top=0.85,
        wspace=0.45,
    )

    color_ax = fig.add_axes([
        0.93,
        0.18,
        0.014,
        0.57,
    ])

    fig.colorbar(
        image,
        cax=color_ax,
        label="Puntuación (0–1)",
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Guardado: {output}")


# ============================================================
# PORCENTAJES DE RELEVANCIA @10
# ============================================================

def plot_percentages(percentages, output, dpi):

    methods = METHOD_ORDER

    data = (
        percentages.pivot(
            index="method",
            columns="score",
            values="percentage",
        )
        .reindex(methods)
    )

    counts = (
        percentages.loc[
            percentages["score"] == 0
        ]
        .set_index("method")[
            "n_chunks_valid"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7.4)
    )

    y = np.arange(len(methods))
    left = np.zeros(len(methods))

    for score, name in (
        (0, "0 · Irrelevante"),
        (1, "1 · Parcialmente relevante"),
        (2, "2 · Suficiente"),
    ):

        vals = data[score].to_numpy(
            dtype=float
        )

        ax.barh(
            y,
            vals,
            left=left,
            height=0.60,
            color=SCORE_COLORS[score],
            label=name,
            edgecolor="white",
            linewidth=1.2,
        )

        for row, (value, start) in enumerate(
            zip(vals, left)
        ):

            if value >= 8:

                ax.text(
                    start + value / 2,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#1F2937",
                )

        left += vals

    # Aplicar el orden fijo en el eje vertical.
    ax.set_yticks(
        y,
        methods,
        fontsize=11,
    )

    # Primera estrategia arriba.
    ax.invert_yaxis()

    ax.set_xlim(0, 107)

    ax.set_xticks(
        [0, 20, 40, 60, 80, 100],
        ["0%", "20%", "40%", "60%", "80%", "100%"],
    )

    # Número de chunks válidos por método.
    for i, method in enumerate(methods):

        n = counts.get(
            method,
            np.nan,
        )

        if pd.notna(n):
            ax.text(
                101,
                i,
                f"n={int(n)}",
                ha="left",
                va="center",
                fontsize=9,
                color="#51606c",
            )

    ax.set_xlabel(
        "Porcentaje de chunks evaluados "
        "entre los 10 primeros de cada pregunta"
    )

    ax.set_title(
        "Distribución de relevancia AtlaAI | top-10",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.20,
    )

    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)

    ax.legend(
        title="Puntuación del juez",
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=10,
    )

    fig.text(
        0.5,
        0.015,
        "Distribución sobre chunks con juicio válido; "
        "los errores del juez quedan excluidos.",
        ha="center",
        fontsize=9,
        color="#596474",
    )

    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        top=0.85,
        bottom=0.23,
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Guardado: {output}")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    summary, metrics, percentages = load_all(
        input_dir=args.input_dir,
        strategy=args.strategy,
        reranking_excel=args.reranking_excel,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exportar también los CSV en el nuevo orden.
    summary.to_csv(
        args.output_dir / "retrieval_methods_summary.csv",
        index=False,
    )

    metrics.to_csv(
        args.output_dir / "retrieval_methods_metrics_by_k.csv",
        index=False,
    )

    percentages.to_csv(
        args.output_dir
        / "retrieval_methods_score_percentages_at_10.csv",
        index=False,
    )

    # Gráfica 1: evolución de las métricas.
    plot_curves(
        metrics,
        args.output_dir
        / "retrieval_metrics_curves_4k.png",
        args.dpi,
    )

    # Gráfica 2: heatmaps.
    plot_heatmaps(
        summary,
        args.output_dir
        / "retrieval_metrics_heatmaps_4k.png",
        args.dpi,
    )

    # Gráfica 3: distribución de relevancia.
    plot_percentages(
        percentages,
        args.output_dir
        / "retrieval_score_distribution_at_10.png",
        args.dpi,
    )

    print(
        "\nComparación completada. "
        "Resultados en:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()

