import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

INPUT_DIRS = {
    "Corto": Path("output/chunks/e5_256_512"),
    "Medio": Path("output/chunks/bge_m3_512_8192"),
    "Largo": Path("output/chunks/qwen_1024_32768"),
}

OUTPUT_DIR = Path("output/graphics/time_chunking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = ["Corto", "Medio", "Largo"]

COLORS = {
    "Corto": "#83CDBD",
    "Medio": "#348F98",
    "Largo": "#244A70",
}

FAMILIES = {
    "JSON": [
        "JSONChunking",
    ],
    "Recursive": [
        "parent_child_chunking",
        "recursive_chunking",
        "recursive_token_fallback_chunking",
    ],
    "Dynamic": [
        "content_density_adaptive_chunking",
        "dynamic_token_size_chunking",
        "semantic_variance_adaptive_chunking",
    ],
    "LLM": [
        "llm_boundary_detection_chunking",
        "llm_segment_then_chunk",
    ],
    "Rulebased": [
        "fixed_character_chunking",
        "fixed_token_chunking",
        "length_aware_chunking",
        "overlapping_token_chunking",
        "paragraph_based_chunking",
        "paragraph_group_chunking",
        "sentence_based_chunking",
        "sentence_group_chunking",
        "sliding_window_token_chunking",
    ],
    "Semantic": [
        "semantic_boundary_detection",
        "semantic_embedding_based_chunking",
        "semantic_similarity_threshold_chunking",
        "topic_based_chunking",
    ],
}

FAMILY_COLORS = {
    "JSON": "#C07A43",
    "Recursive": "#8166A8",
    "Dynamic": "#368B80",
    "LLM": "#B65E83",
    "Rulebased": "#4779B2",
    "Semantic": "#B39843",
}

# Equivalencias entre nombres de configuración y JSON
ALIASES = {
    "jsonchunking": "JSONChunking",
    "json_chunking": "JSONChunking",
    "recursive_token_chunking": "recursive_token_fallback_chunking",
    "llm_segment_then_chunking": "llm_segment_then_chunk",
    "semantic_boundary_chunking": "semantic_boundary_detection",
    "semantic_embedding_chunking": "semantic_embedding_based_chunking",
}

STRATEGY_FAMILY = {
    strategy: family
    for family, strategies in FAMILIES.items()
    for strategy in strategies
}


def normalize_strategy(strategy):
    """Unifica identificadores equivalentes."""
    strategy = str(strategy).strip()
    return ALIASES.get(strategy.lower(), strategy)


def pretty_name(strategy):
    """Convierte identificadores a nombres legibles."""
    if strategy == "JSONChunking":
        return "JSON Chunking"

    return strategy.replace("_", " ").title()


# ============================================================
# 2. LECTURA DE MÉTRICAS
# ============================================================

records = []

for size, base_dir in INPUT_DIRS.items():

    for document_format in ["json", "markdown"]:

        metrics_dir = base_dir / document_format / "metrics"

        if not metrics_dir.exists():
            print(f"Carpeta no encontrada: {metrics_dir}")
            continue

        files = sorted(metrics_dir.glob("*.json"))

        print(
            f"{size} | {document_format}: "
            f"{len(files)} archivos"
        )

        for file in files:

            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("status") != "completed":
                    continue

                execution_time = data.get("execution_time_seconds")

                if execution_time is None:
                    continue

                execution_time = float(execution_time)

                if not np.isfinite(execution_time):
                    continue

                strategy = normalize_strategy(data["strategy"])
                abbreviation = str(data["abbreviation"]).strip()

                records.append({
                    "size": size,
                    "format": document_format,
                    "strategy": strategy,
                    "abbreviation": abbreviation,
                    "execution_time_seconds": execution_time,
                    "document": data.get("doc_id"),
                    "tokens_original": data.get("tokens_original"),
                    "number_of_chunks": data.get("number_of_chunks"),
                })

            except (OSError, ValueError, KeyError, TypeError) as e:
                print(f"Error leyendo {file}: {e}")


df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No se encontraron métricas válidas.")

print(f"\nRegistros válidos: {len(df)}")
print(f"Métodos encontrados: {df['strategy'].nunique()}")


# ============================================================
# 3. CALCULAR TIEMPOS MEDIOS
# ============================================================

df_mean = (
    df.groupby(
        ["strategy", "abbreviation", "size"],
        as_index=False
    )
    .agg(
        mean_time=("execution_time_seconds", "mean"),
        std_time=("execution_time_seconds", "std"),
        n_documents=("execution_time_seconds", "count"),
    )
)

# Ordenar por media de los tres tamaños, sin ponderar
# por el número de documentos de cada tamaño.

strategy_order = (
    df_mean.groupby("strategy")["mean_time"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

abbreviations = (
    df.drop_duplicates("strategy")
    .set_index("strategy")["abbreviation"]
)

df_plot = (
    df_mean.pivot_table(
        index="strategy",
        columns="size",
        values="mean_time",
        aggfunc="mean"
    )
    .reindex(
        index=strategy_order,
        columns=SIZES
    )
)

df_mean.to_csv(
    OUTPUT_DIR / "chunking_mean_times.csv",
    index=False
)

# Comprobar familias
unknown = [
    s for s in strategy_order
    if s not in STRATEGY_FAMILY
]

if unknown:
    print("\nATENCIÓN: métodos sin familia asignada:")
    for s in unknown:
        print(f"  {s}")


# ============================================================
# 4. PREPARAR LEYENDA INFERIOR
# ============================================================

# Ordenar el glosario por familias
legend_entries = []

for family, strategies in FAMILIES.items():

    for strategy in strategies:

        if strategy not in strategy_order:
            continue

        legend_entries.append({
            "abbreviation": abbreviations[strategy],
            "name": pretty_name(strategy),
            "family": family,
        })

# Métodos no reconocidos
for strategy in strategy_order:

    if strategy not in STRATEGY_FAMILY:

        legend_entries.append({
            "abbreviation": abbreviations[strategy],
            "name": pretty_name(strategy),
            "family": None,
        })


# ============================================================
# 5. CREAR FIGURA CON ZONAS INDEPENDIENTES
# ============================================================

fig = plt.figure(figsize=(22, 11.5))

# Tres zonas:
# 1. Gráfica
# 2. Leyenda de familias
# 3. Glosario de abreviaturas

gs = fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[6.2, 0.55, 2.2],
    hspace=0.28,
    left=0.055,
    right=0.98,
    top=0.93,
    bottom=0.055,
)

ax = fig.add_subplot(gs[0])
ax_families = fig.add_subplot(gs[1])
ax_glossary = fig.add_subplot(gs[2])

ax_families.axis("off")
ax_glossary.axis("off")


# ============================================================
# 6. GRÁFICA DE BARRAS
# ============================================================

x = np.arange(len(strategy_order))
width = 0.22

for i, size in enumerate(SIZES):

    offset = (i - 1) * width

    values = df_plot[size].to_numpy(dtype=float)

    ax.bar(
        x + offset,
        values,
        width=width,
        label=size,
        color=COLORS[size],
        edgecolor="white",
        linewidth=0.5,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    [abbreviations[s] for s in strategy_order],
    rotation=55,
    ha="right",
    fontsize=10,
    fontweight="bold"
)

# Colorear las abreviaturas por familia
for tick, strategy in zip(
    ax.get_xticklabels(),
    strategy_order
):

    family = STRATEGY_FAMILY.get(strategy)

    tick.set_color(
        FAMILY_COLORS.get(family, "#444444")
    )

ax.set_ylabel(
    "Tiempo medio de ejecución (segundos)",
    fontsize=12
)

ax.set_xlabel(
    "Método de chunking",
    fontsize=12,
    labelpad=12
)

ax.set_title(
    "Tiempo medio de ejecución por método de chunking",
    fontsize=17,
    fontweight="bold",
    pad=20
)

# Escala logarítmica
ax.set_yscale("log")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Leyenda de longitudes
length_handles = [
    Patch(
        facecolor=COLORS[size],
        label=size
    )
    for size in SIZES
]

ax.legend(
    handles=length_handles,
    title="Longitud de chunk",
    loc="upper left",
    frameon=False,
    fontsize=10
)


# ============================================================
# 7. LEYENDA DE FAMILIAS
# ============================================================

family_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color=color,
        markersize=9,
        label=family
    )
    for family, color in FAMILY_COLORS.items()
]

ax_families.legend(
    handles=family_handles,
    loc="center",
    ncol=6,
    frameon=False,
    fontsize=10,
    title_fontsize=11,
)


# ============================================================
# 8. GLOSARIO DE ABREVIATURAS
# ============================================================

ax_glossary.set_xlim(0, 1)
ax_glossary.set_ylim(0, 1)

ax_glossary.text(
    0.5,
    1.0,
    "Abreviaturas de los métodos",
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    transform=ax_glossary.transAxes
)

ncols = 3
nrows = int(np.ceil(len(legend_entries) / ncols))

# Columnas con espacio suficiente para nombres largos
column_positions = [0.03, 0.36, 0.69]

# Separación vertical automática
y_top = 0.84
y_bottom = 0.08

for i, entry in enumerate(legend_entries):

    col = i // nrows
    row = i % nrows

    x_pos = column_positions[col]

    if nrows > 1:
        y_pos = y_top - row * (
            (y_top - y_bottom) / (nrows - 1)
        )
    else:
        y_pos = y_top

    color = FAMILY_COLORS.get(
        entry["family"],
        "#444444"
    )

    # Abreviatura coloreada
    ax_glossary.text(
        x_pos,
        y_pos,
        f"{entry['abbreviation']}:",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax_glossary.transAxes
    )

    # Nombre completo
    ax_glossary.text(
        x_pos + 0.052,
        y_pos,
        entry["name"],
        fontsize=9.2,
        color="#333333",
        ha="left",
        va="center",
        transform=ax_glossary.transAxes
    )


# ============================================================
# 9. GUARDAR GRÁFICA
# ============================================================

output_file = OUTPUT_DIR / "chunking_execution_time.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(f"\nGráfica guardada en: {output_file}")


# ============================================================
# 10. RESUMEN
# ============================================================

print("\nTiempo medio por método y tamaño:")
print(df_plot.round(4).to_string())

print("\nNúmero de documentos por método y tamaño:")

df_counts = (
    df_mean.pivot_table(
        index="strategy",
        columns="size",
        values="n_documents",
        aggfunc="sum"
    )
    .reindex(
        index=strategy_order,
        columns=SIZES
    )
)

print(df_counts.to_string())

print(f"\nResultados guardados en: {OUTPUT_DIR}")