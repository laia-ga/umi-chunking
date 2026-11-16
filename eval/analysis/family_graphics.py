
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

INPUT_DIR = Path("output/metrics")

OUTPUT_DIR = Path("output/graphics/aggregation/families")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "BGE-M3": "all_bge_m3_atlaai/metrics_local_judge_bge_m3.xlsx",
    "Qwen3": "all_qwen3_4b_atlaai/metrics_local_judge_qwen3_embedding_4b.xlsx",
    "Multilingual-E5": "all_multilingual_e5_atlaai/metrics_local_judge_multilingual_e5_large.xlsx",
    "MedCPT": "all_medcpt_atlaai/metrics_local_judge_medcpt_query_encoder.xlsx",
    # "Jina": "all_jina_v2_atlaai/metrics_local_judge_jina_embeddings_v2_base_es.xlsx",
}

METRICS = [
    "Mean_nDCG@10",
    "HitRate@10",
    "MRR@10"
]


# ============================================================
# 2. DEFINIR FAMILIAS MANUALMENTE
# ============================================================

# Clave: nombre de la familia que aparecerá en los gráficos.
# Valor: estrategias que pertenecen a esa familia.
#
# IMPORTANTE:
# Los nombres de las estrategias deben coincidir con los
# valores de la columna "strategy" de los archivos Excel.

FAMILIES = {

    "JSON": [
        "JSONChunking",
    ],

    "Recursive": [
        "parent_child_chunking",
        "recursive_chunking", "recursive_token_fallback_chunking"
    ],

    "Dynamic": [
        "content_density_adaptive_chunking",
        "dynamic_token_size_chunking", "semantic_variance_adaptive_chunking"
    ],

    "LLM": [
        "llm_boundary_detection_chunking", "llm_segment_then_chunk"
    ],

    "Rulebased": [
        "fixed_character_chunking", "fixed_token_chunking", "length_aware_chunking", 
        "overlapping_token_chunking", "paragraph_based_chunking", "paragraph_group_chunking", 
        "sentence_based_chunking", "sentence_group_chunking", "sliding_window_token_chunking"
    ],

    "Semantic": [
        "semantic_boundary_detection", "semantic_embedding_based_chunking", 
        "semantic_similarity_threshold_chunking", "topic_based_chunking"
    ]

}

# Sustituye los nombres anteriores por tus estrategias reales.


# ============================================================
# 3. CONFIGURACIÓN DE GRÁFICOS
# ============================================================

# Rojo = valores bajos; azul = valores altos
HEATMAP_CMAP = "Spectral"

# Paleta para identificar las familias
LINE_CMAP = "Spectral_r"

HEATMAP_VMIN = 0
HEATMAP_VMAX = 1


# ============================================================
# 4. CARGAR DATOS
# ============================================================

dataframes = []

for model, filename in FILES.items():

    filepath = INPUT_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {filepath}"
        )

    df = pd.read_excel(filepath)

    df.columns = df.columns.astype(str).str.strip()

    required_columns = ["strategy"] + METRICS

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas en {filename}: {missing}"
        )

    df = df[required_columns].copy()

    df["strategy"] = (
        df["strategy"].astype(str).str.strip()
    )

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

    df["embedding_model"] = model

    dataframes.append(df)


df_all = pd.concat(
    dataframes,
    ignore_index=True
)

models = list(FILES.keys())


# ============================================================
# 5. VALIDACIONES
# ============================================================

# Comprobar combinaciones duplicadas

duplicates = df_all.duplicated(
    subset=["embedding_model", "strategy"],
    keep=False
)

if duplicates.any():

    raise ValueError(
        "Hay combinaciones duplicadas de modelo y estrategia:\n"
        + str(
            df_all.loc[
                duplicates,
                ["embedding_model", "strategy"]
            ]
        )
    )


# Comprobar que ninguna estrategia pertenece a dos familias

all_methods = [
    method.strip()
    for methods in FAMILIES.values()
    for method in methods
]

duplicated_methods = sorted({
    method for method in all_methods
    if all_methods.count(method) > 1
})

if duplicated_methods:
    raise ValueError(
        "Hay estrategias asignadas a varias familias: "
        f"{duplicated_methods}"
    )


# Crear correspondencia estrategia -> familia

strategy_to_family = {
    method.strip(): family
    for family, methods in FAMILIES.items()
    for method in methods
}


# Comprobar que todas las estrategias tienen familia

strategies_excel = set(df_all["strategy"])

strategies_defined = set(strategy_to_family)

undefined = sorted(
    strategies_excel - strategies_defined
)

if undefined:

    raise ValueError(
        "Estas estrategias aparecen en los Excel "
        "pero no están definidas en FAMILIES:\n"
        + "\n".join(undefined)
    )


# Avisar de estrategias definidas que no aparecen en Excel

unused = sorted(
    strategies_defined - strategies_excel
)

if unused:

    print("\nAVISO: Estrategias definidas que no aparecen "
          "en ningún Excel:")

    for method in unused:
        print(" -", method)


# ============================================================
# 6. ASIGNAR FAMILIAS
# ============================================================

df_all["family"] = df_all["strategy"].map(
    strategy_to_family
)

families = list(FAMILIES.keys())


# ============================================================
# 7. CALCULAR MEDIAS POR FAMILIA
# ============================================================

# Media de las métricas de todos los métodos de cada familia,
# calculada por separado para cada modelo de embeddings.

df_families = (
    df_all
    .groupby(
        ["embedding_model", "family"],
        as_index=False
    )[METRICS]
    .mean()
)


# Contar métodos evaluados por familia y modelo

df_counts = (
    df_all
    .groupby(
        ["embedding_model", "family"]
    )["strategy"]
    .nunique()
    .reset_index(name="n_methods")
)

df_families = df_families.merge(
    df_counts,
    on=["embedding_model", "family"],
    how="left"
)


# Mostrar un resumen

print("\n=== RESUMEN ===")
print("Modelos:", len(models))
print("Familias:", len(families))
print("Estrategias:", df_all["strategy"].nunique())


print("\n=== MÉTODOS POR FAMILIA ===")

for family in families:

    methods = FAMILIES[family]

    print(f"\n{family}:")

    for method in methods:
        print("  -", method)


print("\n=== MÉTODOS EVALUADOS POR MODELO ===")

print(
    df_counts.pivot(
        index="family",
        columns="embedding_model",
        values="n_methods"
    ).reindex(
        index=families,
        columns=models
    ).to_string()
)


# Guardar las medias en CSV

csv_path = OUTPUT_DIR / "metrics_by_family.csv"

df_families.to_csv(
    csv_path,
    index=False
)

print("\nGuardado:", csv_path)


# ============================================================
# 8. ASIGNAR COLORES A LAS FAMILIAS
# ============================================================

cmap = plt.get_cmap(LINE_CMAP)

n = len(families)

values = np.concatenate([
    np.linspace(0.00, 0.40, n // 2),
    np.linspace(0.60, 1.00, n - n // 2)
])

colors = {
    family: cmap(value)
    for family, value in zip(families, values)
}


# ============================================================
# 9. GRÁFICOS COMBINADOS POR FAMILIA
# ============================================================

sns.set_theme(style="white")

linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D", "v"]


for metric in METRICS:

    # --------------------------------------------------------
    # 9.1. Preparar datos
    # --------------------------------------------------------

    pivot = df_families.pivot(
        index="family",
        columns="embedding_model",
        values=metric
    )

    pivot = pivot.reindex(
        index=families,
        columns=models
    )

    # Ordenar familias por rendimiento medio descendente

    ordered_families = (
        pivot.mean(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )

    pivot = pivot.loc[ordered_families]


    # --------------------------------------------------------
    # 9.2. Crear figura
    # --------------------------------------------------------

    fig_height = max(7, len(families) * 0.8 + 3)

    fig, (ax_lines, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(16, fig_height),
        gridspec_kw={
            "width_ratios": [1.15, 1],
            "wspace": 0.25
        }
    )

    fig.suptitle(
        metric,
        fontsize=19,
        fontweight="bold",
        y=0.98
    )


    # --------------------------------------------------------
    # 9.3. Gráfico de líneas
    # --------------------------------------------------------

    x = np.arange(len(models))

    for i, family in enumerate(ordered_families):

        y = pivot.loc[family].to_numpy(
            dtype=float
        )

        ax_lines.plot(
            x,
            y,
            color=colors[family],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=6,
            linewidth=1.8,
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=0.5
        )

    ax_lines.set_xticks(x)

    ax_lines.set_xticklabels(
        models,
        fontsize=11
    )

    ax_lines.set_xlabel(
        "Embedding model",
        fontsize=12,
        fontweight="bold"
    )

    ax_lines.set_ylabel(
        metric,
        fontsize=12,
        fontweight="bold"
    )

    ax_lines.set_title(
        "Performance trends by family",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax_lines.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax_lines.set_axisbelow(True)

    ax_lines.spines["top"].set_visible(False)
    ax_lines.spines["right"].set_visible(False)


    # --------------------------------------------------------
    # 9.4. Heatmap
    # --------------------------------------------------------

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=HEATMAP_CMAP,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": metric,
            "shrink": 0.75
        },
        annot_kws={"fontsize": 10},
        ax=ax_heatmap
    )

    ax_heatmap.set_title(
        "Mean performance by chunking family",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax_heatmap.set_xlabel(
        "Embedding model",
        fontsize=12,
        fontweight="bold"
    )

    ax_heatmap.set_ylabel("")

    ax_heatmap.tick_params(
        axis="x",
        rotation=0,
        labelsize=10
    )


    # --------------------------------------------------------
    # 9.5. Etiquetas negras e indicadores de color
    # --------------------------------------------------------

    ax_heatmap.set_yticklabels([])
    ax_heatmap.tick_params(
        axis="y",
        length=0
    )

    x_line_start = -0.30
    x_line_end = -0.25
    x_text = -0.22

    for i, family in enumerate(ordered_families):

        y = i + 0.5

        ax_heatmap.plot(
            [x_line_start, x_line_end],
            [y, y],
            transform=ax_heatmap.get_yaxis_transform(),
            color=colors[family],
            linewidth=4,
            solid_capstyle="round",
            clip_on=False,
            zorder=10
        )

        ax_heatmap.text(
            x_text,
            y,
            family,
            transform=ax_heatmap.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            clip_on=False
        )


    # --------------------------------------------------------
    # 9.6. Ajustes finales
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.09,
        top=0.90,
        wspace=0.25
    )


    # --------------------------------------------------------
    # 9.7. Guardar figura
    # --------------------------------------------------------

    filename = metric.replace("@", "_at_")

    png_path = (
        OUTPUT_DIR / f"{filename}_families_combined.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Guardado:", png_path)


print("\nTodos los gráficos por familia se han generado correctamente.")
