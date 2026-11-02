
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

INPUT_DIR = Path("output/metrics")
OUTPUT_DIR = Path("output/graphics/aggregation")
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

# Valores bajos: rojo
# Valores altos: azul
HEATMAP_CMAP = "Spectral"

# Paleta para identificar las estrategias
LINE_CMAP = "Spectral_r"

# Mantener la misma escala para todas las métricas
# (nDCG, HitRate y MRR están entre 0 y 1)
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1


# ============================================================
# 2. CARGAR DATOS
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

strategies = sorted(
    df_all["strategy"].unique()
)


# ============================================================
# 3. VALIDACIONES
# ============================================================

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


for model in models:

    model_strategies = set(
        df_all.loc[
            df_all["embedding_model"] == model,
            "strategy"
        ]
    )

    missing_strategies = (
        set(strategies) - model_strategies
    )

    if missing_strategies:

        print(
            f"AVISO: {model} no contiene estas estrategias: "
            f"{sorted(missing_strategies)}"
        )


print(f"Modelos: {len(models)}")
print(f"Estrategias: {len(strategies)}")
print(f"Métricas: {len(METRICS)}")


# ============================================================
# 4. ASIGNAR COLORES A LAS ESTRATEGIAS
# ============================================================

cmap = plt.get_cmap(LINE_CMAP)

n = len(strategies)

# Usar los extremos de Spectral:
# se evitan los tonos amarillos centrales más claros.
values = np.concatenate([
    np.linspace(0.00, 0.40, n // 2),
    np.linspace(0.60, 1.00, n - n // 2)
])

colors = {
    strategy: cmap(value)
    for strategy, value in zip(strategies, values)
}


# ============================================================
# 5. GRÁFICOS COMBINADOS
# ============================================================

sns.set_theme(style="white")

linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D", "v"]

for metric in METRICS:

    # --------------------------------------------------------
    # 5.1. Preparar datos
    # --------------------------------------------------------

    pivot = df_all.pivot(
        index="strategy",
        columns="embedding_model",
        values=metric
    )

    pivot = pivot.reindex(columns=models)

    # Ordenar por rendimiento medio descendente
    ordered_strategies = (
        pivot.mean(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )

    pivot = pivot.loc[ordered_strategies]

    # --------------------------------------------------------
    # 5.2. Crear figura
    # --------------------------------------------------------

    fig, (ax_lines, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(22, 11),
        gridspec_kw={
            "width_ratios": [1.15, 1],
            "wspace": 0.48
        }
    )

    fig.suptitle(
        metric,
        fontsize=19,
        fontweight="bold",
        y=0.98
    )

    # --------------------------------------------------------
    # 5.3. Gráfico de líneas
    # --------------------------------------------------------

    x = np.arange(len(models))

    for i, strategy in enumerate(ordered_strategies):

        y = pivot.loc[strategy].to_numpy(
            dtype=float
        )

        ax_lines.plot(
            x,
            y,
            color=colors[strategy],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5,
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
        "Performance trends",
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
    # 5.4. Heatmap
    # --------------------------------------------------------

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Spectral",  # Rojo = bajo, azul = alto
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": metric,
            "shrink": 0.75
        },
        annot_kws={"fontsize": 9},
        ax=ax_heatmap
    )

    ax_heatmap.set_title(
        "Performance by chunking strategy",
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
    # 5.5. Etiquetas negras + indicadores alineados
    # --------------------------------------------------------

    # Ocultar etiquetas originales
    ax_heatmap.set_yticklabels([])
    ax_heatmap.tick_params(axis="y", length=0)

    # Posiciones fijas respecto al borde izquierdo del heatmap
    x_line_start = -0.58
    x_line_end = -0.53
    x_text = -0.50

    for i, strategy in enumerate(ordered_strategies):

        y = i + 0.5

        # Indicador de color (misma posición para todos)
        ax_heatmap.plot(
            [x_line_start, x_line_end],
            [y, y],
            transform=ax_heatmap.get_yaxis_transform(),
            color=colors[strategy],
            linewidth=4,
            solid_capstyle="round",
            clip_on=False,
            zorder=10
        )

        # Nombre de la estrategia
        ax_heatmap.text(
            x_text,
            y,
            strategy,
            transform=ax_heatmap.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False
        )

    # --------------------------------------------------------
    # 5.6. Ajustes finales
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.09,
        top=0.90,
        wspace=0.48
    )

    # --------------------------------------------------------
    # 5.7. Guardar gráficos
    # --------------------------------------------------------

    filename = metric.replace("@", "_at_")

    png_path = (
        OUTPUT_DIR / f"{filename}_combined.png"
    )


    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Guardado: {png_path}")

print("Todos los gráficos se han generado correctamente.")
