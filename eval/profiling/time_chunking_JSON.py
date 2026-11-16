
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

INPUT_FILES = {
    "BGE-M3": Path("output/chunks/bge_JSON/json/summary.csv"),
    "Qwen3": Path("output/chunks/qwen_JSON/json/summary.csv"),
}

OUTPUT_DIR = Path("output/graphics/json_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "BGE-M3": "#348F98",
    "Qwen3": "#244A70",
}

MODELS = list(INPUT_FILES)

METRICS = {
    "number_of_chunks": "Chunks por documento",
    "execution_time_seconds": "Tiempo por documento (s)",
    "tokens_mean": "Tokens medios por chunk",
    "tokens_ratio": "Ratio de tokens",
    "ram_increase_mb": "Incremento de RAM (MB)",
    "empty_chunks": "Chunks vacíos",
}

NUMERIC_COLUMNS = [
    "number_of_chunks",
    "empty_chunks",
    "tokens_min",
    "tokens_p25",
    "tokens_median",
    "tokens_mean",
    "tokens_p75",
    "tokens_max",
    "tokens_std",
    "tokens_total",
    "tokens_original",
    "tokens_ratio",
    "execution_time_seconds",
    "ram_before_mb",
    "ram_after_mb",
    "ram_increase_mb",
]


# ============================================================
# 2. CARGAR DATOS
# ============================================================

frames = []

for model, path in INPUT_FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df_model = pd.read_csv(path)

    required = [
        "status",
        "doc_id",
        "number_of_chunks",
        "execution_time_seconds",
    ]

    missing = [
        col for col in required
        if col not in df_model.columns
    ]

    if missing:
        raise ValueError(
            f"{model}: faltan columnas {missing}"
        )

    df_model = df_model[
        df_model["status"] == "completed"
    ].copy()

    for col in NUMERIC_COLUMNS:
        if col not in df_model.columns:
            df_model[col] = np.nan

        df_model[col] = pd.to_numeric(
            df_model[col],
            errors="coerce"
        )

        df_model[col] = df_model[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

    df_model["model"] = model

    # Normalización por tamaño del documento.
    df_model["chunks_per_1000_tokens"] = (
        1000 * df_model["number_of_chunks"]
        / df_model["tokens_original"].replace(0, np.nan)
    )

    df_model["tokens_per_second"] = (
        df_model["tokens_original"]
        / df_model["execution_time_seconds"].replace(0, np.nan)
    )

    frames.append(df_model)

df = pd.concat(frames, ignore_index=True)

print("\nDocumentos procesados:")
print(df.groupby("model").size())


# ============================================================
# 3. RESUMEN ESTADÍSTICO
# ============================================================

summary_rows = []

for model, group in df.groupby("model"):
    row = {
        "model": model,
        "n_documents": len(group),
        "total_chunks": group["number_of_chunks"].sum(),
        "total_empty_chunks": group["empty_chunks"].sum(),
        "total_execution_seconds": (
            group["execution_time_seconds"].sum()
        ),
    }

    for col in [
        *METRICS,
        "chunks_per_1000_tokens",
        "tokens_per_second",
        "tokens_original",
        "tokens_total",
        "tokens_median",
        "tokens_p25",
        "tokens_p75",
        "tokens_max",
        "tokens_std",
        "ram_before_mb",
        "ram_after_mb",
    ]:
        row[f"{col}_mean"] = group[col].mean()
        row[f"{col}_median"] = group[col].median()
        row[f"{col}_std"] = group[col].std()
        row[f"{col}_min"] = group[col].min()
        row[f"{col}_max"] = group[col].max()

    row["documents_per_minute_aggregate"] = (
        60 * len(group)
        / row["total_execution_seconds"]
        if row["total_execution_seconds"] > 0
        else np.nan
    )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "json_comparison_summary.csv",
    index=False
)

df.to_csv(
    OUTPUT_DIR / "json_comparison_documents.csv",
    index=False
)


# ============================================================
# 4. FUNCIÓN PARA GRÁFICAS
# ============================================================

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()


def bar_comparison(metrics, filename, title):
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(5 * len(metrics), 5)
    )

    axes = np.atleast_1d(axes)

    for ax, (column, label) in zip(axes, metrics.items()):
        values = [
            df.loc[df["model"] == model, column].mean()
            for model in MODELS
        ]

        bars = ax.bar(
            MODELS,
            values,
            color=[COLORS[m] for m in MODELS],
            width=0.55
        )

        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:,.2f}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height()
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10
                )

    fig.suptitle(title, fontsize=15, fontweight="bold")
    save_figure(filename)


# ============================================================
# 5. GRÁFICA DE MÉTRICAS PRINCIPALES
# ============================================================

bar_comparison(
    {
        "number_of_chunks": "Chunks / documento",
        "execution_time_seconds": "Tiempo medio (s)",
        "tokens_mean": "Tokens / chunk",
    },
    "01_main_metrics.png",
    "Comparación de JSON Chunking"
)


# ============================================================
# 6. FRAGMENTACIÓN Y OVERHEAD
# ============================================================

bar_comparison(
    {
        "chunks_per_1000_tokens": "Chunks / 1.000 tokens",
        "tokens_ratio": "Ratio de tokens",
        "empty_chunks": "Chunks vacíos / documento",
    },
    "02_fragmentation.png",
    "Fragmentación y redundancia"
)


# ============================================================
# 7. RENDIMIENTO Y MEMORIA
# ============================================================

bar_comparison(
    {
        "tokens_per_second": "Tokens originales / segundo",
        "ram_increase_mb": "Incremento medio de RAM (MB)",
    },
    "03_performance.png",
    "Rendimiento y memoria"
)


# ============================================================
# 8. DISTRIBUCIONES POR DOCUMENTO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, column, title in zip(
    axes,
    ["number_of_chunks", "execution_time_seconds"],
    ["Chunks por documento", "Tiempo por documento (s)"]
):
    data = [
        df.loc[df["model"] == model, column]
        .dropna()
        .to_numpy()
        for model in MODELS
    ]

    bp = ax.boxplot(
        data,
        tick_labels=MODELS,
        patch_artist=True,
        showfliers=False
    )

    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "Variabilidad entre documentos",
    fontsize=15,
    fontweight="bold"
)

save_figure("04_distributions.png")


# ============================================================
# 9. COMPARACIÓN EMPAREJADA POR DOCUMENTO
# ============================================================

# Evitar emparejamientos ambiguos si hay doc_id duplicados.
duplicates = df.duplicated(
    ["model", "doc_id"],
    keep=False
)

if duplicates.any():
    print(
        "\nATENCIÓN: hay doc_id duplicados. "
        "Se excluyen de la comparación emparejada."
    )

    duplicated_ids = set(
        df.loc[duplicates, "doc_id"].dropna()
    )
    paired_source = df[
        ~df["doc_id"].isin(duplicated_ids)
    ].copy()
else:
    paired_source = df.copy()

paired_metrics = [
    "number_of_chunks",
    "execution_time_seconds",
    "tokens_mean",
    "tokens_ratio",
    "ram_increase_mb",
]

paired = paired_source.pivot(
    index="doc_id",
    columns="model",
    values=paired_metrics
)

# Conservar únicamente documentos presentes en ambos modelos.
paired = paired.dropna(
    subset=[
        ("number_of_chunks", MODELS[0]),
        ("number_of_chunks", MODELS[1]),
    ]
)

paired.columns = [
    f"{metric}_{model}"
    for metric, model in paired.columns
]

for metric in paired_metrics:
    paired[f"diff_{metric}"] = (
        paired[f"{metric}_{MODELS[1]}"]
        - paired[f"{metric}_{MODELS[0]}"]
    )

paired.to_csv(
    OUTPUT_DIR / "json_paired_documents.csv"
)

print(f"\nDocumentos comunes: {len(paired)}")

if not paired.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(
        axes,
        ["number_of_chunks", "execution_time_seconds"],
        ["Diferencia de chunks", "Diferencia de tiempo (s)"]
    ):
        differences = paired[
            f"diff_{metric}"
        ].dropna()

        ax.hist(
            differences,
            bins=25,
            color=COLORS[MODELS[1]],
            edgecolor="white"
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--"
        )

        ax.set_title(title)
        ax.set_xlabel(
            f"{MODELS[1]} − {MODELS[0]}"
        )
        ax.set_ylabel("Documentos")

    fig.suptitle(
        "Comparación de documentos comunes",
        fontsize=15,
        fontweight="bold"
    )

    save_figure("05_paired_comparison.png")


# ============================================================
# 10. RESUMEN EN TERMINAL
# ============================================================

display_columns = [
    "model",
    "n_documents",
    "total_chunks",
    "number_of_chunks_mean",
    "execution_time_seconds_mean",
    "total_execution_seconds",
    "tokens_mean_mean",
    "tokens_ratio_mean",
    "ram_increase_mb_mean",
]

print("\nRESUMEN COMPARATIVO")
print(
    summary[display_columns]
    .round(3)
    .to_string(index=False)
)

if not paired.empty:
    print("\nDIFERENCIAS MEDIAS EN DOCUMENTOS COMUNES")

    for metric in paired_metrics:
        difference = paired[f"diff_{metric}"]
        print(
            f"{metric}: "
            f"{difference.mean():+.3f}"
        )

print(f"\nResultados guardados en: {OUTPUT_DIR}")
