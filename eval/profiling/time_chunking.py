"""Compara tiempos de chunking por técnica y por modelo de embeddings.

Desde la raíz de umi-chunking (guardar este archivo en eval/):
    python eval/plot_chunking_times.py
    python eval/plot_chunking_times.py --format markdown
    python eval/plot_chunking_times.py --model BGE-M3
    python eval/plot_chunking_times.py --per-1000-tokens
    python eval/plot_chunking_times.py --matched-docs --format markdown

Lee los JSON de output/chunking/<modelo>/{json,markdown}/metrics/.
Genera dos gráficos (barras agrupadas y resumen por técnica) y dos CSV.
Cada barra representa la MEDIA de los tiempos de ejecución de documentos,
primero agregados por documento (para no sobreponderar ejecuciones repetidas).
El resumen por técnica da igual peso a cada modelo, no a cada archivo.

NOTA: JSON y Markdown son representaciones diferentes; utilizar --format
markdown para comparar técnicas del mismo formato. Las diferencias entre
modelos incluyen posibles costes distintos de embeddings dentro del chunker.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Apto para Slurm / ejecución sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent  # Si el script está guardado en umi-chunking/eval/

# Nombres de modelo, NO tamaños de chunk.
INPUT_DIRS = {
    "BGE-M3": Path("output/chunking/bge_uro"),
    "Multilingual-E5": Path("output/chunking/e5_uro"),
    # "Jina": Path("output/chunking/jina_uro"),
    "MedCPT": Path("output/chunking/medcpt_uro"),
    "Qwen3": Path("output/chunking/qwen_uro"),
}

OUTPUT_DIR = BASE_DIR / "output/graphics/aggregation/time_chunking"

MODEL_COLORS = {
    "BGE-M3": "#3B7896",
    "Multilingual-E5": "#54A38C",
    # "Jina": "#CF8E54",
    "MedCPT": "#896BAC",
    "Qwen3": "#C15E7B",
}

FAMILIES = {
    "JSON": ["JSONChunking"],
    "Recursive": [
        "parent_child_chunking", "recursive_chunking",
        "recursive_token_fallback_chunking",
    ],
    "Dynamic": [
        "content_density_adaptive_chunking", "dynamic_token_size_chunking",
        "semantic_variance_adaptive_chunking",
    ],
    "LLM": ["llm_boundary_detection_chunking", "llm_segment_then_chunk"],
    "Rulebased": [
        "fixed_character_chunking", "fixed_token_chunking", "length_aware_chunking",
        "overlapping_token_chunking", "paragraph_based_chunking",
        "paragraph_group_chunking", "sentence_based_chunking",
        "sentence_group_chunking", "sliding_window_token_chunking",
    ],
    "Semantic": [
        "semantic_boundary_detection", "semantic_embedding_based_chunking",
        "semantic_similarity_threshold_chunking", "topic_based_chunking",
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

ALIASES = {
    "jsonchunking": "JSONChunking",
    "json_chunking": "JSONChunking",
    "recursive_token_chunking": "recursive_token_fallback_chunking",
    "llm_segment_then_chunking": "llm_segment_then_chunk",
    "semantic_boundary_chunking": "semantic_boundary_detection",
    "semantic_embedding_chunking": "semantic_embedding_based_chunking",
}

STRATEGY_FAMILY = {
    name: family for family, names in FAMILIES.items() for name in names
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("all", "json", "markdown"), default="all",
                        help="Formato documental que se representa (por defecto: all).")
    parser.add_argument("--model", choices=tuple(INPUT_DIRS), default=None,
                        help="Representar solo un modelo; por defecto, los cuatro activos.")
    parser.add_argument("--per-1000-tokens", action="store_true",
                        help="Tiempo por 1.000 tokens originales, en lugar de segundos por documento.")
    parser.add_argument("--matched-docs", action="store_true",
                        help="Comparar solo documentos presentes en todos los modelos seleccionados, "
                             "por formato y técnica. Requiere doc_id en los JSON.")
    parser.add_argument("--scale", choices=("log", "linear"), default="log",
                        help="Escala del eje de tiempo (por defecto: log).")
    parser.add_argument("--input-root", type=Path, default=BASE_DIR,
                        help="Ruta raíz que contiene output/chunking/ (por defecto: raíz del repositorio).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_strategy(value):
    value = str(value).strip()
    return ALIASES.get(value.lower(), value)


def pretty_name(strategy):
    return "JSON Chunking" if strategy == "JSONChunking" else strategy.replace("_", " ").title()


def load_records(args, models):
    records = []
    selected_formats = ("json", "markdown") if args.format == "all" else (args.format,)
    unknown_names = set()
    files_seen = 0

    for model in models:
        for document_format in selected_formats:
            metrics_dir = args.input_root / INPUT_DIRS[model] / document_format / "metrics"
            if not metrics_dir.is_dir():
                print(f"AVISO: no existe la carpeta {metrics_dir}")
                continue
            paths = sorted(metrics_dir.glob("*.json"))
            print(f"{model} | {document_format}: {len(paths)} archivos JSON")
            for path in paths:
                files_seen += 1
                try:
                    with path.open(encoding="utf-8") as handle:
                        data = json.load(handle)
                    if data.get("status") != "completed":
                        continue
                    strategy = normalize_strategy(data["strategy"])
                    if strategy not in STRATEGY_FAMILY:
                        unknown_names.add(strategy)
                        continue  # Comparar solo las técnicas solicitadas de FAMILIES.
                    time_s = float(data["execution_time_seconds"])
                    if not math.isfinite(time_s) or time_s <= 0:
                        continue
                    tokens = pd.to_numeric(data.get("tokens_original"), errors="coerce")
                    if args.per_1000_tokens:
                        if not np.isfinite(tokens) or tokens <= 0:
                            print(f"AVISO: sin tokens_original válidos: {path.name}")
                            continue
                        measurement = time_s / tokens * 1000.0
                    else:
                        measurement = time_s
                    raw_document = data.get("doc_id")
                    if args.matched_docs and (raw_document is None or not str(raw_document).strip()):
                        raise ValueError(f"Falta doc_id necesario para --matched-docs: {path}")
                    # Si falta doc_id, se usa el nombre de archivo para no agrupar
                    # accidentalmente todos los documentos sin identificador.
                    document = str(raw_document) if raw_document is not None else f"FILE:{path.stem}"
                    abbreviation = str(data.get("abbreviation") or strategy).strip()
                    records.append({
                        "model": model,
                        "format": document_format,
                        "strategy": strategy,
                        "family": STRATEGY_FAMILY[strategy],
                        "abbreviation": abbreviation,
                        "document": document,
                        "seconds": time_s,
                        "tokens_original": tokens,
                        "measurement": measurement,
                        "source_file": str(path),
                    })
                except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    if args.matched_docs and isinstance(exc, ValueError) and "Falta doc_id" in str(exc):
                        raise
                    print(f"AVISO: error en {path}: {exc}")

    if unknown_names:
        print("AVISO: se han omitido estrategias no incluidas en FAMILIES:", ", ".join(sorted(unknown_names)))
    if not records:
        raise ValueError(f"No se encontraron tiempos válidos entre {files_seen} JSON de métricas.")
    return pd.DataFrame.from_records(records)


def keep_matched_docs(df, models):
    # Se emparejan por estrategia y formato. No obliga a que todas las
    # estrategias procesen los mismos documentos; se informa de ello abajo.
    masks = []
    n_excluded = 0
    for (_, _), group in df.groupby(["format", "strategy"], sort=False):
        per_model = [set(group.loc[group["model"].eq(m), "document"]) for m in models]
        common = set.intersection(*per_model)
        keep = group["document"].isin(common)
        n_excluded += (~keep).sum()
        masks.append(group.loc[keep])
    print(f"Documentos emparejados: {n_excluded} registros descartados por faltar en otro modelo.")
    if not masks:
        raise ValueError("No hay datos para emparejar.")
    matched = pd.concat(masks, ignore_index=True)
    if matched.empty:
        raise ValueError("No se encontraron documentos comunes. Ejecuta sin --matched-docs o revisa doc_id.")
    return matched


def summarize(df):
    # Una repetición de un mismo documento no debe contar como un documento nuevo.
    per_document = (
        df.groupby(["format", "strategy", "family", "model", "document"], as_index=False)
        .agg(time_value=("measurement", "mean"), n_runs=("measurement", "size"))
    )
    summary = (
        per_document.groupby(["format", "strategy", "family", "model"], as_index=False)
        .agg(mean_time=("time_value", "mean"), median_time=("time_value", "median"),
             std_time=("time_value", "std"), n_documents=("time_value", "size"),
             n_runs=("n_runs", "sum"))
    )
    order_df = (
        summary.groupby(["format", "strategy"], as_index=False)
        .agg(equal_model_mean=("mean_time", "mean"), n_models=("model", "nunique"))
        .sort_values("equal_model_mean")
    )
    order = list(zip(order_df["format"], order_df["strategy"]))
    abbreviations = df.groupby(["format", "strategy"])["abbreviation"].first().to_dict()
    return per_document, summary, order_df, order, abbreviations


def method_label(key, abbreviations, all_formats):
    fmt, strategy = key
    text = abbreviations[key]
    return f"{text} ({'JSON' if fmt == 'json' else 'MD'})" if all_formats else text


def save_grouped_bars(summary, order, abbreviations, models, path, args):
    """Barras agrupadas sin glosario y con la leyenda de familias separada."""
    all_formats = len({fmt for fmt, _ in order}) > 1
    n = len(order)

    # Reservamos la parte inferior para las etiquetas largas en diagonal.
    # La leyenda de familias se sitúa por debajo, de forma independiente.
    fig, ax = plt.subplots(figsize=(max(23, n * 1.04), 11.5))
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.91,
        bottom=0.46,
    )

    values = summary.set_index(["format", "strategy", "model"])["mean_time"]
    x = np.arange(n)
    width = min(0.78 / len(models), 0.5)

    for i, model in enumerate(models):
        yy = np.array(
            [values.get((fmt, strategy, model), np.nan) for fmt, strategy in order],
            dtype=float,
        )
        valid = np.isfinite(yy) & (yy > 0)
        ax.bar(
            x[valid] + (i - (len(models) - 1) / 2) * width,
            yy[valid],
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [method_label(key, abbreviations, all_formats) for key in order],
        rotation=50,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
        fontweight="bold",
    )
    for label, (_, strategy) in zip(ax.get_xticklabels(), order):
        label.set_color(FAMILY_COLORS[STRATEGY_FAMILY[strategy]])

    if args.scale == "log":
        ax.set_yscale("log")
    ax.set_xlim(-0.65, n - 0.4)
    ylabel = (
        "Tiempo medio (s / 1.000 tokens)"
        if args.per_1000_tokens else "Tiempo medio (s / documento)"
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Técnica de chunking", fontsize=11, labelpad=12)
    ax.set_title(
        "Tiempo de ejecución por técnica y modelo de embeddings",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="y", alpha=0.24, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Modelo de embeddings",
        frameon=False,
        ncol=min(len(models), 4),
        loc="upper left",
    )

    # Leyenda de familias colocada debajo de los nombres inclinados.
    present_families = [
        family for family in FAMILIES
        if family in {STRATEGY_FAMILY[strategy] for _, strategy in order}
    ]
    family_handles = [
        Line2D(
            [0], [0],
            color=FAMILY_COLORS[family],
            marker="o",
            linestyle="None",
            label=family,
        )
        for family in present_families
    ]
    fig.legend(
        handles=family_handles,
        title="Familia (color de las etiquetas)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045 if all_formats else 0.035),
        ncol=min(len(family_handles), 6),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        columnspacing=2.0,
    )

    if all_formats:
        fig.text(
            0.5, 0.012,
            "Precaución: JSON y Markdown no son formatos equivalentes; comparar dentro de un mismo formato.",
            ha="center", fontsize=9, color="#555555",
        )

    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_strategy_dotplot(summary, order, models, path, args):
    # Gráfico de lectura rápida: cada punto pequeño es un modelo y el
    # rombo negro es la MEDIA de las medias de los modelos disponibles.
    n = len(order)
    fig, ax = plt.subplots(figsize=(16, max(7.5, 0.51*n + 2)))
    values = summary.set_index(["format", "strategy", "model"])["mean_time"]
    yy = np.arange(n)
    mean_per_strategy = []
    for pos, key in enumerate(order):
        model_times = [values.get((key[0], key[1], m), np.nan) for m in models]
        available = np.array(model_times, dtype=float)
        available = available[np.isfinite(available)]
        mean_per_strategy.append(float(available.mean()) if available.size else np.nan)
        for m, t in zip(models, model_times):
            if np.isfinite(t):
                ax.scatter(t, pos, color=MODEL_COLORS[m], alpha=0.83, s=38, zorder=3)
    ax.scatter(mean_per_strategy, yy, color="#242424", marker="D", s=55,
               label="Media equilibrada entre modelos", zorder=4)
    if args.scale == "log":
        ax.set_xscale("log")
    ax.set_yticks(yy)
    all_formats = len({fmt for fmt, _ in order}) > 1
    names = [f"{pretty_name(strategy)}{' [JSON]' if fmt == 'json' else ' [MD]' if all_formats else ''}"
             for fmt, strategy in order]
    ax.set_yticklabels(names, fontsize=9.5)
    for label, (_, strategy) in zip(ax.get_yticklabels(), order):
        label.set_color(FAMILY_COLORS[STRATEGY_FAMILY[strategy]])
    ax.invert_yaxis()  # Las técnicas más rápidas aparecen arriba.
    xlabel = "Tiempo medio (s / 1.000 tokens)" if args.per_1000_tokens else "Tiempo medio (s / documento)"
    ax.set_xlabel(xlabel)
    ax.set_title("Comparación de tiempos por técnica de chunking", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="None", color=MODEL_COLORS[m], label=m)
               for m in models]
    handles.append(Line2D([0], [0], marker="D", linestyle="None", color="#242424",
                          label="Media entre modelos"))
    ax.legend(handles=handles, frameon=False, loc="best", ncol=min(3, len(handles)))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.92, bottom=0.12)
    if all_formats:
        fig.text(0.5, 0.015, "Precaución: JSON y Markdown utilizan representaciones documentales distintas.",
                 ha="center", fontsize=9, color="#555555")
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    models = [args.model] if args.model else list(INPUT_DIRS)
    df = load_records(args, models)
    if args.matched_docs and len(models) > 1:
        df = keep_matched_docs(df, models)
    per_doc, summary, overall, order, abbreviations = summarize(df)

    # Detectar si los documentos coinciden entre modelos.
    if len(models) > 1 and not args.matched_docs:
        counts = summary.groupby(["format", "strategy"])["n_documents"].agg(["min", "max"])
        mismatch = counts[counts["min"] != counts["max"]]
        if not mismatch.empty:
            print(f"AVISO: {len(mismatch)} técnicas presentan distinto número de documentos por modelo. "
                  "Considera --matched-docs para una comparación controlada.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.format}_{args.model.lower().replace('-', '_') if args.model else 'all_models'}"
    if args.per_1000_tokens:
        tag += "_per_1000_tokens"
    if args.matched_docs:
        tag += "_matched"

    csv_file = args.output_dir / f"chunking_times_summary_{tag}.csv"
    raw_file = args.output_dir / f"chunking_times_by_document_{tag}.csv"
    bars_file = args.output_dir / f"chunking_times_grouped_{tag}.png"
    dot_file = args.output_dir / f"chunking_times_by_strategy_{tag}.png"
    summary.sort_values(["strategy", "model"]).to_csv(csv_file, index=False)
    per_doc.sort_values(["strategy", "model", "document"]).to_csv(raw_file, index=False)
    save_grouped_bars(summary, order, abbreviations, models, bars_file, args)
    save_strategy_dotplot(summary, order, models, dot_file, args)
    print(f"\nRegistros válidos: {len(df)}; técnicas: {df['strategy'].nunique()}; modelos: {df['model'].nunique()}")
    print(f"Medidas por documento: {len(per_doc)}")
    print(f"Gráfico por técnica/modelo: {bars_file}")
    print(f"Gráfico resumen de técnicas: {dot_file}")
    print(f"Resumen CSV: {csv_file}")
    print(f"Detalle por documento: {raw_file}")
    if len(models) > 1 and args.format == "all":
        print("AVISO: para una comparación metodológica estricta, repite con --format markdown o json.")


if __name__ == "__main__":
    main()
