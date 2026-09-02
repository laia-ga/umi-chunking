import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENT_COLORS = {
    "Guideline": "#4472C4",
    "Paper": "#ED7D31",
}

def split_into_sentences(text):
    """
    Divide un texto en frases mediante signos de puntuación.

    Es una separación sencilla que evita depender de NLTK
    o de modelos lingüísticos adicionales.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def identify_document_type(
    doc_id,
    input_file,
):
    """
    Considera guideline cualquier documento cuyo nombre
    contenga la palabra 'guideline'. El resto son papers.
    """

    text = f"{doc_id} {input_file}".lower()

    if "guideline" in text:
        return "Guideline"

    return "Paper"


def prepare_semantic_data(
    semantic_cohesion_df,
    selected_strategies=None,
):
    """
    Añade las columnas method y document_type,
    y permite seleccionar métodos concretos.
    """

    df = semantic_cohesion_df.copy()

    if selected_strategies is not None:
        df = df[
            df["strategy"].isin(
                selected_strategies
            )
        ].copy()

    df["method"] = df["abbreviation"].fillna(
        df["strategy"]
    )

    df["document_type"] = [
        identify_document_type(
            doc_id,
            input_file,
        )
        for doc_id, input_file in zip(
            df["doc_id"],
            df["input_file"],
        )
    ]

    return df

###############################
## SIMILITUD SEMÁNTICA
###############################

def calculate_semantic_cohesion(
    chunks_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    batch_size=32,
):
    """
    Calcula la cohesión semántica interna de cada chunk.

    La cohesión se define como la media de la similitud coseno
    entre el embedding de cada frase y el centroide del chunk.

    Devuelve una tabla con una fila por chunk.

    Los chunks con menos de dos frases reciben cohesión NaN,
    ya que no permiten evaluar realmente la coherencia interna.
    """

    chunks_dir = Path(chunks_dir)

    if not chunks_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {chunks_dir}"
        )

    json_files = sorted(
        chunks_dir.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No se encontraron archivos JSON en: "
            f"{chunks_dir}"
        )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    rows = []

    for json_path in json_files:

        with json_path.open(
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        for chunk_position, chunk in enumerate(
            data.get("chunks", [])
        ):

            text = chunk.get("text", "").strip()

            sentences = split_into_sentences(
                text
            )

            number_of_sentences = len(
                sentences
            )

            if number_of_sentences >= 2:

                embeddings = model.encode(
                    sentences,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                centroid = embeddings.mean(
                    axis=0,
                    keepdims=True,
                )

                sentence_centroid_similarities = (
                    cosine_similarity(
                        embeddings,
                        centroid,
                    )
                    .flatten()
                )

                semantic_cohesion = float(
                    sentence_centroid_similarities.mean()
                )

                minimum_sentence_similarity = float(
                    sentence_centroid_similarities.min()
                )

                cohesion_std = float(
                    sentence_centroid_similarities.std()
                )

            else:
                semantic_cohesion = np.nan
                minimum_sentence_similarity = np.nan
                cohesion_std = np.nan

            rows.append(
                {
                    "strategy": data.get("strategy"),
                    "type": data.get("type"),
                    "abbreviation": data.get(
                        "abbreviation"
                    ),
                    "doc_id": data.get("doc_id"),
                    "input_file": data.get(
                        "input_file"
                    ),
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "chunk_position": chunk_position,
                    "number_of_characters": len(
                        text
                    ),
                    "number_of_sentences": (
                        number_of_sentences
                    ),
                    "semantic_cohesion": (
                        semantic_cohesion
                    ),
                    "minimum_sentence_similarity": (
                        minimum_sentence_similarity
                    ),
                    "semantic_cohesion_std": (
                        cohesion_std
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_semantic_cohesion(
    semantic_cohesion_df,
    doc_ids,
):
    """
    Resume la cohesión semántica por método para los
    documentos indicados y genera una gráfica ordenada.

    Los chunks con una sola frase, cuya cohesión es NaN,
    no se incluyen en la media.
    """

    filtered_df = semantic_cohesion_df[
        semantic_cohesion_df["doc_id"].isin(
            doc_ids
        )
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            "No se encontraron filas para los "
            "documentos indicados."
        )

    summary_df = (
        filtered_df
        .groupby(
            [
                "strategy",
                "abbreviation",
            ],
            as_index=False,
        )
        .agg(
            mean_semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            ),
            median_semantic_cohesion=(
                "semantic_cohesion",
                "median",
            ),
            minimum_semantic_cohesion=(
                "semantic_cohesion",
                "min",
            ),
            evaluated_chunks=(
                "semantic_cohesion",
                "count",
            ),
            total_chunks=(
                "chunk_id",
                "count",
            ),
        )
    )

    summary_df[
        "single_sentence_chunks"
    ] = (
        summary_df["total_chunks"]
        - summary_df["evaluated_chunks"]
    )

    summary_df[
        "evaluated_chunks_percentage"
    ] = (
        summary_df["evaluated_chunks"]
        / summary_df["total_chunks"]
        * 100
    ).round(2)

    summary_df = (
        summary_df
        .sort_values(
            by="mean_semantic_cohesion",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    numeric_columns = [
        "mean_semantic_cohesion",
        "median_semantic_cohesion",
        "minimum_semantic_cohesion",
    ]

    summary_df[numeric_columns] = (
        summary_df[numeric_columns].round(4)
    )

    plt.figure(
        figsize=(
            10,
            max(
                5,
                len(summary_df) * 0.4,
            ),
        )
    )

    plt.barh(
        summary_df["abbreviation"],
        summary_df[
            "mean_semantic_cohesion"
        ],
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Cohesión semántica media"
    )

    plt.ylabel(
        "Método"
    )

    plt.title(
        "Cohesión semántica interna media "
        "de los chunks"
    )

    plt.xlim(0, 1)

    for index, value in enumerate(
        summary_df[
            "mean_semantic_cohesion"
        ]
    ):
        plt.text(
            value + 0.005,
            index,
            f"{value:.3f}",
            va="center",
        )

    plt.tight_layout()
    plt.show()

    return summary_df

##########################
## REPRESENTACIÓN
##########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_cohesion_by_method_and_document_type(
    semantic_cohesion_df: pd.DataFrame,
    *,
    selected_strategies: list[str] | None = None,
    method_order: list[str] | None = None,
    save_path: str | None = None,
):
    """
    Representa la cohesión semántica agrupada por método
    y tipo de documento.

    Barras:
        Cohesión semántica media.

    Línea:
        Media de la similitud mínima de las frases.

    La agregación se realiza en dos pasos:
    1. Media de los chunks de cada documento.
    2. Media entre documentos de la misma tipología.
    """

    df = semantic_cohesion_df.copy()

    required_columns = {
        "strategy",
        "abbreviation",
        "doc_id",
        "input_file",
        "chunk_id",
        "semantic_cohesion",
        "minimum_sentence_similarity",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Faltan las siguientes columnas: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Selección de métodos
    # ---------------------------------------------------------

    if selected_strategies is not None:
        df = df[
            df["strategy"].isin(selected_strategies)
        ].copy()

    if df.empty:
        raise ValueError(
            "No hay datos para las estrategias seleccionadas."
        )

    # Utiliza la abreviatura como nombre visible.
    df["method"] = df["abbreviation"].fillna(
        df["strategy"]
    )

    # ---------------------------------------------------------
    # Identificación del tipo de documento
    # ---------------------------------------------------------

    document_text = (
        df["doc_id"].fillna("").astype(str)
        + " "
        + df["input_file"].fillna("").astype(str)
    )

    # Como solamente existen guidelines y papers,
    # todo lo que no contiene "guideline" se considera paper.
    df["document_type"] = np.where(
        document_text.str.contains(
            "guideline",
            case=False,
            regex=False,
        ),
        "Guideline",
        "Paper",
    )

    # ---------------------------------------------------------
    # Primera agregación: media por documento
    # ---------------------------------------------------------

    document_scores = (
        df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
                "doc_id",
            ],
            as_index=False,
        )
        .agg(
            semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            ),
            minimum_sentence_similarity=(
                "minimum_sentence_similarity",
                "mean",
            ),
            evaluated_chunks=(
                "semantic_cohesion",
                "count",
            ),
        )
    )

    # Elimina documentos sin chunks evaluables.
    document_scores = document_scores.dropna(
        subset=[
            "semantic_cohesion",
            "minimum_sentence_similarity",
        ],
        how="all",
    )

    # ---------------------------------------------------------
    # Segunda agregación: media entre documentos
    # ---------------------------------------------------------

    summary_df = (
        document_scores
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            ),
            std_semantic_cohesion=(
                "semantic_cohesion",
                "std",
            ),
            mean_minimum_similarity=(
                "minimum_sentence_similarity",
                "mean",
            ),
            std_minimum_similarity=(
                "minimum_sentence_similarity",
                "std",
            ),
            number_of_documents=(
                "doc_id",
                "nunique",
            ),
        )
    )

    # Error estándar entre documentos.
    summary_df["se_semantic_cohesion"] = (
        summary_df["std_semantic_cohesion"]
        / np.sqrt(summary_df["number_of_documents"])
    ).fillna(0)

    summary_df["se_minimum_similarity"] = (
        summary_df["std_minimum_similarity"]
        / np.sqrt(summary_df["number_of_documents"])
    ).fillna(0)

    # ---------------------------------------------------------
    # Orden de los métodos
    # ---------------------------------------------------------

    if method_order is None:
        method_order = (
            document_scores
            .groupby("method")["semantic_cohesion"]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    document_types = [
        "Guideline",
        "Paper",
    ]

    document_colors = {
        "Guideline": "#4472C4",
        "Paper": "#ED7D31",
    }

    # ---------------------------------------------------------
    # Gráfica
    # ---------------------------------------------------------

    figure_width = max(
        13,
        len(method_order) * 2.2,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(figure_width, 6),
        sharey=True,
    )

    x_positions = np.arange(
        len(method_order)
    )

    for ax, document_type in zip(
        axes,
        document_types,
    ):
        type_data = (
            summary_df[
                summary_df["document_type"]
                == document_type
            ]
            .set_index("method")
            .reindex(method_order)
        )

        bar_values = type_data[
            "mean_semantic_cohesion"
        ].to_numpy()

        bar_errors = type_data[
            "se_semantic_cohesion"
        ].fillna(0).to_numpy()

        line_values = type_data[
            "mean_minimum_similarity"
        ].to_numpy()

        line_errors = type_data[
            "se_minimum_similarity"
        ].fillna(0).to_numpy()

        bars = ax.bar(
            x_positions,
            bar_values,
            yerr=bar_errors,
            capsize=4,
            width=0.65,
            color=document_colors[document_type],
            alpha=0.8,
            edgecolor="white",
        )

        ax.errorbar(
            x_positions,
            line_values,
            yerr=line_errors,
            color="#222222",
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            zorder=5,
        )

        # Valor de cohesión encima de cada barra.
        for bar, value in zip(
            bars,
            bar_values,
        ):
            if not np.isnan(value):
                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(
            document_type,
            color=document_colors[document_type],
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xticks(
            x_positions,
            method_order,
            rotation=30,
            ha="right",
        )

        ax.set_xlabel(
            "Método de chunking"
        )

        ax.set_ylim(0, 1.05)

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Similitud coseno"
    )

    fig.suptitle(
        "Cohesión semántica por método y tipo de documento",
        fontsize=15,
        fontweight="bold",
    )

    legend_elements = [
        Patch(
            facecolor="#888888",
            alpha=0.8,
            label="Cohesión semántica media",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            marker="o",
            linewidth=2,
            label=(
                "Media de la similitud mínima "
                "de cada chunk"
            ),
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=2,
        frameon=False,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.88]
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return document_scores, summary_df, fig


#############################
## 1. COHESIÓN FRENTE A TAMAÑO
#############################

def plot_cohesion_vs_size(
    semantic_cohesion_df,
    selected_strategies=None,
):
    df = prepare_semantic_data(
        semantic_cohesion_df,
        selected_strategies,
    )

    # Primero se calcula un resultado por documento.
    document_df = (
        df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
                "doc_id",
            ],
            as_index=False,
        )
        .agg(
            semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            ),
            median_chunk_characters=(
                "number_of_characters",
                "median",
            ),
            number_of_chunks=(
                "chunk_id",
                "count",
            ),
        )
        .dropna(
            subset=["semantic_cohesion"]
        )
    )

    # Después se calcula la media entre documentos.
    summary_df = (
        document_df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            ),
            mean_median_chunk_characters=(
                "median_chunk_characters",
                "mean",
            ),
            mean_number_of_chunks=(
                "number_of_chunks",
                "mean",
            ),
        )
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=summary_df,
        x="mean_median_chunk_characters",
        y="mean_semantic_cohesion",
        hue="document_type",
        palette=DOCUMENT_COLORS,
        s=120,
    )

    # Añade la abreviatura junto a cada punto.
    for _, row in summary_df.iterrows():
        plt.annotate(
            row["method"],
            (
                row[
                    "mean_median_chunk_characters"
                ],
                row["mean_semantic_cohesion"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel(
        "Mediana del tamaño del chunk "
        "(número de caracteres)"
    )
    plt.ylabel(
        "Cohesión semántica media"
    )
    plt.title(
        "Relación entre tamaño y cohesión semántica"
    )
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    return document_df, summary_df

####################################
## 2. DISTRIBUCIÓN DE LA COHESIÓN
####################################

def plot_cohesion_distribution(
    semantic_cohesion_df,
    selected_strategies=None,
):
    df = prepare_semantic_data(
        semantic_cohesion_df,
        selected_strategies,
    )

    document_df = (
        df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
                "doc_id",
            ],
            as_index=False,
        )
        .agg(
            semantic_cohesion=(
                "semantic_cohesion",
                "mean",
            )
        )
        .dropna(
            subset=["semantic_cohesion"]
        )
    )

    method_order = (
        document_df
        .groupby("method")[
            "semantic_cohesion"
        ]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    plt.figure(
        figsize=(
            max(10, len(method_order) * 1.5),
            6,
        )
    )

    sns.boxplot(
        data=document_df,
        x="method",
        y="semantic_cohesion",
        hue="document_type",
        order=method_order,
        hue_order=["Guideline", "Paper"],
        palette=DOCUMENT_COLORS,
        showfliers=False,
    )

    # Muestra también los documentos individuales.
    sns.stripplot(
        data=document_df,
        x="method",
        y="semantic_cohesion",
        hue="document_type",
        order=method_order,
        hue_order=["Guideline", "Paper"],
        palette=DOCUMENT_COLORS,
        dodge=True,
        jitter=0.12,
        size=6,
        edgecolor="white",
        linewidth=0.5,
    )

    # Evita que la leyenda aparezca duplicada.
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(
        handles[:2],
        labels[:2],
        title="Tipo de documento",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.xlabel("Método de chunking")
    plt.ylabel("Cohesión semántica media")
    plt.title(
        "Distribución de la cohesión semántica "
        "entre documentos"
    )
    plt.ylim(0, 1.02)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return document_df

######################################
## HOMOGENEIDAD INTERNA
######################################

def plot_internal_homogeneity(
    semantic_cohesion_df,
    selected_strategies=None,
):
    df = prepare_semantic_data(
        semantic_cohesion_df,
        selected_strategies,
    )

    # Media de las desviaciones de los chunks de cada documento.
    document_df = (
        df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
                "doc_id",
            ],
            as_index=False,
        )
        .agg(
            mean_internal_variability=(
                "semantic_cohesion_std",
                "mean",
            )
        )
        .dropna(
            subset=[
                "mean_internal_variability"
            ]
        )
    )

    method_order = (
        document_df
        .groupby("method")[
            "mean_internal_variability"
        ]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    plt.figure(
        figsize=(
            max(10, len(method_order) * 1.5),
            6,
        )
    )

    sns.barplot(
        data=document_df,
        x="method",
        y="mean_internal_variability",
        hue="document_type",
        order=method_order,
        hue_order=["Guideline", "Paper"],
        palette=DOCUMENT_COLORS,
        errorbar="se",
        capsize=0.12,
    )

    plt.xlabel("Método de chunking")
    plt.ylabel(
        "Desviación interna media"
    )
    plt.title(
        "Homogeneidad interna de los chunks\n"
        "Un valor menor indica mayor homogeneidad"
    )
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)

    plt.legend(
        title="Tipo de documento",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()

    return document_df

######################################
## 4. CLARIDAD DE LAS FRONTERAS
######################################

def calculate_and_plot_boundary_clarity(
    chunks_dir,
    semantic_cohesion_df,
    *,
    selected_strategies=None,
    model=None,
    model_name="BAAI/bge-m3",
    k=2,
    batch_size=32,
):
    chunks_dir = Path(chunks_dir)

    metrics_df = prepare_semantic_data(
        semantic_cohesion_df,
        selected_strategies,
    )

    # Cohesión de cada chunk, ya calculada anteriormente.
    cohesion_lookup = (
        metrics_df
        .groupby(
            [
                "strategy",
                "doc_id",
                "chunk_id",
            ]
        )["semantic_cohesion"]
        .mean()
        .to_dict()
    )

    method_lookup = (
        metrics_df[
            ["strategy", "method"]
        ]
        .drop_duplicates("strategy")
        .set_index("strategy")["method"]
        .to_dict()
    )

    boundary_specs = []
    boundary_contexts = []

    for json_path in chunks_dir.rglob("*.json"):
        try:
            with json_path.open(
                encoding="utf-8"
            ) as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue

        chunks = data.get("chunks")
        strategy = data.get("strategy")

        if not isinstance(chunks, list):
            continue

        if strategy is None:
            continue

        if (
            selected_strategies is not None
            and strategy
            not in selected_strategies
        ):
            continue

        doc_id = data.get(
            "doc_id",
            json_path.stem,
        )

        input_file = data.get(
            "input_file",
            "",
        )

        document_type = identify_document_type(
            doc_id,
            input_file,
        )

        method = method_lookup.get(
            strategy,
            data.get("abbreviation", strategy),
        )

        for position in range(
            len(chunks) - 1
        ):
            previous_chunk = chunks[position]
            next_chunk = chunks[position + 1]

            previous_id = previous_chunk.get(
                "chunk_id"
            )
            next_id = next_chunk.get(
                "chunk_id"
            )

            previous_cohesion = (
                cohesion_lookup.get(
                    (
                        strategy,
                        doc_id,
                        previous_id,
                    ),
                    np.nan,
                )
            )

            next_cohesion = (
                cohesion_lookup.get(
                    (
                        strategy,
                        doc_id,
                        next_id,
                    ),
                    np.nan,
                )
            )

            # Si no se pudo calcular la cohesión interna
            # de alguno de los chunks, se omite la frontera.
            if (
                pd.isna(previous_cohesion)
                or pd.isna(next_cohesion)
            ):
                continue

            previous_sentences = (
                split_into_sentences(
                    previous_chunk.get(
                        "text",
                        "",
                    )
                )
            )

            next_sentences = (
                split_into_sentences(
                    next_chunk.get(
                        "text",
                        "",
                    )
                )
            )

            if (
                not previous_sentences
                or not next_sentences
            ):
                continue

            previous_context = " ".join(
                previous_sentences[-k:]
            )

            next_context = " ".join(
                next_sentences[:k]
            )

            boundary_contexts.extend(
                [
                    previous_context,
                    next_context,
                ]
            )

            boundary_specs.append(
                {
                    "strategy": strategy,
                    "method": method,
                    "doc_id": doc_id,
                    "document_type": document_type,
                    "previous_chunk_id": previous_id,
                    "next_chunk_id": next_id,
                    "previous_cohesion": (
                        previous_cohesion
                    ),
                    "next_cohesion": (
                        next_cohesion
                    ),
                }
            )

    if not boundary_specs:
        raise ValueError(
            "No se encontraron fronteras evaluables."
        )

    if model is None:
        model = SentenceTransformer(
            model_name
        )

    context_embeddings = model.encode(
        boundary_contexts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    rows = []

    for index, boundary in enumerate(
        boundary_specs
    ):
        previous_embedding = (
            context_embeddings[index * 2]
        )

        next_embedding = (
            context_embeddings[index * 2 + 1]
        )

        boundary_similarity = float(
            np.dot(
                previous_embedding,
                next_embedding,
            )
        )

        mean_internal_cohesion = (
            boundary["previous_cohesion"]
            + boundary["next_cohesion"]
        ) / 2

        boundary_clarity = (
            mean_internal_cohesion
            - boundary_similarity
        )

        rows.append(
            {
                **boundary,
                "boundary_similarity": (
                    boundary_similarity
                ),
                "mean_internal_cohesion": (
                    mean_internal_cohesion
                ),
                "boundary_clarity": (
                    boundary_clarity
                ),
            }
        )

    boundary_df = pd.DataFrame(rows)

    # Una media por documento.
    document_boundary_df = (
        boundary_df
        .groupby(
            [
                "document_type",
                "strategy",
                "method",
                "doc_id",
            ],
            as_index=False,
        )
        .agg(
            mean_boundary_clarity=(
                "boundary_clarity",
                "mean",
            ),
            mean_boundary_similarity=(
                "boundary_similarity",
                "mean",
            ),
            evaluated_boundaries=(
                "boundary_clarity",
                "count",
            ),
        )
    )

    method_order = (
        document_boundary_df
        .groupby("method")[
            "mean_boundary_clarity"
        ]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    plt.figure(
        figsize=(
            max(10, len(method_order) * 1.5),
            6,
        )
    )

    sns.barplot(
        data=document_boundary_df,
        x="method",
        y="mean_boundary_clarity",
        hue="document_type",
        order=method_order,
        hue_order=["Guideline", "Paper"],
        palette=DOCUMENT_COLORS,
        errorbar="se",
        capsize=0.12,
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
    )

    plt.xlabel("Método de chunking")
    plt.ylabel("Claridad media de las fronteras")
    plt.title(
        "Claridad semántica de las fronteras\n"
        "Un valor mayor indica una separación más clara"
    )
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)

    plt.legend(
        title="Tipo de documento",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()

    return boundary_df, document_boundary_df