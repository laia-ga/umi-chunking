import json
import re
import unicodedata

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$"
)


DOCUMENT_COLORS = {
    "Guideline": "#4472C4",
    "Paper": "#ED7D31",
    "Ficha técnica": "#70AD47",
}


def normalize_text(text):
    """
    Normaliza el texto para comparar contenido aunque cambien
    los saltos de línea o los espacios.
    """

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def identify_document_type(
    doc_id,
    input_file,
):
    text = normalize_text(
        f"{doc_id} {input_file}"
    )

    if "guideline" in text:
        return "Guideline"

    if (
        "ficha tecnica" in text
        or "ficha técnica" in text
        or "smpc" in text
    ):
        return "Ficha técnica"

    return "Paper"


def extract_markdown_headings(
    markdown_text,
    probe_words=12,
):
    """
    Extrae los encabezados Markdown y busca el primer párrafo,
    elemento de lista o fila de tabla asociado a cada título.

    Para encabezados padre, permite atravesar encabezados hijos
    hasta encontrar el primer contenido de su sección.
    """

    lines = markdown_text.splitlines()

    headings = []

    # Localiza todos los encabezados.
    for line_index, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)

        if match:
            headings.append(
                {
                    "line_index": line_index,
                    "level": len(match.group(1)),
                    "heading": match.group(2).strip(),
                }
            )

    results = []

    for heading_position, heading in enumerate(
        headings
    ):
        start = heading["line_index"] + 1
        end = len(lines)

        # La sección termina en el siguiente encabezado
        # del mismo nivel o de un nivel superior.
        for next_heading in headings[
            heading_position + 1:
        ]:
            if (
                next_heading["level"]
                <= heading["level"]
            ):
                end = next_heading["line_index"]
                break

        description_lines = []
        content_started = False

        for line in lines[start:end]:
            stripped_line = line.strip()

            # Los encabezados hijos no son la descripción.
            if HEADING_PATTERN.match(line):
                continue

            if not stripped_line:
                if content_started:
                    break

                continue

            description_lines.append(
                stripped_line
            )
            content_started = True

        description = " ".join(
            description_lines
        )

        description_words = normalize_text(
            description
        ).split()

        description_probe = " ".join(
            description_words[:probe_words]
        )

        results.append(
            {
                **heading,
                "description": description,
                "description_probe": (
                    description_probe
                ),
                "has_description": bool(
                    description_probe
                ),
            }
        )

    return results


def find_heading_suffixes(
    chunk_text,
    heading,
):
    """
    Busca un título como línea independiente dentro de un chunk.

    Devuelve el texto que aparece después de cada aparición
    del título.
    """

    chunk_lines = chunk_text.splitlines()
    normalized_heading = normalize_text(
        heading
    )

    suffixes = []

    for line_index, line in enumerate(
        chunk_lines
    ):
        match = HEADING_PATTERN.match(line)

        if match:
            candidate = match.group(2)
        else:
            # También permite encontrar títulos si el chunker
            # ha eliminado los símbolos #.
            candidate = line.strip().strip(
                "*_`"
            )

        if (
            normalize_text(candidate)
            == normalized_heading
        ):
            suffixes.append(
                "\n".join(
                    chunk_lines[
                        line_index + 1:
                    ]
                )
            )

    return suffixes


def evaluate_heading_integration(
    markdown_dir,
    chunks_dir,
    *,
    selected_strategies=None,
    probe_words=12,
    orphan_max_words=5,
    save_path=None,
):
    """
    Compara los Markdown originales con sus diferentes
    resultados de chunking.

    Devuelve
    --------
    heading_df:
        Una fila por título y método.

    document_summary:
        Métricas por documento y método.

    method_summary:
        Resumen por método y tipo documental.

    fig, ax:
        Gráfica de integración de títulos.
    """

    markdown_dir = Path(markdown_dir)
    chunks_dir = Path(chunks_dir)

    # Índice de Markdown originales por nombre de archivo.
    markdown_files = {
        path.name: path
        for path in markdown_dir.rglob("*.md")
    }

    if not markdown_files:
        raise FileNotFoundError(
            "No se encontraron archivos Markdown en "
            f"{markdown_dir}"
        )

    rows = []

    for result_path in chunks_dir.rglob(
        "*.json"
    ):
        try:
            with result_path.open(
                encoding="utf-8"
            ) as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, dict):
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

        input_file = data.get(
            "input_file",
            "",
        )

        # Permite nombres de Windows y Linux.
        input_name = (
            str(input_file)
            .replace("\\", "/")
            .split("/")[-1]
        )

        # Ignora resultados procedentes de archivos JSON.
        if not input_name.lower().endswith(
            ".md"
        ):
            continue

        markdown_path = markdown_files.get(
            input_name
        )

        if markdown_path is None:
            print(
                "No se encontró el Markdown original:",
                input_name,
            )
            continue

        markdown_text = markdown_path.read_text(
            encoding="utf-8"
        )

        headings = extract_markdown_headings(
            markdown_text,
            probe_words=probe_words,
        )

        doc_id = data.get(
            "doc_id",
            markdown_path.stem,
        )

        abbreviation = (
            data.get("abbreviation")
            or strategy
        )

        document_type = (
            identify_document_type(
                doc_id,
                input_file,
            )
        )

        chunk_texts = [
            chunk.get("text", "")
            for chunk in chunks
        ]

        for heading_number, heading in enumerate(
            headings,
            start=1,
        ):
            suffixes = []
            heading_chunk_ids = []

            for chunk_position, chunk in enumerate(
                chunks
            ):
                chunk_text = chunk.get(
                    "text",
                    "",
                )

                found_suffixes = (
                    find_heading_suffixes(
                        chunk_text,
                        heading["heading"],
                    )
                )

                if found_suffixes:
                    suffixes.extend(
                        found_suffixes
                    )

                    heading_chunk_ids.append(
                        chunk.get(
                            "chunk_id",
                            chunk_position,
                        )
                    )

            heading_found = bool(suffixes)

            probe = heading[
                "description_probe"
            ]

            # El título y el principio de su contenido
            # deben aparecer en el mismo chunk.
            integrated = (
                heading_found
                and bool(probe)
                and any(
                    probe
                    in normalize_text(suffix)
                    for suffix in suffixes
                )
            )

            # Comprueba si el contenido aparece en algún chunk,
            # aunque esté separado del título.
            description_found = (
                bool(probe)
                and any(
                    probe
                    in normalize_text(chunk_text)
                    for chunk_text in chunk_texts
                )
            )

            # Título localizado muy cerca del final del chunk.
            orphan_at_end = (
                heading_found
                and not integrated
                and any(
                    len(
                        normalize_text(
                            suffix
                        ).split()
                    )
                    <= orphan_max_words
                    for suffix in suffixes
                )
            )

            if not heading[
                "has_description"
            ]:
                status = "without_description"

            elif integrated:
                status = "integrated"

            elif not heading_found:
                status = "heading_missing"

            elif orphan_at_end:
                status = "orphan_at_end"

            elif description_found:
                status = "separated_from_content"

            else:
                status = "content_missing"

            rows.append(
                {
                    "result_file": str(
                        result_path
                    ),
                    "strategy": strategy,
                    "abbreviation": abbreviation,
                    "doc_id": doc_id,
                    "input_file": input_name,
                    "document_type": (
                        document_type
                    ),
                    "heading_number": (
                        heading_number
                    ),
                    "heading_level": (
                        heading["level"]
                    ),
                    "heading": (
                        heading["heading"]
                    ),
                    "description_probe": probe,
                    "has_description": (
                        heading[
                            "has_description"
                        ]
                    ),
                    "heading_found": (
                        heading_found
                    ),
                    "description_found": (
                        description_found
                    ),
                    "integrated": integrated,
                    "orphan_at_end": (
                        orphan_at_end
                    ),
                    "status": status,
                    "heading_chunk_ids": (
                        heading_chunk_ids
                    ),
                }
            )

    heading_df = pd.DataFrame(rows)

    if heading_df.empty:
        raise ValueError(
            "No se pudieron comparar títulos. "
            "Comprueba las rutas y los nombres de archivo."
        )

    # Los encabezados sin contenido asociado se excluyen.
    evaluable_df = heading_df[
        heading_df["has_description"]
    ].copy()

    if evaluable_df.empty:
        raise ValueError(
            "No se encontraron títulos con contenido "
            "asociado."
        )

    # Métricas por documento.
    document_summary = (
        evaluable_df
        .groupby(
            [
                "result_file",
                "strategy",
                "abbreviation",
                "doc_id",
                "input_file",
                "document_type",
            ],
            as_index=False,
        )
        .agg(
            total_headings=(
                "heading",
                "count",
            ),
            found_headings=(
                "heading_found",
                "sum",
            ),
            integrated_headings=(
                "integrated",
                "sum",
            ),
            orphan_headings=(
                "orphan_at_end",
                "sum",
            ),
        )
    )

    document_summary[
        "heading_coverage_percentage"
    ] = (
        document_summary["found_headings"]
        / document_summary["total_headings"]
        * 100
    )

    document_summary[
        "heading_integration_percentage"
    ] = (
        document_summary[
            "integrated_headings"
        ]
        / document_summary["total_headings"]
        * 100
    )

    document_summary[
        "conditional_integration_percentage"
    ] = np.where(
        document_summary["found_headings"] > 0,
        (
            document_summary[
                "integrated_headings"
            ]
            / document_summary[
                "found_headings"
            ]
            * 100
        ),
        np.nan,
    )

    document_summary[
        "orphan_percentage"
    ] = (
        document_summary["orphan_headings"]
        / document_summary["total_headings"]
        * 100
    )

    # Resumen macro: cada documento pesa lo mismo.
    method_summary = (
        document_summary
        .groupby(
            [
                "strategy",
                "abbreviation",
                "document_type",
            ],
            as_index=False,
        )
        .agg(
            mean_heading_coverage=(
                "heading_coverage_percentage",
                "mean",
            ),
            mean_heading_integration=(
                "heading_integration_percentage",
                "mean",
            ),
            mean_conditional_integration=(
                "conditional_integration_percentage",
                "mean",
            ),
            mean_orphan_percentage=(
                "orphan_percentage",
                "mean",
            ),
            number_of_documents=(
                "doc_id",
                "nunique",
            ),
        )
    )

    # Orden de mayor a menor integración.
    method_order = (
        document_summary
        .groupby("abbreviation")[
            "heading_integration_percentage"
        ]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    figure_width = max(
        10,
        len(method_order) * 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, 6)
    )

    sns.barplot(
        data=document_summary,
        x="abbreviation",
        y="heading_integration_percentage",
        hue="document_type",
        order=method_order,
        palette=DOCUMENT_COLORS,
        errorbar="se",
        capsize=0.12,
        ax=ax,
    )

    ax.set_xlabel("Método de chunking")
    ax.set_ylabel(
        "Integración título-contenido (%)"
    )
    ax.set_title(
        "Porcentaje de encabezados unidos "
        "a su contenido"
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    ax.legend(
        title="Tipo de documento",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return (
        heading_df,
        document_summary,
        method_summary,
        fig,
        ax,
    )