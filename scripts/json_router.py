import json

from pathlib import Path

## chunking si se adapta a alguna estructura
from chunkers.json.JSON_chunker import (
    HierarchicalJSONChunker,
)

## chunking general
from preprocessing.json_structured_chunker import (
    chunk_document as generic_chunk_document,
)


def extract_key_paths(value, prefix=""):
    """
    Extrae todas las rutas de claves de un JSON.

    Los valores no se tienen en cuenta. Por tanto, estas claves
    cuentan como existentes aunque sus valores estén vacíos:

        "title": ""
        "authors": []
        "year": null

    Los índices de las listas se sustituyen por [] para que dos
    listas con diferente número de elementos sean comparables.
    """

    paths = set()

    if isinstance(value, dict):
        for key, child_value in value.items():
            current_path = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            # La clave cuenta aunque su valor esté vacío.
            paths.add(current_path)

            paths.update(
                extract_key_paths(
                    child_value,
                    current_path,
                )
            )

    elif isinstance(value, list):
        list_prefix = (
            f"{prefix}[]"
            if prefix
            else "[]"
        )

        # Se recorren todos los elementos porque diferentes
        # diccionarios podrían contener claves distintas.
        for item in value:
            if isinstance(item, (dict, list)):
                paths.update(
                    extract_key_paths(
                        item,
                        list_prefix,
                    )
                )

    return paths


def load_json_templates(templates_dir):
    """
    Carga todos los moldes JSON de una carpeta.
    """

    templates_dir = Path(templates_dir)

    template_paths = sorted(
        templates_dir.glob("*.json")
    )

    if not template_paths:
        raise FileNotFoundError(
            f"No hay moldes JSON en: {templates_dir}"
        )

    templates = {}

    for template_path in template_paths:
        with template_path.open(
            encoding="utf-8",
        ) as file:
            templates[template_path.stem] = json.load(file)

    return templates


def compare_document_with_templates(
    document,
    templates,
    *,
    threshold=0.50,
    minimum_common_paths=5,
    ignore_single_root=False,
):
    """
    Compara un documento con varios moldes.

    template_coverage:
        Porcentaje de las rutas del molde que aparecen en el
        documento.

    input_precision:
        Porcentaje de las rutas del documento que también aparecen
        en el molde.

    f1_score:
        Combina ambas métricas y se utiliza para seleccionar el
        mejor molde entre los que superan el umbral.
    """

    document_paths = extract_key_paths(document)

    if not document_paths:
        raise ValueError(
            "El documento no contiene ninguna clave."
        )

    comparisons = []

    for template_name, template in templates.items():
        template_paths = extract_key_paths(template)

        if not template_paths:
            continue

        common_paths = (
            document_paths
            & template_paths
        )

        template_coverage = (
            len(common_paths)
            / len(template_paths)
        )

        input_precision = (
            len(common_paths)
            / len(document_paths)
        )

        if template_coverage + input_precision:
            f1_score = (
                2
                * template_coverage
                * input_precision
                / (
                    template_coverage
                    + input_precision
                )
            )
        else:
            f1_score = 0.0

        union_paths = (
            document_paths
            | template_paths
        )

        jaccard_similarity = (
            len(common_paths)
            / len(union_paths)
            if union_paths
            else 0.0
        )

        accepted = (
            template_coverage >= threshold
            and len(common_paths) >= minimum_common_paths
        )

        comparisons.append(
            {
                "template_name": template_name,
                "accepted": accepted,
                "template_coverage": template_coverage,
                "input_precision": input_precision,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
                "common_paths": sorted(common_paths),
                "missing_template_paths": sorted(
                    template_paths - document_paths
                ),
                "extra_document_paths": sorted(
                    document_paths - template_paths
                ),
                "number_of_common_paths": len(
                    common_paths
                ),
                "number_of_template_paths": len(
                    template_paths
                ),
                "number_of_document_paths": len(
                    document_paths
                ),
            }
        )

    if not comparisons:
        raise ValueError(
            "Ningún molde contiene claves comparables."
        )

    accepted_comparisons = [
        comparison
        for comparison in comparisons
        if comparison["accepted"]
    ]

    # Entre los moldes que superan el 50 %, se selecciona el que
    # presenta el mejor equilibrio entre cobertura y precisión.
    if accepted_comparisons:
        best_match = max(
            accepted_comparisons,
            key=lambda comparison: (
                comparison["f1_score"],
                comparison["template_coverage"],
                comparison["number_of_common_paths"],
            ),
        )

        matched = True

    else:
        # Se devuelve el más cercano únicamente como diagnóstico,
        # pero el documento irá al método 2.
        best_match = max(
            comparisons,
            key=lambda comparison: (
                comparison["f1_score"],
                comparison["template_coverage"],
                comparison["number_of_common_paths"],
            ),
        )

        matched = False

    return {
        "matched": matched,
        "best_match": best_match,
        "comparisons": comparisons,
    }

def route_and_chunk_json(
    json_path,
    *,
    templates_dir,
    plan_registry,
    template_plan_map,
    generic_chunk_function,
    structured_chunker_kwargs=None,
    threshold=0.50,
    minimum_common_paths=5,
    ignore_single_root=False,
    min_chunk_tokens=100,
    doc_id=None,
):
    """
    Detecta la estructura de un documento JSON y selecciona
    automáticamente el método de chunking.

    Los archivos JSON de `templates_dir` se utilizan únicamente
    como moldes estructurales.

    Si el documento coincide con un molde:
        Aplica HierarchicalJSONChunker utilizando el DOCUMENT_PLAN
        asociado a ese molde.

    Si no coincide con ningún molde:
        Aplica la función genérica:

            generic_chunk_function(
                doc=document,
                doc_id=doc_id,
                min_chunk_tokens=min_chunk_tokens,
            )

    Parámetros
    ----------
    json_path:
        Archivo JSON que se quiere procesar.

    templates_dir:
        Carpeta que contiene los moldes JSON.

    plan_registry:
        Diccionario que relaciona un nombre de plan con un objeto
        SplitNode.

    template_plan_map:
        Diccionario que relaciona el nombre de cada archivo molde
        con el nombre correspondiente en plan_registry.

    generic_chunk_function:
        Función de chunking utilizada como método 2.

    structured_chunker_kwargs:
        Parámetros que se enviarán a HierarchicalJSONChunker.

    threshold:
        Proporción mínima de rutas del molde que deben estar
        presentes en el documento.

    minimum_common_paths:
        Número mínimo absoluto de rutas coincidentes.

    ignore_single_root:
        Indica si debe ignorarse una raíz única como "document"
        durante la comparación.

    min_chunk_tokens:
        Tamaño mínimo utilizado por el método 2.

    doc_id:
        Identificador del documento. Si no se proporciona,
        se utiliza el nombre del archivo sin extensión.

    Devuelve
    --------
    tuple:
        chunks, routing_metadata
    """

    json_path = Path(json_path)
    templates_dir = Path(templates_dir)

    if doc_id is None:
        doc_id = json_path.stem

    # ---------------------------------------------------------
    # Cargar el documento
    # ---------------------------------------------------------

    with json_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    # ---------------------------------------------------------
    # Cargar los archivos JSON utilizados como moldes
    # ---------------------------------------------------------

    templates = load_json_templates(
        templates_dir
    )

    # ---------------------------------------------------------
    # Comparar la estructura del documento con los moldes
    # ---------------------------------------------------------

    matching_result = compare_document_with_templates(
        document=document,
        templates=templates,
        threshold=threshold,
        minimum_common_paths=minimum_common_paths,
        ignore_single_root=ignore_single_root,
    )

    best_match = matching_result[
        "best_match"
    ]

    best_template_name = best_match[
        "template_name"
    ]

    # ---------------------------------------------------------
    # MÉTODO 1: HierarchicalJSONChunker
    # ---------------------------------------------------------

    if matching_result["matched"]:
        if best_template_name not in template_plan_map:
            raise KeyError(
                "El documento coincide con el molde "
                f"'{best_template_name}', pero dicho molde no "
                "aparece en template_plan_map."
            )

        plan_name = template_plan_map[
            best_template_name
        ]

        if plan_name not in plan_registry:
            raise KeyError(
                f"El plan '{plan_name}' no aparece en "
                "plan_registry."
            )

        document_plan = plan_registry[
            plan_name
        ]

        chunker_kwargs = dict(
            structured_chunker_kwargs or {}
        )

        # El plan se selecciona automáticamente según
        # el molde reconocido.
        chunker_kwargs["document_plan"] = (
            document_plan
        )

        if plan_name == "guideline":
            chunker_kwargs["indivisible_list_paths"] = {"toc", "references", "abbreviations", "metadata.authors", "recommendations",}

        elif plan_name == "paper":
            chunker_kwargs["indivisible_list_paths"] = {}

        elif plan_name == "ficha_tecnica":
            chunker_kwargs["indivisible_list_paths"] = {}

        chunker = HierarchicalJSONChunker(
            **chunker_kwargs
        )

        chunks = chunker.chunk_document(
            document
        )

        route = "method_1_hierarchical"
        selected_template = best_template_name
        selected_plan = plan_name

    # ---------------------------------------------------------
    # MÉTODO 2: chunk_document genérico
    # ---------------------------------------------------------

    else:
        chunks = generic_chunk_function(
            doc=document,
            doc_id=doc_id,
            min_chunk_tokens=min_chunk_tokens,
        )

        route = "method_2_generic"
        selected_template = None
        selected_plan = None

    # ---------------------------------------------------------
    # Información sobre la decisión tomada
    # ---------------------------------------------------------

    routing_metadata = {
        "input_file": json_path.name,
        "doc_id": doc_id,
        "route": route,
        "selected_template": selected_template,
        "selected_plan": selected_plan,
        "best_candidate": best_template_name,
        "template_coverage": best_match[
            "template_coverage"
        ],
        "input_precision": best_match[
            "input_precision"
        ],
        "f1_score": best_match[
            "f1_score"
        ],
        "jaccard_similarity": best_match[
            "jaccard_similarity"
        ],
        "number_of_common_paths": best_match[
            "number_of_common_paths"
        ],
        "number_of_template_paths": best_match[
            "number_of_template_paths"
        ],
        "number_of_document_paths": best_match[
            "number_of_document_paths"
        ],
        "matched_paths": best_match[
            "common_paths"
        ],
        "missing_template_paths": best_match[
            "missing_template_paths"
        ],
        "extra_document_paths": best_match[
            "extra_document_paths"
        ],
        "all_template_comparisons": (
            matching_result["comparisons"]
        ),
    }

    return chunks, routing_metadata