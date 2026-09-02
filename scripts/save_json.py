import json

from pathlib import Path
from typing import Sequence

from chunkers.json.JSON_chunker import Chunk


def save_json_chunking_result(
    chunks: Sequence[Chunk],
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_tokens: int,
    max_tokens: int,
    token_counter_name: str | None = None,
    document_plan_name: str | None = None,
    root_field: str | None = None,
    indivisible_list_paths: set[str] | None = None,
) -> Path:
    """
    Guarda los chunks producidos por HierarchicalJSONChunker
    con una estructura equivalente a las salidas de los
    chunkers de Markdown.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Identificador común de todos los chunks del documento.
    doc_id = input_path.stem

    # Número de caracteres del archivo JSON original.
    original_text = input_path.read_text(
        encoding="utf-8",
    )

    # Parámetros generales del chunker.
    params = {
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
    }

    if token_counter_name is not None:
        params["token_counter"] = token_counter_name

    if document_plan_name is not None:
        params["document_plan"] = document_plan_name

    if root_field is not None:
        params["root_field"] = root_field

    if indivisible_list_paths is not None:
        params["indivisible_list_paths"] = sorted(
            indivisible_list_paths
        )

    serialized_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_metadata = {
            "chunker": "hierarchical_json_chunking",
            "group_name": chunk.group_name,
            "level": chunk.level,
            "json_paths": list(chunk.json_paths),
            "token_count": chunk.token_count,
            "character_count": len(chunk.text),
            "exceeds_target_limit": (
                chunk.exceeds_target_limit
            ),
            "exceeds_max_limit": (
                chunk.exceeds_max_limit
            ),
        }

        # Se añade block solamente si existe en Chunk.
        block = getattr(chunk, "block", None)

        if block is not None:
            chunk_metadata["block"] = block

        serialized_chunks.append(
            {
                "text": chunk.text,
                "metadata": chunk_metadata,
                "chunk_id": (
                    f"{doc_id}_chunk_{index:04d}"
                ),
                "doc_id": doc_id,
            }
        )

    result = {
        "strategy": "hierarchical_json_chunking",
        "type": "HierarchicalJSONChunker",
        "abbreviation": "HJC",
        "params": params,
        "doc_id": doc_id,
        "input_file": input_path.name,
        "original_character_count": len(original_text),
        "number_of_chunks": len(chunks),
        "chunks": serialized_chunks,
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path