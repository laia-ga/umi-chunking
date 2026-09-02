import json

from pathlib import Path
from typing import Callable

from chunkers.json.JSON_chunker import (
    Chunk,
    SplitNode,
    HierarchicalJSONChunker,
    approximate_token_counter,
)


def chunking_test(
    json_path: str | Path,
    *,
    document_plan: SplitNode,
    target_tokens: int = 512,
    max_tokens: int = 8192,
    token_counter: Callable[[str], int] | None = None,
    root_field: str | None = None,
    indivisible_list_paths: set[str] | None = None,
    mostrar_texto: bool = True,
) -> list[Chunk]:
    """
    Carga un JSON, aplica HierarchicalJSONChunker
    y muestra los chunks generados.

    Parámetros
    ----------
    target_tokens:
        Tamaño recomendado utilizado para decidir cuándo
        subdividir las ramas del árbol.

    max_tokens:
        Límite absoluto admitido por el modelo de embeddings.

    Devuelve
    --------
    list[Chunk]
        Lista de chunks generados.
    """

    json_path = Path(json_path)

    # Si no se proporciona un contador, utiliza
    # el contador aproximado por palabras.
    if token_counter is None:
        token_counter = approximate_token_counter

    # Carga el documento JSON.
    with json_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    # Configura el chunker.
    chunker = HierarchicalJSONChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        token_counter=token_counter,
        document_plan=document_plan,
        root_field=root_field,
        indivisible_list_paths=indivisible_list_paths,
    )

    # Genera los chunks.
    chunks = chunker.chunk_document(document)

    total_chunks = len(chunks)

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    n_target = sum(
        chunk.exceeds_target_limit
        for chunk in chunks
    )

    n_max = sum(
        chunk.exceeds_max_limit
        for chunk in chunks
    )

    n_unmapped = sum(
        chunk.group_name.startswith("unmapped_")
        for chunk in chunks
    )

    def percentage(count: int) -> float:
        return (
            count / total_chunks * 100
            if total_chunks
            else 0.0
        )

    print(f"Documento: {json_path.name}")
    print(f"Número de chunks: {total_chunks}")

    print(
        "Superan el objetivo de tokens: "
        f"{n_target} "
        f"({percentage(n_target):.2f}%)"
    )

    print(
        "Superan el límite del modelo: "
        f"{n_max} "
        f"({percentage(n_max):.2f}%)"
    )

    print(
        "Unmapped content: "
        f"{n_unmapped} "
        f"({percentage(n_unmapped):.2f}%)"
    )

    print()

    # --------------------------------------------------------
    # Información de cada chunk
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print("=" * 70)
        print(f"Chunk: {index}")
        print(f"Grupo: {chunk.group_name}")
        print(f"Nivel: {chunk.level}")
        print(f"Tokens: {chunk.token_count}")

        print(
            "Supera el objetivo:",
            chunk.exceeds_target_limit,
        )

        print(
            "Supera el límite del modelo:",
            chunk.exceeds_max_limit,
        )

        print(
            "Rutas JSON:",
            chunk.json_paths,
        )

        if mostrar_texto:
            print()
            print(chunk.text)

        print()

    return chunks
