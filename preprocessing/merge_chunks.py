import math
import re
from typing import List

from preprocessing.json_structured_chunker import Chunk

def _split_long_text(
    text: str,
    max_length: int,
) -> List[str]:
    """
    Divide un texto que supera la longitud máxima.

    Intenta cortar, por este orden:
    1. Párrafos
    2. Líneas
    3. Frases
    4. Palabras
    """

    text = text.strip()

    if len(text) <= max_length:
        return [text]

    separators = [
        "\n\n",
        "\n",
    ]

    units = None
    joiner = " "

    for separator in separators:
        candidate_units = [
            part.strip()
            for part in text.split(separator)
            if part.strip()
        ]

        if len(candidate_units) > 1:
            units = candidate_units
            joiner = separator
            break

    if units is None:
        sentence_units = [
            part.strip()
            for part in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if part.strip()
        ]

        if len(sentence_units) > 1:
            units = sentence_units
            joiner = " "

    if units is None:
        units = text.split()
        joiner = " "

    fragments = []
    current_units = []
    current_length = 0

    for unit in units:
        # Último recurso: una unidad aislada sigue siendo demasiado larga
        if len(unit) > max_length:
            if current_units:
                fragments.append(joiner.join(current_units))
                current_units = []
                current_length = 0

            for start in range(0, len(unit), max_length):
                fragments.append(
                    unit[start:start + max_length]
                )

            continue

        separator_length = len(joiner) if current_units else 0

        candidate_length = (
            current_length
            + separator_length
            + len(unit)
        )

        if current_units and candidate_length > max_length:
            fragments.append(joiner.join(current_units))
            current_units = [unit]
            current_length = len(unit)

        else:
            current_units.append(unit)
            current_length = candidate_length

    if current_units:
        fragments.append(joiner.join(current_units))

    return [
        fragment.strip()
        for fragment in fragments
        if fragment.strip()
    ]

def _balance_group(
    chunks: List[Chunk],
    max_length: int,
) -> List[List[Chunk]]:
    """
    Divide un grupo de chunks consecutivos del mismo tipo
    en grupos equilibrados sin superar max_length.
    """

    if not chunks:
        return []

    separator_length = 2  # "\n\n"

    total_length = (
        sum(len(chunk.text) for chunk in chunks)
        + separator_length * (len(chunks) - 1)
    )

    number_of_groups = max(
        1,
        math.ceil(total_length / max_length),
    )

    target_length = math.ceil(
        total_length / number_of_groups
    )

    result = []
    current_group = []
    current_length = 0

    for chunk in chunks:
        separator = separator_length if current_group else 0

        candidate_length = (
            current_length
            + separator
            + len(chunk.text)
        )

        should_close = (
            current_group
            and (
                candidate_length > max_length
                or current_length >= target_length
            )
        )

        if should_close:
            result.append(current_group)
            current_group = []
            current_length = 0
            separator = 0

        current_group.append(chunk)
        current_length += separator + len(chunk.text)

    if current_group:
        result.append(current_group)

    return result

import re


def _get_record_path(path: str) -> str:
    """
    Obtiene el path correspondiente al elemento principal de una lista.

    Ejemplos:
    - safety.toxicities[0]
      -> safety.toxicities[0]

    - safety.toxicities[0].recognition
      -> safety.toxicities[0]

    - safety.toxicities[0].management[0]
      -> safety.toxicities[0]

    - safety.toxicities[1].prevention
      -> safety.toxicities[1]

    Si el path no contiene índices, se conserva completo.
    """

    if not path:
        return ""

    match = re.search(r"\[\d+\]", path)

    if match:
        return path[:match.end()]

    return path

def merge_structured_chunks(
    original_chunks: List[Chunk],
    doc_id: str,
    max_chunk_length: int,
) -> List[Chunk]:
    """
    Postprocesa los chunks generados por chunk_document().

    Prioridad de agrupación
    -----------------------
    1. Mismo registro o elemento de lista:
       safety.toxicities[0]
       safety.toxicities[0].recognition
       safety.toxicities[0].management[0]

       Todos forman una unidad correspondiente a:
       safety.toxicities[0]

    2. Mismo chunk_type:
       Solo se aplica a grupos que no pertenecen a un elemento
       indexado de una lista.

    Reglas adicionales
    -------------------
    - Solo se agrupan chunks consecutivos.
    - Cada elemento indexado queda separado de los demás:
      toxicities[0] no se une con toxicities[1].
    - Los chunks de tipo "table" nunca se agrupan.
    - Los chunks o grupos que superan max_chunk_length se dividen.
    - Se conserva la trazabilidad en metadata.
    """

    if max_chunk_length <= 0:
        raise ValueError(
            "max_chunk_length debe ser mayor que 0."
        )

    if not original_chunks:
        return []

    # ======================================================
    # 1. Preparar los chunks
    # ======================================================

    prepared_chunks = []

    for chunk in original_chunks:

        # Las tablas no se unen, pero se dividen si superan
        # la longitud máxima.
        fragments = _split_long_text(
            text=chunk.text,
            max_length=max_chunk_length,
        )

        record_path = _get_record_path(chunk.path)

        for fragment_index, fragment_text in enumerate(fragments):

            prepared_chunks.append(
                Chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    path=chunk.path,
                    chunk_type=chunk.chunk_type,
                    text=fragment_text,
                    metadata={
                        **chunk.metadata,
                        "original_chunk_id": chunk.chunk_id,
                        "original_path": chunk.path,
                        "record_path": record_path,
                        "split_from_original": len(fragments) > 1,
                        "fragment_index": fragment_index,
                        "fragment_count": len(fragments),
                    },
                )
            )

    if not prepared_chunks:
        return []

    # ======================================================
    # 2. Primera prioridad: agrupar por registro
    # ======================================================

    record_groups = []
    current_group = []

    for chunk in prepared_chunks:

        # Las tablas siempre forman un grupo independiente
        if chunk.chunk_type == "table":
            if current_group:
                record_groups.append(current_group)
                current_group = []

            record_groups.append([chunk])
            continue

        chunk_record_path = _get_record_path(chunk.path)

        if not current_group:
            current_group = [chunk]
            continue

        previous_record_path = _get_record_path(
            current_group[-1].path
        )

        # Se agrupan únicamente cuando pertenecen exactamente
        # al mismo registro, por ejemplo toxicities[0].
        if (
            chunk_record_path
            and chunk_record_path == previous_record_path
        ):
            current_group.append(chunk)

        else:
            record_groups.append(current_group)
            current_group = [chunk]

    if current_group:
        record_groups.append(current_group)

    # ======================================================
    # 3. Segunda prioridad: agrupar por tipo
    # ======================================================

    groups = []
    current_type_group = []

    def group_types(group):
        """Obtiene los tipos únicos presentes en un grupo."""
        return list(
            dict.fromkeys(
                chunk.chunk_type
                for chunk in group
            )
        )

    def group_record_path(group):
        """
        Obtiene el record_path común del grupo.
        Devuelve una cadena vacía si no existe uno común.
        """
        paths = list(
            dict.fromkeys(
                _get_record_path(chunk.path)
                for chunk in group
                if _get_record_path(chunk.path)
            )
        )

        if len(paths) == 1:
            return paths[0]

        return ""

    def is_indexed_record_group(group):
        """
        Indica si el grupo pertenece a un elemento concreto
        de una lista, como safety.toxicities[0].
        """
        record_path = group_record_path(group)

        return bool(
            record_path
            and re.search(r"\[\d+\]", record_path)
        )

    for record_group in record_groups:

        first_chunk = record_group[0]

        # Las tablas siempre quedan aisladas
        if first_chunk.chunk_type == "table":
            if current_type_group:
                groups.append(current_type_group)
                current_type_group = []

            groups.append(record_group)
            continue

        # Los grupos correspondientes a un registro indexado,
        # por ejemplo toxicities[0], permanecen aislados.
        # No se vuelven a unir por tipo con toxicities[1].
        if is_indexed_record_group(record_group):
            if current_type_group:
                groups.append(current_type_group)
                current_type_group = []

            groups.append(record_group)
            continue

        # Para grupos no indexados sí se permite la unión por tipo
        if not current_type_group:
            current_type_group = record_group.copy()
            continue

        current_types = group_types(current_type_group)
        next_types = group_types(record_group)

        same_single_type = (
            len(current_types) == 1
            and len(next_types) == 1
            and current_types[0] == next_types[0]
        )

        if same_single_type:
            current_type_group.extend(record_group)

        else:
            groups.append(current_type_group)
            current_type_group = record_group.copy()

    if current_type_group:
        groups.append(current_type_group)

    # ======================================================
    # 4. Equilibrar y construir los chunks finales
    # ======================================================

    final_chunks = []
    final_counter = 0

    for group in groups:

        # Las tablas se mantienen aisladas
        if group[0].chunk_type == "table":
            balanced_groups = [[chunk] for chunk in group]

        else:
            balanced_groups = _balance_group(
                chunks=group,
                max_length=max_chunk_length,
            )

        for balanced_group in balanced_groups:

            combined_text = "\n\n".join(
                chunk.text
                for chunk in balanced_group
            )

            source_chunk_ids = list(
                dict.fromkeys(
                    chunk.metadata.get(
                        "original_chunk_id",
                        chunk.chunk_id,
                    )
                    for chunk in balanced_group
                )
            )

            source_paths = list(
                dict.fromkeys(
                    chunk.path
                    for chunk in balanced_group
                )
            )

            source_record_paths = list(
                dict.fromkeys(
                    _get_record_path(chunk.path)
                    for chunk in balanced_group
                    if _get_record_path(chunk.path)
                )
            )

            source_chunk_types = list(
                dict.fromkeys(
                    chunk.chunk_type
                    for chunk in balanced_group
                )
            )

            # Si todos tienen el mismo tipo, se conserva.
            # Si el grupo contiene distintos tipos, será "mixed".
            if len(source_chunk_types) == 1:
                final_chunk_type = source_chunk_types[0]
            else:
                final_chunk_type = "mixed"

            # Conservamos como path el registro común.
            if len(source_record_paths) == 1:
                final_path = source_record_paths[0]
            elif len(source_paths) == 1:
                final_path = source_paths[0]
            else:
                final_path = ""

            final_chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=(
                        f"{doc_id}_chunk_{final_counter:04d}"
                    ),
                    path=final_path,
                    chunk_type=final_chunk_type,
                    text=combined_text,
                    metadata={
                        "postprocessed": True,
                        "merged": len(balanced_group) > 1,
                        "source_chunk_count": len(
                            balanced_group
                        ),
                        "source_chunk_ids": source_chunk_ids,
                        "source_paths": source_paths,
                        "source_record_paths": source_record_paths,
                        "source_chunk_types": source_chunk_types,
                        "character_count": len(combined_text),
                        "max_chunk_length": max_chunk_length,
                        "table_kept_separate": (
                            final_chunk_type == "table"
                        ),
                    },
                )
            )

            final_counter += 1

    return final_chunks