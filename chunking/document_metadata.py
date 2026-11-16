
import json
import os
import re
import unicodedata
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "by", "for", "from", "in",
    "of", "on", "the", "to", "with",
    "de", "del", "el", "en", "la", "las",
    "los", "para", "por", "y", "con", "un", "una",
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def document_key(specialty: str, filename: str) -> str:
    """Clave compartida entre JSON y Markdown."""
    return f"{normalize_name(specialty)}/{normalize_name(Path(filename).stem)}"


def document_prefix(specialty: str, filename: str) -> str:
    words = normalize_name(Path(filename).stem).split("_")
    words = [word for word in words if word not in STOPWORDS]

    short_title = "_".join(words[:4]) or "document"
    return f"{normalize_name(specialty)}_{short_title}"


def assign_document_ids(
    documents: list[tuple[str, str]],
    registry_path: Path,
) -> dict[str, str]:
    """
    Asigna IDs persistentes a documentos (especialidad, nombre).
    JSON y Markdown comparten ID cuando coinciden sus claves.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}

    if not isinstance(registry, dict):
        raise ValueError("El registro de document_ids debe ser un objeto JSON.")

    requested = {
        document_key(specialty, filename): (specialty, filename)
        for specialty, filename in documents
    }

    used_ids = set(registry.values())

    # Orden estable, independiente del orden de lectura.
    for key in sorted(requested):
        if key in registry:
            continue

        specialty, filename = requested[key]
        prefix = document_prefix(specialty, filename)

        number = 1
        while f"{prefix}_{number:03d}" in used_ids:
            number += 1

        doc_id = f"{prefix}_{number:03d}"
        registry[key] = doc_id
        used_ids.add(doc_id)

    if len(used_ids) != len(registry):
        raise ValueError("El registro contiene doc_id duplicados.")

    # Escritura atómica para evitar registros incompletos.
    temporary_path = registry_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, registry_path)

    return {key: registry[key] for key in requested}
