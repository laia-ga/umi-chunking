"""Identificadores persistentes compartidos entre documentos JSON y Markdown.

Admite rutas relativas dentro de la especialidad. Ejemplos:
    document_key('urologia', 'guia.json') -> 'urologia/guia'
    document_key('urologia', 'litiasis/guia.json') -> 'urologia/litiasis/guia'

Así evita colisiones cuando dos subespecialidades tienen archivos llamados igual.
IMPORTANTE: mover archivos previamente registrados cambia la clave y puede
asignar un ID nuevo; migra manualmente data/document_ids.json si necesitas
preservar ese ID al reorganizar carpetas.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "by", "for", "from", "in", "of", "on", "the", "to", "with",
    "de", "del", "el", "en", "la", "las", "los", "para", "por", "y", "con", "un", "una",
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def document_key(specialty: str, filename: str | Path) -> str:
    """Clave con especialidad + ruta relativa, sin extensión.

    Conserva el esquema antiguo para archivos directamente en la especialidad.
    """
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("filename debe ser una ruta relativa dentro de la especialidad")
    parts = [normalize_name(specialty)]
    parts += [normalize_name(part) for part in path.parts[:-1]]
    parts.append(normalize_name(path.stem))
    if any(not component for component in parts):
        raise ValueError("specialty o filename no contienen un nombre válido")
    return "/".join(parts)


def document_prefix(specialty: str, filename: str | Path) -> str:
    words = normalize_name(Path(filename).stem).split("_")
    words = [word for word in words if word and word not in STOPWORDS]
    short_title = "_".join(words[:4]) or "document"
    return f"{normalize_name(specialty)}_{short_title}"


def assign_document_ids(
    documents: list[tuple[str, str]], registry_path: Path,
) -> dict[str, str]:
    """Crea o recupera identificadores estables con registro compartido.

    JSON y Markdown obtienen el mismo ID cuando specialty y ruta sin extensión
    coinciden. Las nuevas claves usan sufijos numéricos para evitar colisiones.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = (json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.exists() else {})
    if not isinstance(registry, dict) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in registry.items()
    ):
        raise ValueError("El registro de document_ids debe ser un objeto string->string")
    if len(set(registry.values())) != len(registry):
        raise ValueError("El registro contiene doc_id duplicados")

    requested = {document_key(specialty, filename): (specialty, filename)
                 for specialty, filename in documents}
    if len(requested) != len(documents):
        raise ValueError("Dos documentos tienen la misma clave normalizada")
    used_ids = set(registry.values())
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

    temporary = registry_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, registry_path)
    return {key: registry[key] for key in requested}
