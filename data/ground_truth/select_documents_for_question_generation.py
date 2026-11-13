
"""Selecciona un subconjunto reproducible de documentos JSON para generar preguntas.

Propósito del script:
- recorrer la carpeta de documentos fuente en data/raw/json;
- detectar la especialidad y el tipo de documento de cada JSON;
- agruparlos en los 4 grandes bloques requeridos por el proyecto;
- tomar una muestra aleatoria reproducible con una semilla fija;
- equilibrar la selección por especialidad y, si procede, por tipo documental.

La salida por defecto se guarda dentro de data/ground_truth/selection/ porque ese
espacio es versionable y comparte mejor con el equipo que la carpeta output/,
que suele estar en .gitignore.

Ejemplo de uso:
    python data/ground_truth/select_documents_for_question_generation.py --docs-per-specialty 5 --question-total 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def safe_console_text(value: Any) -> str:
    """Convierte a texto ASCII seguro para no romper la consola en Windows."""
    text = str(value)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_specialty(value: Any) -> str:
    """Normaliza un nombre de especialidad eliminando espacios repetidos."""
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_specialty(doc: dict[str, Any]) -> str:
    """Intenta leer la especialidad desde metadata o desde la raíz del documento."""
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    for key in ("specialty", "specialty_name", "category"):
        value = metadata.get(key)
        if value:
            return normalize_specialty(value)
    for key in ("specialty", "specialty_name", "category"):
        value = doc.get(key)
        if value:
            return normalize_specialty(value)
    return ""


def infer_document_type(doc: dict[str, Any]) -> str:
    """Clasifica el documento en un tipo general para balancear la muestra."""
    doc_type = str(doc.get("document_type") or "").strip().lower()
    if doc_type in {"guideline", "clinical_guideline"}:
        return "guideline"
    if doc_type in {"technical_sheet", "ficha_tecnica", "medicine", "pharmaceutical_product"}:
        return "ficha_tecnica"
    if doc_type in {"paper", "article", "review", "narrative_review", "systematic_review", "meta_analysis", "rct", "cohort", "case_series", "other"}:
        return "paper"

    if any(key in doc for key in ("medicine_name", "pharmaceutical_form", "posology", "indications")):
        return "ficha_tecnica"

    if any(key in doc for key in ("article_type", "medical_specialty", "author", "journal")):
        return "paper"

    if any(key in doc for key in ("recommendations", "guideline_recommendations", "document_sections")):
        return "guideline"

    return "paper"


def load_specialty_map(path: Path | None) -> dict[str, list[str]]:
    """Carga el JSON que redefine los 4 grupos grandes de especialidad."""
    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de mapeo de especialidades: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"El archivo {path} debe contener un JSON con un diccionario por grupo.")

    normalized: dict[str, list[str]] = {}
    for group, values in raw.items():
        if isinstance(values, str):
            normalized[str(group)] = [values]
        elif isinstance(values, list):
            normalized[str(group)] = [str(v) for v in values]
        else:
            raise ValueError(f"El grupo '{group}' debe ser una lista o string.")
    return normalized


def map_specialty_to_group(specialty: str, specialty_map: dict[str, list[str]]) -> str | None:
    """Asigna una especialidad concreta a uno de los 4 grupos principales."""
    if not specialty_map:
        return specialty

    normalized_specialty = normalize_specialty(specialty).lower()
    for group_name, values in specialty_map.items():
        for value in values:
            normalized_value = normalize_specialty(value).lower()
            if normalized_value == normalized_specialty or normalized_specialty in normalized_value or normalized_value in normalized_specialty:
                return group_name
    return None


def allocate_doc_type_counts(doc_type_counts: dict[str, int], target_count: int) -> dict[str, int]:
    """Distribuye el número deseado de documentos por tipo documental."""
    doc_types = sorted(doc_type_counts)
    if target_count <= 0 or not doc_types:
        return {doc_type: 0 for doc_type in doc_types}

    if len(doc_types) == 1:
        only_type = doc_types[0]
        return {only_type: min(target_count, doc_type_counts[only_type])}

    base = target_count // len(doc_types)
    remainder = target_count % len(doc_types)
    counts = {doc_type: 0 for doc_type in doc_types}

    for idx, doc_type in enumerate(doc_types):
        desired = base + (1 if idx < remainder else 0)
        counts[doc_type] = min(desired, doc_type_counts[doc_type])

    total_assigned = sum(counts.values())
    if total_assigned < target_count:
        remaining = target_count - total_assigned
        order = sorted(doc_types, key=lambda item: (doc_type_counts[item] - counts[item], item), reverse=True)
        while remaining > 0:
            for doc_type in order:
                if remaining <= 0:
                    break
                if counts[doc_type] < doc_type_counts[doc_type]:
                    counts[doc_type] += 1
                    remaining -= 1
    return counts


def pick_documents_for_bucket(items: list[dict[str, Any]], target_count: int, rng: random.Random, balance_document_types: bool) -> list[dict[str, Any]]:
    """Selecciona documentos desde un bucket: aleatorio o equilibrado por tipo."""
    if target_count <= 0:
        return []

    if target_count >= len(items):
        return list(items)

    if balance_document_types:
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_type[str(item.get("document_type") or "unknown")].append(item)

        doc_type_counts = {doc_type: len(group) for doc_type, group in by_type.items()}
        assigned_by_type = allocate_doc_type_counts(doc_type_counts, target_count)

        selected: list[dict[str, Any]] = []
        for doc_type, count in assigned_by_type.items():
            if count <= 0:
                continue
            sample = rng.sample(by_type.get(doc_type, []), min(count, len(by_type.get(doc_type, []))))
            selected.extend(sample)
        if len(selected) < target_count:
            remaining = [item for item in items if item not in selected]
            selected = selected + rng.sample(remaining, min(target_count - len(selected), len(remaining)))
        return selected

    return rng.sample(items, target_count)


def extract_title(doc: dict[str, Any], fallback: str) -> str:
    """Extrae el título del documento si está disponible."""
    if isinstance(doc.get("metadata"), dict):
        title = doc["metadata"].get("title")
        if title:
            return str(title)
    if doc.get("title"):
        return str(doc["title"])
    return fallback


def discover_documents(root_dir: Path) -> list[dict[str, Any]]:
    """Recorre todos los JSON del corpus y devuelve solo los documentos válidos."""
    docs: list[dict[str, Any]] = []
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {root_dir}")

    for json_path in sorted(root_dir.rglob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(payload, dict):
            continue

        specialty = detect_specialty(payload)
        if not specialty:
            continue

        docs.append(
            {
                "file_path": json_path,
                "relative_path": json_path.as_posix(),
                "specialty": specialty,
                "title": extract_title(payload, json_path.stem),
                "document_type": infer_document_type(payload),
                "raw_document_type": str(payload.get("document_type") or "unknown"),
                "content": payload,
            }
        )

    return docs


def build_selection(
    documents: list[dict[str, Any]],
    docs_per_specialty: int,
    seed: int,
    specialty_map: dict[str, list[str]] | None = None,
    balance_document_types: bool = True,
) -> list[dict[str, Any]]:
    """Agrupa por especialidad y escoge la muestra final con una semilla reproducible."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        specialty = doc["specialty"]
        group_name = map_specialty_to_group(specialty, specialty_map) if specialty_map else specialty
        if specialty_map is not None and group_name is None:
            continue
        final_group = group_name or specialty
        grouped[final_group].append(doc)
        doc["selection_group"] = final_group

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for group_name in sorted(grouped):
        items = grouped[group_name]
        allowed = min(docs_per_specialty, len(items))
        if allowed <= 0:
            continue
        chosen = pick_documents_for_bucket(items, allowed, rng, balance_document_types=balance_document_types)
        selected.extend(chosen)

    return sorted(selected, key=lambda item: (item["selection_group"], item["specialty"], str(item["relative_path"])))


def write_csv(output_path: Path, selected: list[dict[str, Any]], total_questions: int, seed: int) -> None:
    """Escribe la lista final de documentos elegidos para la generación de preguntas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection_order",
        "specialty",
        "title",
        "document_type",
        "relative_path",
        "questions_target",
        "seed",
    ]

    docs_count = len(selected)
    base = total_questions // docs_count if docs_count else 0
    remainder = total_questions % docs_count if docs_count else 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, doc in enumerate(selected, start=1):
            questions_target = base + (1 if i <= remainder else 0)
            doc["seed"] = seed
            writer.writerow(
                {
                    "selection_order": i,
                    "specialty": doc["specialty"],
                    "title": doc["title"],
                    "document_type": doc["document_type"],
                    "relative_path": str(doc["relative_path"]),
                    "questions_target": questions_target,
                    "seed": seed,
                }
            )


def write_manifest(output_path: Path, docs_per_specialty: int, seed: int, total_questions: int, selected: list[dict[str, Any]]) -> None:
    """Guarda una vista resumida de la muestra con las cantidades por grupo."""
    grouped: dict[str, int] = defaultdict(int)
    for doc in selected:
        group_name = doc.get("selection_group") or doc["specialty"]
        grouped[group_name] += 1

    manifest = {
        "selection_strategy": "equal_per_specialty_random_sample",
        "seed": seed,
        "docs_per_specialty_requested": docs_per_specialty,
        "selected_documents_count": len(selected),
        "question_total_target": total_questions,
        "questions_per_document_target": round(total_questions / len(selected)) if selected else 0,
        "specialties_selected": dict(sorted(grouped.items())),
        "documents": [
            {
                "group": doc.get("selection_group") or doc["specialty"],
                "specialty": doc["specialty"],
                "title": doc["title"],
                "document_type": doc["document_type"],
                "relative_path": str(doc["relative_path"]),
            }
            for doc in selected
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parámetros de ejecución del script."""
    parser = argparse.ArgumentParser(description="Selecciona documentos aleatorios equilibrando por especialidad y tipo documental.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/json"), help="Carpeta con los JSON fuente.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ground_truth/selection"), help="Carpeta para guardar los resultados.")
    parser.add_argument("--docs-per-specialty", type=int, default=5, help="Documentos a tomar por especialidad/grupo, si existen suficientes.")
    parser.add_argument("--question-total", type=int, default=200, help="Número total de preguntas a generar sobre la muestra final.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la reproducción del muestreo aleatorio.")
    parser.add_argument(
        "--specialty-map-file",
        type=Path,
        default=Path("data/ground_truth/four_specialty_groups.json"),
        help="JSON con los 4 grupos de especialidad a usar.",
    )
    parser.add_argument(
        "--balance-document-types",
        action="store_true",
        default=True,
        help="Equilibra también el número de guideline, paper y ficha técnica dentro de cada grupo cuando sea posible.",
    )
    parser.add_argument(
        "--no-balance-document-types",
        dest="balance_document_types",
        action="store_false",
        help="Desactiva el equilibrio por tipo documental.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del selector."""
    args = parse_args()

    if args.docs_per_specialty <= 0:
        raise ValueError("--docs-per-specialty debe ser un entero positivo.")

    specialty_map = load_specialty_map(args.specialty_map_file) if args.specialty_map_file else None

    if specialty_map and not any(specialty_map.values()):
        raise ValueError("El archivo de mapeo de especialidades está vacío.")

    documents = discover_documents(args.input_dir)
    if not documents:
        raise FileNotFoundError(f"No encontramos documentos JSON válidos en {args.input_dir}")

    selected = build_selection(
        documents,
        args.docs_per_specialty,
        args.seed,
        specialty_map=specialty_map,
        balance_document_types=args.balance_document_types,
    )
    if not selected:
        raise RuntimeError("La selección final está vacía.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "selected_documents.csv"
    manifest_path = args.output_dir / "selection_manifest.json"

    write_csv(csv_path, selected, args.question_total, args.seed)
    write_manifest(manifest_path, args.docs_per_specialty, args.seed, args.question_total, selected)

    print(f"Documentos encontrados: {len(documents)}")
    print(f"Especialidades detectadas: {len({d['specialty'] for d in documents})}")
    print(f"Grupos usados: {len({d.get('selection_group') or d['specialty'] for d in selected})}")
    print(f"Documentos seleccionados: {len(selected)}")
    print(f"Documentos por grupo: {args.docs_per_specialty}")
    print(f"Preguntas objetivo: {args.question_total}")
    print(f"Preguntas aprox. por documento: {round(args.question_total / len(selected))}")
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")

    print("\nListado de selección:")
    for index, doc in enumerate(selected, start=1):
        print(
            f"{index:02d}. [{safe_console_text(doc['specialty'])}] "
            f"{safe_console_text(doc['title'])} -> {safe_console_text(doc['relative_path'])}"
        )


if __name__ == "__main__":
    main()
