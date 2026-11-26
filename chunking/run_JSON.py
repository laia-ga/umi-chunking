"""Ejecuta JSON Chunking con IDs persistentes y metadatos clínicos.

Compatible con dos esquemas de configuración:

- umi-chunking: configs/chunker_config.json (chunk_fold) +
  configs/tokenizer_config.json (tokenizer).
- umi-rag: configs/chunker_config.json con paths, tokenizer y execution.

Entrada recomendada:
    data/raw/json/<especialidad>/<subespecialidad>/<documento>.json
También admite <especialidad>/<documento>.json y documentos planos que
incluyan la especialidad en el JSON.

Metadatos: specialty (str), subspecialty (str | None),
subspecialties (list[str]). Pueden definirse en el documento o inferirse
mediante las carpetas. Una discrepancia se considera un error, no se ignora.

Uso:
    python -m chunking.run_JSON
    python -m chunking.run_JSON --no-cache
    python -m chunking.run_JSON --config configs/chunker_config.json
    python -m chunking.run_JSON --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import psutil

from chunking.chunk_cache import cache_fingerprint, load_cached_artifacts
from chunking.document_metadata import (
    assign_document_ids,
    document_key,
    normalize_name,
)
from chunking.utilities import (
    calculate_stats,
    create_tokenizer,
    create_token_counter,
    save_json,
)

# Compatibilidad con el layout del paquete de los dos repositorios.
try:
    from chunking.chunkers.json.JSON_chunker import HierarchicalJSONChunker
except ModuleNotFoundError as error:
    if error.name != "chunking.chunkers.json.JSON_chunker":
        raise
    from chunking.chunkers.json import HierarchicalJSONChunker


# Solo se usa cuando no existe get_document_config() en el paquete.
FALLBACK_CONTEXT_FIELDS = {
    "paper": [],
    "guideline": [],
    "ficha_tecnica": ["medicine_name"],
}
FALLBACK_INDIVISIBLE_LIST_PATHS = {
    "paper": ["authors", "keywords"],
    "guideline": ["toc", "references", "abbreviations", "metadata.authors"],
    "ficha_tecnica": ["excipients", "national_codes", "national_codes_queried"],
}
FICHA_TECNICA_SIGNATURE = {
    "medicine_name", "composition", "pharmaceutical_form",
    "indications", "posology",
}
CHUNKER_SCHEMA_VERSION = 5  # Aumentar si cambia la lógica de chunking o payload.
_WORKER_CONTEXT = threading.local()


class DocumentConfig(NamedTuple):
    plan: Any
    indivisible_list_paths: tuple[str, ...]
    context_fields: tuple[str, ...]


def get_active_document_config(document_type: str) -> DocumentConfig:
    """Permite utilizar get_document_config() o los módulos document_plan_*.

    IMPORTANTE: para reproducir exactamente el mismo chunking en dos repos,
    los document plans y las listas de contexto deben ser idénticos.
    """
    try:
        from chunking.chunkers.json import get_document_config
    except ImportError:
        get_document_config = None

    if get_document_config is not None:
        config = get_document_config(document_type)
        return DocumentConfig(
            config.plan,
            tuple(config.indivisible_list_paths),
            tuple(config.context_fields),
        )

    from chunking.chunkers.json.document_plan_ficha_tecnica import DOCUMENT_PLAN_FC
    from chunking.chunkers.json.document_plan_guideline import DOCUMENT_PLAN_GL
    from chunking.chunkers.json.document_plan_paper import DOCUMENT_PLAN_PAPER
    plans = {
        "paper": DOCUMENT_PLAN_PAPER,
        "guideline": DOCUMENT_PLAN_GL,
        "ficha_tecnica": DOCUMENT_PLAN_FC,
    }
    if document_type not in plans:
        raise ValueError(f"Tipo de documento no soportado: {document_type!r}")
    return DocumentConfig(
        plans[document_type],
        tuple(FALLBACK_INDIVISIBLE_LIST_PATHS[document_type]),
        tuple(FALLBACK_CONTEXT_FIELDS[document_type]),
    )


def detect_document_type(data: dict[str, Any], source_path: str) -> str:
    if FICHA_TECNICA_SIGNATURE.issubset(data):
        return "ficha_tecnica"
    declared = data.get("document_type")
    if isinstance(declared, str) and declared.strip():
        declared = declared.strip().casefold().replace("-", "_")
        if declared in ("guideline", "clinical_guideline"):
            if data.get("article_type"):
                raise ValueError(
                    f"Tipo documental contradictorio en {source_path}: "
                    "guideline y article_type."
                )
            return "guideline"
        raise ValueError(f"Tipo documental no soportado en {source_path}: {declared}")
    if isinstance(data.get("article_type"), str) and data["article_type"].strip():
        return "paper"
    raise ValueError(f"No se pudo detectar el tipo documental de {source_path}")


def _find_project_root(config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.resolve().parent.parent
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "chunker_config.json").is_file():
            return parent
    raise FileNotFoundError(
        "No se encontró configs/chunker_config.json ascendiendo desde run_JSON.py. "
        "Indica --config explícitamente."
    )


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"La configuración debe ser un objeto JSON: {path}")
    return result


def _rooted_path(root: Path, raw: str | Path, field: str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError(f"Ruta ausente o vacía: {field}")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


class Settings(NamedTuple):
    root: Path
    input_dir: Path
    output_dir: Path
    registry_path: Path
    model_name: str
    add_special_tokens: bool
    target_tokens: int
    max_tokens: int
    workers: int
    default_specialty: str | None


def load_settings(
    config_path: Path | None = None,
    tokenizer_config_path: Path | None = None,
    input_override: Path | None = None,
    output_override: Path | None = None,
    registry_override: Path | None = None,
    workers_override: int | None = None,
) -> Settings:
    root = _find_project_root(config_path)
    config_path = (config_path or root / "configs" / "chunker_config.json").resolve()
    config = _read_config(config_path)
    paths = config.get("paths")

    if paths is not None and not isinstance(paths, dict):
        raise ValueError("'paths' debe ser un objeto JSON")
    if isinstance(paths, dict):
        if "input_dir" not in paths or "output_dir" not in paths:
            raise ValueError("'paths' necesita input_dir y output_dir")
        # umi-rag: configuración unificada en chunker_config.json
        input_dir = _rooted_path(root, paths["input_dir"], "paths.input_dir")
        output_dir = _rooted_path(root, paths["output_dir"], "paths.output_dir")
    else:
        # umi-chunking: chunk_fold + tokenizer_config.json aparte
        folder = config.get("chunk_fold")
        if not isinstance(folder, str) or not folder.strip() or Path(folder).name != folder or folder in (".", ".."):
            raise ValueError("'chunk_fold' debe ser un nombre simple de carpeta")
        input_dir = root / "data" / "raw" / "json"
        output_dir = root / "output" / "chunking" / folder

    if input_override is not None:
        input_dir = _rooted_path(root, input_override, "--input-dir")
    if output_override is not None:
        output_dir = _rooted_path(root, output_override, "--output-dir")

    token_cfg = config.get("tokenizer")
    if tokenizer_config_path is not None:
        token_cfg = _read_config(tokenizer_config_path.resolve()).get("tokenizer")
    elif not isinstance(token_cfg, dict):
        token_cfg = _read_config(root / "configs" / "tokenizer_config.json").get("tokenizer")
    if not isinstance(token_cfg, dict):
        raise ValueError("Falta el objeto 'tokenizer' en la configuración")

    model_name = token_cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("tokenizer.model_name debe ser un nombre válido")
    add_special_tokens = token_cfg.get("add_special_tokens", True)
    if not isinstance(add_special_tokens, bool):
        raise ValueError("tokenizer.add_special_tokens debe ser booleano")
    target_tokens = token_cfg.get("target_tokens")
    max_tokens = token_cfg.get("max_tokens")
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (target_tokens, max_tokens)):
        raise ValueError("target_tokens y max_tokens deben ser enteros positivos")
    if target_tokens > max_tokens:
        raise ValueError("target_tokens no puede superar max_tokens")

    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("'execution' debe ser un objeto")
    workers = workers_override if workers_override is not None else execution.get("workers", 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers debe ser un entero positivo")

    registry_raw = registry_override or config.get("document_ids_path") or "data/document_ids.json"
    registry_path = _rooted_path(root, registry_raw, "document_ids_path")
    taxonomy_cfg = config.get("taxonomy", {})
    if not isinstance(taxonomy_cfg, dict):
        raise ValueError("'taxonomy' debe ser un objeto JSON")
    default_specialty = _normalized_string(
        taxonomy_cfg.get("default_specialty"),
        "taxonomy.default_specialty", str(config_path),
    )
    return Settings(root, input_dir, output_dir, registry_path,
                    model_name.strip(), add_special_tokens,
                    target_tokens, max_tokens, workers, default_specialty)


def _normalized_string(value: Any, field: str, source: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: '{field}' debe ser una cadena no vacía")
    result = normalize_name(value)
    if not result:
        raise ValueError(f"{source}: '{field}' vacío tras normalizar")
    return result


def _subspecialties(value: Any, source: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{source}: 'subspecialties' debe ser una cadena o lista")
    output: list[str] = []
    for item in value:
        slug = _normalized_string(item, "subspecialties", source)
        if slug and slug not in output:
            output.append(slug)
    return output


def _extract_taxonomy(
    data: dict[str, Any], relative_path: Path,
    default_specialty: str | None = None,
) -> tuple[str, list[str]]:
    """La carpeta manda cuando hay jerarquía; el JSON puede añadir otras etiquetas.

    Si las fuentes discrepan sobre la especialidad, se falla explícitamente.
    """
    path_parts = relative_path.parts
    dir_specialty = normalize_name(path_parts[0]) if len(path_parts) >= 2 else None
    dir_subspecialty = normalize_name(path_parts[1]) if len(path_parts) >= 3 else None
    declared_specialty = _normalized_string(data.get("specialty"), "specialty", str(relative_path))
    if dir_specialty and declared_specialty and dir_specialty != declared_specialty:
        raise ValueError(
            f"{relative_path}: especialidad de carpeta ({dir_specialty}) "
            f"distinta de la del JSON ({declared_specialty})"
        )
    specialty = declared_specialty or dir_specialty or default_specialty
    if not specialty:
        raise ValueError(
            f"{relative_path}: no se conoce la especialidad. Usa una carpeta "
            "<specialty>/ o añade 'specialty' al JSON."
        )

    raw_plural = data.get("subspecialties")
    raw_singular = data.get("subspecialty")
    tags = _subspecialties(raw_plural, str(relative_path))
    singular = _normalized_string(raw_singular, "subspecialty", str(relative_path))
    if singular and singular not in tags:
        tags.append(singular)
    if dir_subspecialty:
        if tags and dir_subspecialty not in tags:
            raise ValueError(
                f"{relative_path}: subespecialidad de carpeta "
                f"({dir_subspecialty}) no coincide con las etiquetas del JSON ({tags})"
            )
        if dir_subspecialty not in tags:
            tags.insert(0, dir_subspecialty)
    return specialty, tags


def _source_path(path: Path, root: Path, input_dir: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.relative_to(input_dir).as_posix()


def load_documents(settings: Settings) -> list[dict[str, Any]]:
    if not settings.input_dir.is_dir():
        raise FileNotFoundError(f"Directorio de documentos inexistente: {settings.input_dir}")
    files = sorted(p for p in settings.input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".json")
    if not files:
        raise FileNotFoundError(f"No hay documentos JSON en: {settings.input_dir}")

    documents = []
    for path in files:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"Un documento JSON debe ser un objeto: {path}")
        relative_path = path.relative_to(settings.input_dir)
        source_path = _source_path(path, settings.root, settings.input_dir)
        doc_type = detect_document_type(data, source_path)
        specialty, subspecialties = _extract_taxonomy(
            data, relative_path, settings.default_specialty
        )

        # Estas etiquetas son metadatos, no contenido clínico; si están en la
        # raíz del JSON, no deben generar un chunk 'unmapped_content'.
        chunk_data = {key: value for key, value in data.items()
                      if key not in {"specialty", "subspecialty", "subspecialties"}}
        registry_filename = (Path(*relative_path.parts[1:]).as_posix()
                             if len(relative_path.parts) >= 2 else relative_path.name)
        documents.append({
            "document_key": document_key(specialty, registry_filename),
            "registry_filename": registry_filename,
            "document_type": doc_type,
            "specialty": specialty,
            "subspecialties": subspecialties,
            "subspecialty": subspecialties[0] if len(subspecialties) == 1 else None,
            "input_file": path.name,
            "source_path": source_path,
            "data": chunk_data,
            "full_text": json.dumps(data, ensure_ascii=False),
        })

    registry_input = [(doc["specialty"], doc["registry_filename"]) for doc in documents]
    keys = [doc["document_key"] for doc in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("Hay documentos con la misma clave de registro (JSON). "
                         "Revisa nombres y rutas de los documentos.")
    assigned = assign_document_ids(registry_input, settings.registry_path)
    for doc in documents:
        doc["doc_id"] = assigned[doc.pop("document_key")]

    ids = [doc["doc_id"] for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("El registro contiene doc_id repetidos entre documentos")
    return documents


def _initialize_worker(model_name: str, add_special_tokens: bool) -> None:
    tokenizer = create_tokenizer(model_name)
    _WORKER_CONTEXT.token_counter = create_token_counter(
        tokenizer, add_special_tokens=add_special_tokens
    )


def _process_document(
    doc: dict[str, Any], target_tokens: int, max_tokens: int,
    measure_ram: bool,
) -> dict[str, Any]:
    token_counter = _WORKER_CONTEXT.token_counter
    document_config = get_active_document_config(doc["document_type"])
    original_token_count = token_counter(doc["full_text"])
    chunker = HierarchicalJSONChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        token_counter=token_counter,
        document_plan=document_config.plan,
        root_field=None,
        indivisible_list_paths=set(document_config.indivisible_list_paths),
        context_fields=list(document_config.context_fields),
    )
    process = psutil.Process(os.getpid()) if measure_ram else None
    before = process.memory_info().rss / (1024 ** 2) if process else None
    start = time.perf_counter()
    chunks = chunker.chunk_document(doc["data"])
    elapsed = time.perf_counter() - start
    after = process.memory_info().rss / (1024 ** 2) if process else None

    stats = calculate_stats(
        chunks=chunks,
        original_token_count=original_token_count,
        execution_time_seconds=elapsed,
        ram_before_mb=before if before is not None else 0.0,
        ram_after_mb=after if after is not None else 0.0,
        token_counter=token_counter,
    )
    if before is None or after is None:
        stats.update(ram_before_mb=None, ram_after_mb=None, ram_increase_mb=None)
    return {"chunks": chunks, "original_token_count": original_token_count, "stats": stats}


def _serialize_chunks(chunks: list[Any], doc: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for chunk in chunks:
        item = asdict(chunk)
        metadata = dict(item.get("metadata") or {})
        metadata.update({
            "specialty": doc["specialty"],
            "subspecialty": doc["subspecialty"],
            "subspecialties": list(doc["subspecialties"]),
        })
        item["metadata"] = metadata
        output.append(item)
    return output


def _save_document_result(
    doc: dict[str, Any], processed: dict[str, Any], settings: Settings,
    fingerprint: str, summary_results: list[dict[str, Any]],
) -> None:
    print(f"Procesado: {doc['source_path']}")
    chunks = processed["chunks"]
    params = {"target_tokens": settings.target_tokens, "max_tokens": settings.max_tokens}
    result = {
        "strategy": "JSONChunking",
        "type": "HierarchicalJSONChunker",
        "abbreviation": "JC",
        "format": "json",
        "doc_id": doc["doc_id"],
        "document_type": doc["document_type"],
        "specialty": doc["specialty"],
        "subspecialty": doc["subspecialty"],
        "subspecialties": doc["subspecialties"],
        "input_file": doc["input_file"],
        "source_path": doc["source_path"],
        "tokenizer_model": settings.model_name,
        "params": params,
        "original_token_count": processed["original_token_count"],
        "number_of_chunks": len(chunks),
        "chunks": _serialize_chunks(chunks, doc),
        "_cache_fingerprint": fingerprint,
    }
    chunks_dir = settings.output_dir / "json" / "chunk_results"
    metrics_dir = settings.output_dir / "json" / "metrics"
    save_json(result, chunks_dir / f"JC_{doc['doc_id']}.json")

    metrics = {
        "strategy": "JSONChunking", "type": "HierarchicalJSONChunker",
        "abbreviation": "JC", "format": "json", "status": "completed",
        "doc_id": doc["doc_id"], "document_type": doc["document_type"],
        "specialty": doc["specialty"],
        "subspecialty": doc["subspecialty"],
        "subspecialties": doc["subspecialties"],
        "input_file": doc["input_file"], "source_path": doc["source_path"],
        "tokenizer_model": settings.model_name,
        "params": params, **processed["stats"],
        "_cache_fingerprint": fingerprint,
    }
    save_json(metrics, metrics_dir / f"JC_{doc['doc_id']}_metrics.json")
    summary_results.append(metrics)


def _fingerprint(doc: dict[str, Any], settings: Settings) -> str:
    config = get_active_document_config(doc["document_type"])
    if is_dataclass(config.plan):
        plan_data = asdict(config.plan)
    else:
        # Una configuración no serializable explícitamente no se debe cachear
        # bajo un fingerprint que ignore el plan semántico.
        raise TypeError("El document plan debe ser un dataclass serializable")
    return cache_fingerprint({
        "version": CHUNKER_SCHEMA_VERSION,
        "chunker": "HierarchicalJSONChunker",
        "doc_id": doc["doc_id"],
        "document_type": doc["document_type"],
        "specialty": doc["specialty"],
        "subspecialties": doc["subspecialties"],
        "source_path": doc["source_path"],
        "content": doc["full_text"],
        "document_plan": plan_data,
        "context_fields": config.context_fields,
        "indivisible_list_paths": config.indivisible_list_paths,
        "tokenizer_model": settings.model_name,
        "add_special_tokens": settings.add_special_tokens,
        "target_tokens": settings.target_tokens,
        "max_tokens": settings.max_tokens,
    })


def main(
    config_path: Path | None = None,
    tokenizer_config_path: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    workers_override: int | None = None,
    use_cache: bool = True,
) -> None:
    settings = load_settings(config_path, tokenizer_config_path,
                             input_dir, output_dir, registry_path, workers_override)
    documents = load_documents(settings)
    print(f"JSON Chunking | {len(documents)} documentos | entrada: {settings.input_dir}")
    print(f"Salida: {settings.output_dir} | registro IDs: {settings.registry_path}")

    summary_file = settings.output_dir / "json" / "summary.csv"
    existing = (pd.read_csv(summary_file, dtype={"doc_id": str})
                if summary_file.is_file() else pd.DataFrame())
    active_ids = {doc["doc_id"] for doc in documents}
    if "doc_id" in existing:
        existing = existing[~existing["doc_id"].astype(str).isin(active_ids)]

    pending = []
    summary_rows: list[dict[str, Any]] = []
    fingerprints = {}
    for doc in documents:
        fingerprint = _fingerprint(doc, settings)
        fingerprints[doc["doc_id"]] = fingerprint
        chunks_path = settings.output_dir / "json" / "chunk_results" / f"JC_{doc['doc_id']}.json"
        metrics_path = settings.output_dir / "json" / "metrics" / f"JC_{doc['doc_id']}_metrics.json"
        cached = (load_cached_artifacts(chunks_path, metrics_path, fingerprint)
                  if use_cache else None)
        if cached is not None:
            chunks_result, metrics_result = cached
            if (
                chunks_result.get("specialty") == doc["specialty"]
                and chunks_result.get("subspecialties") == doc["subspecialties"]
                and metrics_result.get("specialty") == doc["specialty"]
            ):
                summary_rows.append(metrics_result)
                print(f"Caché reutilizada: {doc['source_path']}")
                continue
        pending.append(doc)

    if pending:
        available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        workers = min(settings.workers, available, len(pending))
        print(f"Workers: {workers}; documentos pendientes: {len(pending)}")
        if workers == 1:
            _initialize_worker(settings.model_name, settings.add_special_tokens)
            for doc in pending:
                processed = _process_document(doc, settings.target_tokens, settings.max_tokens, True)
                _save_document_result(doc, processed, settings, fingerprints[doc["doc_id"]], summary_rows)
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(settings.model_name, settings.add_special_tokens),
                thread_name_prefix="json-chunker",
            ) as pool:
                for doc, processed in zip(pending, pool.map(
                    lambda d: _process_document(d, settings.target_tokens, settings.max_tokens, False), pending
                )):
                    _save_document_result(doc, processed, settings, fingerprints[doc["doc_id"]], summary_rows)

    frames = [frame for frame in (existing, pd.DataFrame(summary_rows)) if not frame.empty]
    if frames:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(summary_file, index=False, encoding="utf-8-sig")
    elif summary_file.is_file():
        existing.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Terminado. Resumen: {summary_file}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "--chunker-config", type=Path, dest="config_path", default=None)
    parser.add_argument("--tokenizer-config", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config_path, tokenizer_config_path=args.tokenizer_config,
         input_dir=args.input_dir, output_dir=args.output_dir, registry_path=args.registry,
         workers_override=args.workers, use_cache=not args.no_cache)


if __name__ == "__main__":
    _cli()