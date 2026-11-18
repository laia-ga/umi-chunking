
"""Genera preguntas clínicas a partir de los documentos seleccionados.

Qué hace este script:
- lee el CSV de selección generado por select_documents_for_question_generation.py;
- abre cada documento JSON del corpus;
- extrae su contenido relevante y se lo pasa al modelo LLM;
- genera preguntas clínicas con su respuesta corta;
- para cada pregunta añade metadatos útiles: especialidad, tipo de documento,
  titulo, referencia y categoria.

La idea es tener un flujo reproducible y portable: las rutas por defecto quedan en
la carpeta data/ground_truth para poder compartirlas con otra persona sin depender
de archivos ignorados por git.

Uso típico con OpenAI:
    python scripts/ground_truth/question_generation/generate_questions_from_selected_documents.py

Modo de prueba sin llamar al modelo:
    python scripts/ground_truth/question_generation/generate_questions_from_selected_documents.py --dry-run

También se puede forzar un proveedor concreto si hace falta:
    python scripts/ground_truth/question_generation/generate_questions_from_selected_documents.py \
        --model gpt-4o-mini \
        --api-base https://api.openai.com/v1 \
        --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import request
from tempfile import mkdtemp
from tqdm import tqdm
from urllib.error import HTTPError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some environments
    def load_dotenv(*_args, **_kwargs):
        return False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
FORMATS = {"json": {".json"}, "markdown": {".md", ".markdown"}}
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

DEFAULT_OUTPUT = REPO_ROOT / "data" / "ground_truth" / "generated" / "question_generation" / "generated_questions.jsonl"

load_dotenv(REPO_ROOT / ".env")

DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL") or os.environ.get("QUESTIONS_MODEL") or "gpt-5.6-luna"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY") or ""


def normalize_text(value: Any) -> str:
    """Normaliza texto eliminando espaciados repetidos y cadenas vacías."""
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def recursive_text_extract(value: Any, max_depth: int = 6, depth: int = 0) -> list[str]:
    """Recorre un JSON recursivamente y extrae solo texto útil para el prompt."""
    if depth > max_depth:
        return []

    if isinstance(value, str):
        text = normalize_text(value)
        return [text] if text else []

    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            chunks.extend(recursive_text_extract(item, max_depth, depth + 1))
        return chunks

    if isinstance(value, dict):
        chunks: list[str] = []
        for key, item in value.items():
            if key in {"_id", "id", "metadata", "source", "filename", "path", "relative_path"}:
                continue
            chunks.extend(recursive_text_extract(item, max_depth, depth + 1))
        return chunks

    return []


def load_selected_documents(csv_path: Path) -> list[dict[str, str]]:
    """Lee el CSV con la selección final de documentos."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row.get("relative_path"):
                continue
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def resolve_document_path(relative_path: str) -> Path:
    """Resuelve la ruta del documento desde la relativa del CSV o desde la raíz del repo."""
    candidate = Path(relative_path)
    if candidate.exists():
        return candidate
    repo_root = REPO_ROOT
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate
    return repo_root / relative_path

def discover_documents(raw_dir, doc_format, specialty):
    formats = [doc_format] if doc_format else FORMATS
    for name in formats:
        format_dir = raw_dir / name
        for folder in sorted(format_dir.glob("*")):
            if not folder.is_dir() or (specialty and folder.name.casefold() != specialty):
                continue
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in FORMATS[name]:
                    try:
                        relative_path = Path(os.path.relpath(path, REPO_ROOT))
                    except ValueError:  # Windows: el corpus está en otra unidad.
                        relative_path = path
                    yield {
                        "relative_path": relative_path.as_posix(),
                        "specialty": folder.name,
                        "format": name,
                    }

def read_document(path):
    if path.suffix.lower() == ".json":
        return read_json_document(path)
    if path.suffix.lower() not in FORMATS["markdown"]:
        raise ValueError(f"Formato de documento no soportado: {path}")
    text = path.read_text(encoding="utf-8-sig")
    heading = re.search(r"^ {0,3}#[ \t]+(.+)$", text, flags=re.MULTILINE)
    title = heading.group(1).strip().rstrip("#").strip() if heading else path.stem
    return {"title": title, "content": text, "format": "markdown"}

def read_json_document(path: Path) -> dict[str, Any]:
    """Carga un documento JSON y valida que sea un diccionario."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"El archivo {path} no contiene un JSON dict: {type(payload).__name__}")
    return payload


def build_document_text(document: dict[str, Any]) -> str:
    """Extrae el contenido textual de documentos JSON y Markdown."""

    def extract_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            return "\n".join(
                text
                for item in value
                if (text := extract_text(item))
            )

        if isinstance(value, dict):
            return "\n".join(
                f"{key}: {text}"
                for key, item in value.items()
                if (text := extract_text(item))
            )

        return ""

    if isinstance(document, str):
        return document.strip()

    if not isinstance(document, dict):
        return ""

    # Campos clínicos principales, en orden de prioridad
    priority_fields = [
        # Identificación del medicamento
        "medicine_name",
        "active_ingredient",
        "strength",
        "pharmaceutical_form",
        "route_of_administration",
        "composition",

        # Información clínica de fichas técnicas
        "clinical_indications",
        "clinical_posology",
        "clinical_contraindications",
        "clinical_warnings",
        "clinical_interactions",
        "clinical_pregnancy",
        "clinical_adverse",
        "clinical_overdose",

        # Farmacología
        "pharmacology",
        "pharmaceutical",

        # Información regulatoria
        "marketing_authorisation_holder",
        "marketing_authorisation_number",

        # Artículos y guías clínicas
        "clinical_background",
        "objectives",
        "clinical_question",
        "population",
        "interventions",
        "comparators",
        "outcomes",
        "main_findings",
        "secondary_results",
        "safety_results",
        "clinical_implications",
        "practice_implications",
        "main_conclusion",
        "narrative_summary",
        "document_sections",
        "content",
    ]

    parts = []

    for field in priority_fields:
        if field in document:
            text = extract_text(document[field])
            if text:
                parts.append(f"{field}:\n{text}")

    if parts:
        return "\n\n".join(parts)

    # Si no se encuentran los campos habituales,
    # extraer recursivamente el contenido del JSON.
    return "\n".join(recursive_text_extract(document))


def build_prompt(document: dict[str, Any], specialty: str, doc_type: str, max_questions: int) -> str:
    """Configura el prompt para generar preguntas clínicas y respuestas de referencia."""
    title = normalize_text(
        document.get("title")
        or document.get("metadata", {}).get("title")
        or "Documento médico"
    )
    text = build_document_text(document)
    clean_text = text[:30000]

    print("\n--- CONTENIDO EXTRAÍDO ---")
    print("Título:", title)
    print("Caracteres extraídos:", len(text))
    print("Primeros 500 caracteres:")
    print(repr(text[:500]))
    print("--------------------------\n")

    return f"""Eres un experto en medicina y en evaluación de sistemas RAG (retrieval-augmented generation).

Tu objetivo es generar preguntas clínicas realistas, similares a las que formularía un profesional sanitario al consultar un chatbot especializado en medicina.

Utiliza el documento suministrado exclusivamente como fuente de información para identificar preguntas relevantes y elaborar sus respuestas de referencia.

Genera exactamente {max_questions} preguntas clínicas.

INSTRUCCIONES PARA UN TEST DE RAG:

1. FORMULACIÓN DE LAS PREGUNTAS
- Formula preguntas autónomas, naturales y clínicamente relevantes, como las que podría realizar un médico durante su práctica asistencial o al consultar información científica.
- Las preguntas deben poder entenderse sin conocer el documento del que proceden.
- Las preguntas deben abordar directamente cuestiones clínicas o científicas, no el contenido de un documento como tal.
- No utilices expresiones como "según el documento", "qué se menciona", "qué describe el texto", "qué indica el artículo" o "qué recomienda esta guía".
- Incluye el contexto clínico necesario para que cada pregunta sea suficientemente específica y no resulte ambigua.

2. SELECCIÓN DEL CONTENIDO
- Todas las preguntas deben poder responderse utilizando exclusivamente información explícita del documento suministrado.
- Prioriza información clínica concreta y verificable: recomendaciones, indicaciones, contraindicaciones, criterios diagnósticos, manejo, seguimiento, riesgos, criterios de exclusión o evidencia relevante, resultados, procedimientos, riesgos, características de enfermedades y evidencia científica.
- Evita preguntas demasiado generales, triviales, redundantes o con múltiples interpretaciones.
- No generes varias preguntas que puedan responderse esencialmente con la misma información.
- Si el documento contiene información muy específica, formula preguntas igualmente específicas, sin ampliar su alcance mediante conocimiento externo.
- Procura que las preguntas evalúen aspectos diferentes del contenido.

3. RESPUESTAS DE REFERENCIA
- Cada respuesta debe responder directamente a su pregunta.
- Utiliza únicamente la evidencia presente en el documento.
- Incluye la información necesaria para que la respuesta sea precisa y verificable, sin añadir conocimiento externo.

4. VALIDACIÓN FINAL
Antes de devolver las preguntas, comprueba que:
- Cada pregunta podría formularse a un chatbot clínico sin mencionar ni conocer el documento original.
- Ninguna pregunta hace referencia al documento, artículo, estudio, texto o guía como objeto de consulta.
- La respuesta está respaldada por información explícita del documento.
- La pregunta es suficientemente específica para tener una respuesta de referencia clara.

FORMATO DE SALIDA:
- Devuelve exclusivamente un JSON válido con una lista de objetos.
- Cada objeto debe tener exactamente estos campos:
  {{
    "question": "pregunta clínica autónoma, clara y específica",
    "gold_answer": "respuesta de referencia basada exclusivamente en la evidencia suministrada"
  }}
- No incluyas markdown ni bloques de código.
- Genera exactamente {max_questions} preguntas por documento siempre que exista información suficiente.

Especialidad: {specialty}
Tipo de documento: {doc_type}
Título: {title}

Texto del documento:
{clean_text}
"""

def extract_json_payload(content: str) -> list[dict[str, Any]]:
    """Extrae una lista de preguntas desde contenido JSON o markdown con bloques JSON."""
    stripped = content.strip()
    if not stripped:
        return []
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        stripped = match.group(1)
    payload = json.loads(stripped)
    if isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            return payload["questions"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("items"), list):
            return payload["items"]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"La respuesta del modelo no tenía el formato JSON esperado: {type(payload).__name__}")

def call_model(prompt: str, model: str, api_base: str, api_key: str) -> list[dict[str, Any]]:
    """Llama al endpoint compatible con OpenAI para obtener las preguntas."""
    url = api_base.rstrip("/") + "/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Eres un asistente clínico que responde con JSON válido."},
            {"role": "user", "content": prompt},
        ],
        #"temperature": 0.2,
        "max_completion_tokens": 6000,
    }
    payload = json.dumps(data).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choice = body["choices"][0]
        usage = body.get("usage", {})

        print("\n--- INFORMACIÓN DE GENERACIÓN ---")
        print("Motivo de finalización:", choice.get("finish_reason"))
        print("Tokens de entrada:", usage.get("prompt_tokens"))
        print("Tokens de salida:", usage.get("completion_tokens"))
        print(
            "Tokens de razonamiento:",
            usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        )
        print("--------------------------------\n")

    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"\nError HTTP {e.code} al llamar al modelo {model}")
        print(f"Detalle de la API: {error_body}")
        raise
    message = body["choices"][0]["message"]["content"]
    items = extract_json_payload(message)
    return items

def build_mock_questions(document: dict[str, Any], specialty: str, doc_type: str, max_questions: int) -> list[dict[str, Any]]:
    """Genera un ejemplo local reproducible para validar sin llamar al modelo."""
    title = normalize_text(document.get("title") or document.get("metadata", {}).get("title") or "Documento médico")
    sample_questions: list[dict[str, Any]] = []
    for i in range(1, max_questions + 1):
        sample_questions.append(
            {
                "question": f"{i}. ¿Qué aspecto principal del documento de {specialty} es relevante para la práctica clínica en {title}?",
                "gold_answer": "Se recomienda revisar el documento original y resumir la recomendación principal según el contexto clínico."
            }
        )
    return sample_questions


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Escribe la salida final en JSONL, una línea por pregunta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_question_record(row, file_path, question, question_counter):

    source_title = normalize_text(
        row.get("title") or file_path.stem
    )

    return {
        "question_id": f"q-{question_counter:05d}",
        "question": normalize_text(
            question.get("question", "")
        ),
        "gold_answer": normalize_text(
            question.get("gold_answer", "")
        ),
        "specialty": row.get("specialty", "unknown"),
        "source_title": source_title,
    }

def distribute_questions(
    has_markdown: bool,
    has_json: bool,
    max_questions: int,
) -> dict[str, int]:

    if has_markdown and has_json:
        md_questions = max_questions // 2

        return {
            "markdown": md_questions,
            "json": max_questions - md_questions,
        }

    if has_json:
        return {
            "markdown": 0,
            "json": max_questions,
        }

    if has_markdown:
        return {
            "markdown": max_questions,
            "json": 0,
        }

    raise ValueError("El documento no tiene ningún formato disponible.")

def group_document_pairs(
    selected_docs: list[dict],
    raw_dir: Path,
) -> list[dict]:
    """Agrupa las versiones Markdown y JSON de cada documento."""

    pairs = {}
    raw_dir = raw_dir.resolve()

    for row in selected_docs:

        file_path = resolve_document_path(
            row["relative_path"]
        ).resolve()

        if not file_path.is_file():
            print(f"AVISO: no existe {file_path}, se omite.")
            continue

        relative = file_path.relative_to(raw_dir)

        fmt = relative.parts[0]
        specialty = relative.parts[1]

        if fmt not in ("markdown", "json"):
            continue

        relative_name = Path(*relative.parts[2:]).with_suffix("")
        document_id = row.get("document_id") or relative_name.as_posix()

        key = (specialty, document_id)

        if key not in pairs:
            pairs[key] = {
                "document_id": document_id,
                "specialty": specialty,
                "document_type": row.get("document_type", ""),
                "title": row.get("title", ""),
                "relative_name": relative_name,
                "paths": {},
            }

        pairs[key]["paths"][fmt] = file_path

    # Buscar automáticamente la representación que falte
    for pair in pairs.values():

        for fmt, extensions in {
            "markdown": [".md", ".markdown"],
            "json": [".json"],
        }.items():

            if fmt in pair["paths"]:
                continue

            for extension in extensions:
                candidate = (
                    raw_dir
                    / fmt
                    / pair["specialty"]
                    / f"{pair['relative_name']}{extension}"
                )

                if candidate.is_file():
                    pair["paths"][fmt] = candidate
                    break

    return list(pairs.values())

def main() -> None:
    """Punto de entrada principal del generador de preguntas."""
    parser = argparse.ArgumentParser(description="Genera preguntas desde documentos médicos seleccionados.")
    parser.add_argument("filters", nargs="*", metavar="FILTRO", help="Formato y/o especialidad, en cualquier orden.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raíz del corpus con las carpetas json y markdown.")
    parser.add_argument("--selected-docs", type=Path, help="CSV con la lista de documentos seleccionados.")
    parser.add_argument("--output", type=Path, help="Archivo JSONL de salida con las preguntas generadas.")
    parser.add_argument("--max-questions-per-doc", type=int, default=3, help="Máximo de preguntas por documento.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo del endpoint compatible con OpenAI/Ollama.")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="Base URL del endpoint LLM. Por defecto usa OPENAI_BASE_URL o la API de OpenAI.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key para el endpoint. Si no se indica, se usa OPENAI_API_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="Genera preguntas de ejemplo sin llamar al modelo.")
    args = parser.parse_intermixed_args()
    if args.max_questions_per_doc <= 0:
        raise ValueError("--max-questions-per-doc debe ser un entero positivo.")
    if not args.dry_run and not args.api_key:
        raise ValueError(
            "No se encontró una API key. Define OPENAI_API_KEY o OPENAI_KEY en el entorno o pasa --api-key."
        )
    doc_format = specialty = None
    for value in args.filters:
        value = value.casefold()
        if value in FORMATS and doc_format is None:
            doc_format = value
        elif value not in FORMATS and specialty is None:
            specialty = value
        else:
            parser.error("Indica como máximo un formato y una especialidad.")
    if args.selected_docs and args.filters:
        parser.error("--selected-docs se usa sin filtros posicionales.")
    if args.selected_docs:
        selected_docs = load_selected_documents(args.selected_docs)
    else:
        selected_docs = list(
            discover_documents(
                args.raw_dir.resolve(),
                doc_format,
                specialty
            )
        )

    if not selected_docs:
        parser.error("No se encontraron documentos.")

    # Emparejar Markdown y JSON
    document_pairs = group_document_pairs(
        selected_docs,
        args.raw_dir,
    )

    if not document_pairs:
        parser.error("No se encontraron documentos válidos.")

        if args.output and (args.output.exists() or args.output.is_symlink()):
            parser.error(f"La salida ya existe: {args.output}")

    specialty_filter = specialty
    all_records: list[dict[str, Any]] = []
    processed = 0

    print(f"\nDocumentos originales encontrados: {len(document_pairs)}")
    print(f"Modelo: {args.model}")
    print(f"Preguntas por documento: {args.max_questions_per_doc}\n")

    with tqdm(
        document_pairs,
        desc="Generando preguntas",
        unit="doc",
        dynamic_ncols=True
    ) as progress:

        for pair in progress:

            document_id = pair["document_id"]
            specialty = pair["specialty"]
            doc_type = pair["document_type"] or "unknown"

            paths = pair["paths"]

            progress.set_postfix_str(document_id[:35])

            # Determinar cuántas preguntas generar por formato
            distribution = distribute_questions(
                has_markdown="markdown" in paths,
                has_json="json" in paths,
                max_questions=args.max_questions_per_doc,
            )

            document_processed = False

            for fmt, n_questions in distribution.items():

                if n_questions == 0:
                    continue

                file_path = paths[fmt]

                # Leer documento
                try:
                    doc = read_document(file_path)

                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError
                ) as e:
                    tqdm.write(
                        f"ERROR AL LEER: {file_path} -> {e}"
                    )
                    continue

                document_text = build_document_text(doc)

                if not document_text.strip():
                    tqdm.write(
                        f"DOCUMENTO OMITIDO (sin texto): {file_path}"
                    )
                    continue

                # Preparar metadatos
                raw_metadata = doc.get("metadata")
                metadata = (
                    raw_metadata
                    if isinstance(raw_metadata, dict)
                    else {}
                )

                doc["metadata"] = metadata

                doc["title"] = normalize_text(
                    doc.get("title")
                    or doc.get("medicine_name")
                    or metadata.get("title")
                    or pair["title"]
                    or file_path.stem
                )

                current_doc_type = normalize_text(
                    pair["document_type"]
                    or doc.get("document_type")
                    or metadata.get("document_type")
                    or doc.get("type")
                    or metadata.get("type")
                    or "unknown"
                )

                row = {
                    "title": doc["title"],
                    "specialty": specialty,
                    "document_type": current_doc_type,
                    "format": fmt,
                }

                document_processed = True

                # Generar preguntas
                if args.dry_run:

                    generated_questions = build_mock_questions(
                        doc,
                        specialty,
                        current_doc_type,
                        n_questions
                    )

                else:

                    prompt = build_prompt(
                        doc,
                        specialty,
                        current_doc_type,
                        n_questions
                    )

                    generated_questions = call_model(
                        prompt,
                        args.model,
                        args.api_base,
                        args.api_key
                    )

                    tqdm.write(
                        f"{file_path.name} ({fmt}): "
                        f"{len(generated_questions)} preguntas generadas"
                    )

                # Guardar las preguntas
                question_counter = len(all_records) + 1

                for q in generated_questions[:n_questions]:

                    if not isinstance(q, dict):
                        continue

                    record = build_question_record(
                        row,
                        file_path,
                        q,
                        question_counter
                    )

                    if not record["question"] or not record["gold_answer"]:
                        continue

                    # Nuevos metadatos
                    record["document_id"] = document_id
                    record["origin_format"] = fmt
                    record["available_formats"] = list(paths.keys())

                    all_records.append(record)
                    question_counter += 1

            if document_processed:
                processed += 1

            progress.set_postfix(
                preguntas=len(all_records),
                refresh=True
            )

    if not processed:
        parser.error("No se pudo procesar ningún documento.")
    if args.output is None:
        output_dir = DEFAULT_OUTPUT.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        label = "-".join(
            filter(None, (doc_format, specialty_filter))
        ) or ("csv" if args.selected_docs else "all")

        label = re.sub(r"[^a-z0-9_-]", "-", label)[:60]

        mode = "-dry-run" if args.dry_run else ""

        run_dir = Path(
            mkdtemp(prefix=f"{label}{mode}-", dir=output_dir)
        )

        args.output = run_dir / DEFAULT_OUTPUT.name

    write_jsonl(args.output, all_records)
    print(f"Documentos procesados: {processed}")
    print(f"Preguntas generadas: {len(all_records)}")
    print(f"Salida técnica: {args.output}")


if __name__ == "__main__":
    main()