
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
    python data/ground_truth/generate_questions_from_selected_documents.py

Modo de prueba sin llamar al modelo:
    python data/ground_truth/generate_questions_from_selected_documents.py --dry-run

También se puede forzar un proveedor concreto si hace falta:
    python data/ground_truth/generate_questions_from_selected_documents.py \
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

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some environments
    def load_dotenv(*_args, **_kwargs):
        return False


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_DOCS = REPO_ROOT / "data" / "ground_truth" / "selection" / "selected_documents.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "ground_truth" / "question_generation" / "generated_questions.jsonl"
DEFAULT_SHAREABLE_OUTPUT = REPO_ROOT / "data" / "ground_truth" / "question_generation" / "shareable_question_dataset.jsonl"

load_dotenv(REPO_ROOT / ".env")

DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL") or os.environ.get("QUESTIONS_MODEL") or "gpt-4o-mini"
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


def read_json_document(path: Path) -> dict[str, Any]:
    """Carga un documento JSON y valida que sea un diccionario."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"El archivo {path} no contiene un JSON dict: {type(payload).__name__}")
    return payload


def build_document_text(doc: dict[str, Any]) -> str:
    """Construye el texto principal del documento para el prompt del modelo."""
    text_parts = []
    for key in ("title", "content", "body", "text", "summary", "abstract", "description"):
        if key in doc:
            text_parts.extend(recursive_text_extract(doc[key]))
    if not text_parts:
        text_parts.extend(recursive_text_extract(doc))
    return "\n\n".join(part for part in text_parts if part)


def build_prompt(document: dict[str, Any], specialty: str, doc_type: str, max_questions: int) -> str:
    """Configura el prompt para que el modelo genere JSON con preguntas y respuestas."""
    title = normalize_text(document.get("title") or document.get("metadata", {}).get("title") or "Documento médico")
    text = build_document_text(document)
    clean_text = text[:12000]

    return f"""Eres un experto en medicina y en evaluación de sistemas RAG (retrieval-augmented generation).

Tu objetivo es generar preguntas que puedan responderse usando exclusivamente la evidencia presente en el documento médico suministrado.

Genera exactamente {max_questions} preguntas de evaluación a partir del siguiente documento médico.

INSTRUCCIONES PARA UN TEST DE RAG:
- Las preguntas deben ser útiles para comprobar si un chatbot puede responder correctamente usando los chunks recuperados.
- La respuesta debe poder derivarse de la información explícita del documento; no uses conocimiento externo ni preguntas demasiado generales.
- Prioriza preguntas con respuesta concreta y verificable: recomendación, indicación, contraindicación, diagnóstico, manejo, seguimiento, riesgo, criterio de exclusión o evidencia clave.
- Evita preguntas demasiado abiertas, demasiado amplias o con múltiples interpretaciones.
- Las preguntas deben ser útiles para un benchmark de recuperación + respuesta.
- Devuelve un JSON válido con una lista de objetos. Cada objeto debe tener exactamente estos campos:
  {{
    "question": "pregunta clara y específica",
    "answer": "respuesta breve y precisa basada en el documento",
    "difficulty": "easy|medium|hard",
    "question_type": "diagnosis|management|guideline|evidence|interpretation",
    "source_title": "titulo del documento fuente",
    "category": "especialidad o tema médico",
    "reference": "titulo + ruta del documento fuente"
  }}
- No incluyas markdown ni bloques de código. Solo JSON puro.
- Si el documento no permite tantas preguntas, genera menos, pero nunca más de {max_questions}.

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

    match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", stripped, flags=re.DOTALL)
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
        "temperature": 0.2,
        "max_tokens": 2500,
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

    with request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))

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
                "answer": "Se recomienda revisar el documento original y resumir la recomendación principal según el contexto clínico.",
                "difficulty": "medium",
                "question_type": "interpretation",
                "source_title": title,
                "category": specialty,
                "reference": f"{title} ({doc_type})",
            }
        )
    return sample_questions


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Escribe la salida final en JSONL, una línea por pregunta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_shareable_record(record: dict[str, Any]) -> dict[str, Any]:
    """Crea una versión limpia y compartible del dataset para el equipo."""
    return {
        "id": record.get("question_id", ""),
        "specialty": record.get("specialty", ""),
        "category": record.get("category", record.get("specialty", "")),
        "document_title": record.get("source_title", ""),
        "document_type": record.get("document_type", ""),
        "source_path": record.get("relative_path", ""),
        "reference": record.get("reference", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "difficulty": record.get("difficulty", "medium"),
        "question_type": record.get("question_type", "interpretation"),
    }


def build_question_record(row: dict[str, str], file_path: Path, question: dict[str, Any], question_counter: int) -> dict[str, Any]:
    """Normaliza la estructura final para que todas las preguntas sigan el mismo esquema."""
    source_title = normalize_text(question.get("source_title") or row.get("title") or file_path.stem)
    specialty = row.get("specialty") or "unknown"
    doc_type = row.get("document_type") or "unknown"
    final_category = normalize_text(question.get("category") or specialty)
    reference = normalize_text(question.get("reference") or f"{source_title} [{row.get('relative_path', '')}]")

    return {
        "question_id": f"q-{question_counter:05d}",
        "document_path": str(file_path),
        "relative_path": row.get("relative_path", ""),
        "specialty": specialty,
        "category": final_category,
        "document_type": doc_type,
        "type": doc_type,
        "question": normalize_text(question.get("question") or question.get("prompt") or ""),
        "answer": normalize_text(question.get("answer") or question.get("solution") or ""),
        "difficulty": str(question.get("difficulty") or "medium").lower(),
        "question_type": str(question.get("question_type") or "interpretation").lower(),
        "source_title": source_title,
        "reference": reference,
    }


def main() -> None:
    """Punto de entrada principal del generador de preguntas."""
    parser = argparse.ArgumentParser(description="Genera preguntas desde documentos médicos seleccionados.")
    parser.add_argument("--selected-docs", type=Path, default=DEFAULT_SELECTED_DOCS, help="CSV con la lista de documentos seleccionados.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Archivo JSONL de salida con las preguntas generadas.")
    parser.add_argument("--shareable-output", type=Path, default=DEFAULT_SHAREABLE_OUTPUT, help="Versión limpia y lista para compartir con el equipo.")
    parser.add_argument("--max-questions-per-doc", type=int, default=10, help="Máximo de preguntas por documento.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo del endpoint compatible con OpenAI/Ollama.")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="Base URL del endpoint LLM. Por defecto usa OPENAI_BASE_URL o la API de OpenAI.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key para el endpoint. Si no se indica, se usa OPENAI_API_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="Genera preguntas de ejemplo sin llamar al modelo.")
    args = parser.parse_args()

    if args.max_questions_per_doc <= 0:
        raise ValueError("--max-questions-per-doc debe ser un entero positivo.")

    if not args.dry_run and not args.api_key:
        raise ValueError(
            "No se encontró una API key. Define OPENAI_API_KEY o OPENAI_KEY en el entorno o pasa --api-key."
        )

    selected_docs = load_selected_documents(args.selected_docs)
    if not selected_docs:
        raise FileNotFoundError(f"No se encontraron documentos en {args.selected_docs}")

    all_records: list[dict[str, Any]] = []
    for row in selected_docs:
        relative_path = row["relative_path"]
        file_path = resolve_document_path(relative_path)
        if not file_path.exists():
            print(f"AVISO: no existe {file_path}, se omite.")
            continue

        doc = read_json_document(file_path)
        specialty = row.get("specialty") or "unknown"
        doc_type = row.get("document_type") or "unknown"

        if args.dry_run:
            generated_questions = build_mock_questions(doc, specialty, doc_type, min(args.max_questions_per_doc, 2))
        else:
            prompt = build_prompt(doc, specialty, doc_type, args.max_questions_per_doc)
            generated_questions = call_model(prompt, args.model, args.api_base, args.api_key)

        question_counter = len(all_records) + 1
        for q in generated_questions:
            record = build_question_record(row, file_path, q, question_counter)
            if not record["question"]:
                continue
            all_records.append(record)
            question_counter += 1

    write_jsonl(args.output, all_records)
    shareable_records = [build_shareable_record(record) for record in all_records]
    write_jsonl(args.shareable_output, shareable_records)
    print(f"Documentos procesados: {len(selected_docs)}")
    print(f"Preguntas generadas: {len(all_records)}")
    print(f"Salida técnica: {args.output}")
    print(f"Salida compartible: {args.shareable_output}")


if __name__ == "__main__":
    main()
