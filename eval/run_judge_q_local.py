import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# RUTAS
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent

QUERIES_FILE = BASE_DIR / "data" / "ground_truth" / "queries.jsonl"
RETRIEVAL_DIR = BASE_DIR / "output" / "retrieval"
OUTPUT_DIR = BASE_DIR / "output" / "judge"


# ============================================================
# MODELO LLM JUDGE
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 150


print("=" * 70)
print("CARGANDO MODELO LLM JUDGE")
print("=" * 70)

print(f"Modelo: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)

model.eval()

print("Modelo cargado correctamente.")


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evalúa la relevancia de los chunks recuperados "
            "respecto a las preguntas mediante un LLM."
        )
    )

    parser.add_argument(
        "retrieval_file",
        help="Archivo JSON generado por run_retrieval.py",
    )

    return parser.parse_args()


# ============================================================
# CARGAR PREGUNTAS
# ============================================================

def load_queries(file_path):

    queries = {}

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            query_id = item["query_id"]

            queries[query_id] = item["input"]

    return queries


# ============================================================
# GENERACIÓN DEL LLM
# ============================================================

def generate_llm_response(prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an information retrieval evaluator. "
                "Follow the relevance scale exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# PARSEAR RESPUESTA JSON
# ============================================================

def parse_llm_json(response):

    if not response:
        raise ValueError(
            "El LLM devolvió una respuesta vacía."
        )

    cleaned = response.strip()

    # Eliminar bloques ```json ... ```
    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    # Intentar cargar directamente
    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # Buscar el primer objeto JSON
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = cleaned[start:end + 1]

        return json.loads(json_text)

    raise ValueError(
        f"No se pudo interpretar la respuesta del LLM: {response}"
    )


# ============================================================
# JUEZ DE RELEVANCIA
# ============================================================

def llm_judge_relevance(query, chunk_text):

    prompt = f"""
Your task is to evaluate whether a RETRIEVED CHUNK contains
useful information for answering a QUESTION.

QUESTION:
{query}

RETRIEVED CHUNK:
{chunk_text}

Evaluate the chunk using this scale:

2 = HIGHLY RELEVANT
The chunk contains enough information to answer the question
correctly, or contains the essential information needed to
answer it.

1 = PARTIALLY RELEVANT
The chunk contains useful information for answering the
question, but the information is incomplete and the chunk
would not be sufficient by itself to fully answer it.

0 = NOT RELEVANT
The chunk does not contain useful information for answering
the question.

IMPORTANT RULES:

- Evaluate semantic relevance, not lexical similarity.
- The chunk does not need to use the same wording as the question.
- If the chunk directly contains the information needed to answer
  the question, assign 2.
- If it contains only part of the useful information, assign 1.
- Assign 0 only when the chunk provides no useful information
  for answering the question.
- Do not use a default score.
- Choose the score independently for every chunk.

Return only a JSON object with two fields:
"score": your selected integer (0, 1, or 2)
"reason": a brief explanation
"""

    raw_response = generate_llm_response(
        prompt
    )

    try:

        result = parse_llm_json(
            raw_response
        )

        score = result.get("score")
        reason = result.get(
            "reason",
            "",
        )

        score = int(score)

        if score not in [0, 1, 2]:
            raise ValueError(
                f"Puntuación no válida: {score}"
            )

        return score, reason, None

    except Exception as error:

        return (
            None,
            "",
            f"{type(error).__name__}: {error}",
        )


# ============================================================
# EVALUAR CHUNKS
# ============================================================

def judge_chunks(
    query_id,
    query,
    chunks,
    writer,
):

    print(f"\nPregunta: {query}")
    print(f"Chunks recuperados: {len(chunks)}")

    for chunk in chunks:

        rank = chunk.get("rank")
        chunk_text = chunk.get("text", "")

        score, reason, error = llm_judge_relevance(
            query,
            chunk_text,
        )

        print(
            f"Rank {rank}: "
            f"score={score}"
        )

        print(
            f"  Reason: {reason}"
        )

        if error:
            print(
                f"  ERROR: {error}"
            )

        writer.writerow(
            {
                "query_id": query_id,
                "query": query,
                "rank": rank,
                "strategy": chunk.get("strategy"),
                "document_id": chunk.get("document_id"),
                "document_type": chunk.get("document_type"),
                "format": chunk.get("format"),
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_text": chunk_text,
                "relevance_score": score,
                "llm_reason": reason,
                "judge_error": error,
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    retrieval_file = (
        RETRIEVAL_DIR
        / args.retrieval_file
    )

    if not retrieval_file.exists():

        raise FileNotFoundError(
            f"No existe el archivo de retrieval:\n"
            f"{retrieval_file}"
        )

    if not QUERIES_FILE.exists():

        raise FileNotFoundError(
            f"No existe el archivo de preguntas:\n"
            f"{QUERIES_FILE}"
        )

    # Cargar preguntas
    queries = load_queries(
        QUERIES_FILE
    )

    # Cargar resultados del retrieval
    with retrieval_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        retrieval_results = json.load(
            file
        )

    # Crear nombre del output
    suffix = retrieval_file.stem

    if suffix.startswith("retrieval_"):
        suffix = suffix[len("retrieval_"):]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"judge_{suffix}.csv"
    )

    fieldnames = [
        "query_id",
        "query",
        "rank",
        "strategy",
        "document_id",
        "document_type",
        "format",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "relevance_score",
        "llm_reason",
        "judge_error",
    ]

    print("\n" + "=" * 70)
    print("EVALUACIÓN DE RETRIEVAL MEDIANTE LLM")
    print("=" * 70)

    print(
        f"\nModelo juez: {MODEL_NAME}"
    )

    print(
        f"Preguntas en queries.jsonl: "
        f"{len(queries)}"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        evaluated = 0

        for result in retrieval_results:

            query_id = result.get(
                "query_id"
            )

            # Solo evaluar preguntas que estén
            # actualmente en queries.jsonl
            if query_id not in queries:
                continue

            query = queries[
                query_id
            ]

            chunks = result.get(
                "retrieved_chunks",
                [],
            )

            evaluated += 1

            print(
                f"\n[{evaluated}/{len(queries)}]"
            )

            judge_chunks(
                query_id=query_id,
                query=query,
                chunks=chunks,
                writer=writer,
            )

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)

    print(
        f"\nPreguntas evaluadas: {evaluated}"
    )

    print(
        f"\nResultados guardados en:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()