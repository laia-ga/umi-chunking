################################################################################
# RUN_JUDGE.PY
#
# Evalúa mediante un LLM la relevancia de los chunks recuperados.
#
# Para cada pregunta:
#   - lee la pregunta
#   - lee la respuesta de referencia
#   - evalúa cada chunk recuperado
#   - asigna:
#         0 = no relevante
#         1 = parcialmente relevante
#         2 = totalmente relevante
#
# El modelo se carga directamente desde Hugging Face.
################################################################################


# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import csv
import json
import sys
from pathlib import Path

import argparse

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ==============================================================================
# 2. RUTAS
# ==============================================================================

# Carpeta donde está este script
SCRIPTS_DIR = Path(__file__).resolve().parent

# Raíz del proyecto
BASE_DIR = SCRIPTS_DIR.parent

# Añadir la raíz del proyecto al path
sys.path.append(str(BASE_DIR))

from chunking.utilities import load_config


# Archivo de configuración del retrieval
RETRIEVAL_CONFIG_FILE = (
    BASE_DIR
    / "configs"
    / "retrieval_config.json"
)

# Carpeta de salida
OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "judge"
)


# Nombre del archivo de salida según configuración
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa los resultados de un retrieval mediante un LLM."
    )

    parser.add_argument(
        "retrieval_file",
        help="Nombre del archivo JSON generado por run_retrieval.py",
    )

    return parser.parse_args()

# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================

# Modelo utilizado como juez
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Máximo de tokens que generará el modelo
MAX_NEW_TOKENS = 150

# ==============================================================================
# 4. CARGAR TOKENIZER Y MODELO
# ==============================================================================

print("=" * 70)
print("CARGANDO MODELO LLM JUDGE")
print("=" * 70)

print(
    f"Modelo: {MODEL_NAME}"
)

# La primera vez se descarga automáticamente
# desde Hugging Face.
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
print()


# ==============================================================================
# 5. GENERACIÓN CON EL MODELO
# ==============================================================================

def generate_llm_response(
    prompt: str,
):
    """
    Envía un prompt al modelo y devuelve únicamente
    el texto generado.
    """

    # Qwen es un modelo instruct,
    # así que utilizamos su plantilla de chat
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict information "
                "retrieval evaluator."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Convertir mensajes al formato esperado por Qwen
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenizar
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    # Llevar inputs al mismo dispositivo del modelo
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # Generar sin gradientes
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,

            # Generación determinista
            do_sample=False,
        )

    # Queremos solamente los tokens nuevos
    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated_tokens = outputs[0][
        input_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ==============================================================================
# 6. EXTRAER JSON DE LA RESPUESTA
# ==============================================================================

def parse_llm_json(
    raw_response: str,
):
    """
    Intenta extraer el JSON generado por el modelo.

    Devuelve:
        parsed_json
        error
    """

    if not raw_response:

        return (
            None,
            "El modelo ha devuelto una respuesta vacía.",
        )

    text = raw_response.strip()

    # ----------------------------------------------------------
    # Eliminar bloques Markdown si aparecen
    # ----------------------------------------------------------

    if text.startswith("```"):

        text = text.strip("`").strip()

        if text.startswith("json"):
            text = text[4:].strip()

    # ----------------------------------------------------------
    # Primer intento:
    # respuesta completamente JSON
    # ----------------------------------------------------------

    try:

        return (
            json.loads(text),
            "",
        )

    except json.JSONDecodeError:
        pass

    # ----------------------------------------------------------
    # Segundo intento:
    # buscar desde la primera { hasta la última }
    # ----------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        possible_json = text[
            start:
            end + 1
        ]

        try:

            return (
                json.loads(possible_json),
                "",
            )

        except json.JSONDecodeError as error:

            return (
                None,
                (
                    "No se pudo interpretar el JSON. "
                    f"Respuesta del modelo: {raw_response}. "
                    f"Error: {error}"
                ),
            )

    return (
        None,
        (
            "No se encontró ningún JSON "
            f"en la respuesta: {raw_response}"
        ),
    )


# ==============================================================================
# 7. FUNCIÓN LLM JUDGE
# ==============================================================================

def llm_judge_relevance(
    query: str,
    answer: str,
    chunk_text: str,
):
    prompt = f"""
    Your task is to determine how much of the REFERENCE ANSWER
    is supported by the RETRIEVED CHUNK.

    QUESTION:
    {query}

    REFERENCE ANSWER:
    {answer}

    RETRIEVED CHUNK:
    {chunk_text}

    Compare the retrieved chunk directly with the reference answer.

    Assign:

    2 = The chunk contains the answer, or enough information to
    answer the question correctly.

    1 = The chunk contains some information from the reference
    answer or useful information toward answering the question,
    but the answer is incomplete.

    0 = The chunk contains no information useful for answering
    the question.

    IMPORTANT:
    If the chunk contains the same factual information as the
    reference answer, even using different words, assign 2.

    If the chunk contains only part of the reference answer,
    assign 1.

    Do NOT require exact wording.

    Return only:

    {{
        "score": 0,
        "reason": "brief explanation"
    }}
    """

    try:

        raw_response = generate_llm_response(
            prompt
        )

        parsed, error = parse_llm_json(
            raw_response
        )

        if error:

            return (
                None,
                "",
                error,
            )

        # ------------------------------------------------------
        # Extraer score
        # ------------------------------------------------------

        score = parsed.get(
            "score"
        )

        if score is None:

            return (
                None,
                parsed.get(
                    "reason",
                    "",
                ),
                (
                    "El JSON generado no contiene "
                    "el campo 'score'."
                ),
            )

        try:

            score = int(
                score
            )

        except Exception:

            return (
                None,
                parsed.get(
                    "reason",
                    "",
                ),
                (
                    "El campo 'score' no puede "
                    "convertirse a entero."
                ),
            )

        # ------------------------------------------------------
        # Comprobar rango
        # ------------------------------------------------------

        if score not in [
            0,
            1,
            2,
        ]:

            return (
                None,
                parsed.get(
                    "reason",
                    "",
                ),
                (
                    f"Score no válido: {score}"
                ),
            )

        reason = parsed.get(
            "reason",
            "",
        )

        return (
            score,
            reason,
            "",
        )

    except Exception as error:

        return (
            None,
            "",
            f"judge_error: {error}",
        )


# ==============================================================================
# 8. GUARDAR FILA EN CSV
# ==============================================================================

def append_judge_row(
    csv_path: Path,
    row: dict,
):
    """
    Añade una fila al archivo CSV.
    """

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ==============================================================================
# 9. EVALUAR LOS CHUNKS DE UNA PREGUNTA
# ==============================================================================

def judge_chunks(
    query_id,
    query_text,
    answers,
    retrieved_chunks,
    top_k,
    output_file,
    strategy=None,
):
    """
    Evalúa los Top-K chunks recuperados para una pregunta.
    """

    chunks_to_judge = (
        retrieved_chunks[:top_k]
    )

    for rank, chunk in enumerate(
        chunks_to_judge,
        start=1,
    ):

        chunk_text = chunk.get(
            "text",
            "",
        )

        # ------------------------------------------------------
        # Variables para guardar la mejor evaluación
        # ------------------------------------------------------

        best_score = None
        best_reason = ""
        best_answer = ""
        judge_error = ""

        # ------------------------------------------------------
        # Evaluar contra todas las respuestas
        # ------------------------------------------------------

        for answer in answers:

            score, reason, error = (
                llm_judge_relevance(
                    query=query_text,
                    answer=answer,
                    chunk_text=chunk_text,
                )
            )

            # Si hubo error técnico
            if error:

                judge_error = error

                print(
                    f"    ERROR rank {rank}: "
                    f"{error}"
                )

                continue

            # Primera evaluación válida
            if best_score is None:

                best_score = score
                best_reason = reason
                best_answer = answer

            # Conservar la puntuación más alta
            elif score > best_score:

                best_score = score
                best_reason = reason
                best_answer = answer

        # ------------------------------------------------------
        # Strategy
        # ------------------------------------------------------

        chunk_strategy = strategy

        if chunk_strategy is None:

            chunk_strategy = chunk.get(
                "strategy"
            )

        # ------------------------------------------------------
        # Mostrar resultado
        # ------------------------------------------------------

        print(
            f"    Rank {rank}: "
            f"score={best_score}"
        )

        print(
            f"    Reason: {best_reason}"
        )

        print(
            f"    Chunk: {chunk_text[:500]}"
        )

        print()

        # ------------------------------------------------------
        # Guardar resultado
        # ------------------------------------------------------

        append_judge_row(
            output_file,
            {
                "query_id":
                    query_id,

                "query":
                    query_text,

                "rank":
                    rank,

                "strategy":
                    chunk_strategy,

                "document_id":
                    chunk.get(
                        "document_id"
                    ),

                "document_type":
                    chunk.get(
                        "document_type"
                    ),

                "format":
                    chunk.get(
                        "format"
                    ),

                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "chunk_index":
                    chunk.get(
                        "chunk_index"
                    ),

                "reference_answer":
                    best_answer,

                "chunk_text":
                    chunk_text,

                "relevance_score":
                    best_score,

                "llm_reason":
                    best_reason,

                "judge_error":
                    judge_error,
            },
        )


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():
    args = parse_args()

    retrieval_results_file = (
        BASE_DIR
        / "output"
        / "retrieval"
        / args.retrieval_file
    )

    retrieval_suffix = retrieval_results_file.stem

    if retrieval_suffix.startswith("retrieval_"):
        retrieval_suffix = retrieval_suffix[len("retrieval_"):]

    output_file = (
        OUTPUT_DIR
        / f"judge_{retrieval_suffix}.csv"
    )

    print("=" * 70)
    print("EVALUACIÓN DE RETRIEVAL MEDIANTE LLM")
    print("=" * 70)

    print(
        f"Modelo juez: {MODEL_NAME}"
    )

    print()


    # ==========================================================================
    # 10.1 CARGAR CONFIGURACIÓN
    # ==========================================================================

    retrieval_config = load_config(
        RETRIEVAL_CONFIG_FILE
    )

    top_k = (
        retrieval_config[
            "retrieval"
        ][
            "top_k"
        ]
    )

    print(
        f"Top K: {top_k}"
    )


    # ==========================================================================
    # 10.2 COMPROBAR RESULTADOS DE RETRIEVAL
    # ==========================================================================

    if not retrieval_results_file.exists():

        raise FileNotFoundError(
            "\nNo se ha encontrado:\n"
            f"{retrieval_results_file}\n\n"
            "Ejecuta primero run_retrieval.py."
        )


    # ==========================================================================
    # 10.3 CARGAR RETRIEVAL_RESULTS.JSON
    # ==========================================================================

    with retrieval_results_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(
            file
        )

    print(
        f"Preguntas encontradas: {len(results)}"
    )


    # ==========================================================================
    # 10.4 PREPARAR SALIDA
    # ==========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Borrar CSV anterior
    # para evitar duplicados
    if output_file.exists():

        output_file.unlink()


    # ==========================================================================
    # 10.5 RECORRER PREGUNTAS
    # ==========================================================================

    for query_number, result in enumerate(
        results,
        start=1,
    ):

        # ------------------------------------------------------
        # QUERY ID
        # ------------------------------------------------------

        query_id = result.get(
            "query_id",
            f"q{query_number:03d}",
        )


        # ------------------------------------------------------
        # PREGUNTA
        #
        # queries_and_answers.jsonl usa "input"
        # retrieval_results.json puede usar "query"
        #
        # Admitimos ambos para evitar desajustes.
        # ------------------------------------------------------

        query_text = result.get(
            "input"
        )

        if not query_text:

            query_text = result.get(
                "query",
                "",
            )


        # ------------------------------------------------------
        # RESPUESTAS DE REFERENCIA
        # ------------------------------------------------------

        answers = result.get(
            "answers",
            [],
        )

        # Si hay una respuesta como string,
        # convertirla en lista
        if isinstance(
            answers,
            str,
        ):

            answers = [
                answers
            ]


        # ------------------------------------------------------
        # Mostrar pregunta
        # ------------------------------------------------------

        print()
        print(
            f"[{query_number}/{len(results)}]"
        )

        print(
            f"Pregunta: {query_text}"
        )


        # ------------------------------------------------------
        # Comprobar pregunta
        # ------------------------------------------------------

        if not query_text:

            print(
                "  AVISO: no se encontró "
                "el texto de la pregunta."
            )

            continue


        # ------------------------------------------------------
        # Comprobar respuestas
        # ------------------------------------------------------

        if not answers:

            print(
                "  AVISO: la pregunta no tiene "
                "respuesta de referencia."
            )

            continue


        # ======================================================================
        # CASO 1:
        # Retrieval global
        # ======================================================================

        if "retrieved_chunks" in result:

            retrieved_chunks = result.get(
                "retrieved_chunks",
                [],
            )

            print(
                f"  Chunks recuperados: "
                f"{len(retrieved_chunks)}"
            )

            judge_chunks(
                query_id=query_id,
                query_text=query_text,
                answers=answers,
                retrieved_chunks=retrieved_chunks,
                top_k=top_k,
                output_file=output_file,
            )


        # ======================================================================
        # CASO 2:
        # Retrieval separado por estrategia
        # ======================================================================

        elif "strategies" in result:

            strategy_results = result.get(
                "strategies",
                {},
            )

            for (
                strategy,
                retrieved_chunks,
            ) in strategy_results.items():

                print(
                    f"  Estrategia: {strategy}"
                )

                judge_chunks(
                    query_id=query_id,
                    query_text=query_text,
                    answers=answers,
                    retrieved_chunks=retrieved_chunks,
                    top_k=top_k,
                    output_file=output_file,
                    strategy=strategy,
                )


        # ======================================================================
        # FORMATO NO RECONOCIDO
        # ======================================================================

        else:

            print(
                "  AVISO: no se encontraron "
                "chunks recuperados."
            )


    # ==========================================================================
    # 10.6 FINAL
    # ==========================================================================

    print()
    print("=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)

    print(
        f"Resultados guardados en:\n"
        f"{output_file}"
    )


# ==============================================================================
# 11. EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()