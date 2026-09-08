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

# Resultados creados por run_retrieval.py
RETRIEVAL_RESULTS_FILE = (
    BASE_DIR
    / "output"
    / "retrieval"
    / "retrieval_results.json"
)

# Carpeta de salida
OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "judge"
)

# CSV final
OUTPUT_FILE = (
    OUTPUT_DIR
    / "llm_judge_results.csv"
)


# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================

# Modelo utilizado como juez
MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

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
    """
    Evalúa si el chunk contiene la información necesaria
    para responder correctamente a la pregunta.

    Devuelve:
        score:
            0 = no relevante
            1 = parcialmente relevante
            2 = totalmente relevante

        reason:
            explicación breve

        error:
            mensaje de error si algo falla
    """

    prompt = f"""
Evaluate whether the retrieved chunk contains the information
needed to answer the question according to the reference answer.

Question:
{query}

Reference answer:
{answer}

Retrieved chunk:
{chunk_text}

Use the following relevance scale:

0 = NOT RELEVANT
The retrieved chunk does not contain the information needed
to answer the question.

1 = PARTIALLY RELEVANT
The retrieved chunk contains some useful information,
but it is incomplete or insufficient to fully answer
the question.

2 = FULLY RELEVANT
The retrieved chunk contains enough information
to correctly answer the question.

Important rules:

- Judge the information, not exact wording.
- Do not penalize paraphrases.
- A chunk is fully relevant if it contains the information
  needed to answer the question correctly.
- Return JSON only.
- Do not include Markdown.
- Do not include text outside the JSON.

Return exactly this structure:

{{
    "score": 0,
    "reason": "short explanation"
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

        # ------------------------------------------------------
        # Guardar resultado
        # ------------------------------------------------------

        append_judge_row(
            OUTPUT_FILE,
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

    if not RETRIEVAL_RESULTS_FILE.exists():

        raise FileNotFoundError(
            "\nNo se ha encontrado:\n"
            f"{RETRIEVAL_RESULTS_FILE}\n\n"
            "Ejecuta primero run_retrieval.py."
        )


    # ==========================================================================
    # 10.3 CARGAR RETRIEVAL_RESULTS.JSON
    # ==========================================================================

    with RETRIEVAL_RESULTS_FILE.open(
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
    if OUTPUT_FILE.exists():

        OUTPUT_FILE.unlink()


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
        f"{OUTPUT_FILE}"
    )


# ==============================================================================
# 11. EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()