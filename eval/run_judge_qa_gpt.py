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
################################################################################


# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import csv
import json
import os
import sys
from pathlib import Path
from typing import Literal

import argparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


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

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignora el CSV de una ejecución anterior y empieza de cero "
            "(por defecto, si el CSV ya existe, se reanuda a partir de "
            "donde se quedó)."
        ),
    )

    return parser.parse_args()

# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================

# Modelo utilizado como juez
# gpt-5.6-luna: el más económico de la gama actual (sept. 2026),
# más que suficiente para una clasificación 0/1/2 con criterio claro.
MODEL_NAME = "gpt-5.6-luna"

# ==============================================================================
# 4. CLIENTE OPENAI
# ==============================================================================

# Cargar variables del archivo .env
load_dotenv()

# El cliente de OpenAI busca por defecto la variable de entorno
# OPENAI_API_KEY. Si en .env se llama distinto (p.ej. OPENAI_KEY),
# la leemos explícitamente por su nombre.
_openai_api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_KEY")
)

if not _openai_api_key:
    raise RuntimeError(
        "No se ha encontrado la clave de la API de OpenAI. "
        "Asegúrate de tener un archivo .env con "
        "OPENAI_API_KEY=... (o OPENAI_KEY=...) en la raíz del proyecto."
    )

# Cliente OpenAI
client = OpenAI(api_key=_openai_api_key)

# ==============================================================================
# 5. GENERACIÓN CON EL MODELO
# ==============================================================================

class RelevanceJudgment(BaseModel):
    """
    Esquema de la respuesta esperada del juez.

    Usar Structured Outputs con este modelo garantiza que la API
    devuelve JSON válido con estos dos campos, y que 'score' solo
    puede ser 0, 1 o 2 (lo aplica la propia API, no un parseo manual).
    """

    score: Literal[0, 1, 2]
    reason: str


def generate_llm_response(
    prompt: str,
):
    """
    Envía un prompt al modelo mediante la API de OpenAI y devuelve
    el objeto RelevanceJudgment ya parseado (Structured Outputs).
    """

    response = client.responses.parse(
        model=MODEL_NAME,
        instructions=(
            "You are a strict information "
            "retrieval evaluator."
        ),
        input=prompt,
        text_format=RelevanceJudgment,
    )

    return response.output_parsed


# ==============================================================================
# 6. FUNCIÓN LLM JUDGE
# ==============================================================================

def llm_judge_relevance(
    question: str,
    gold_answer: str,
    chunk_text: str,
):
    """
    Evalúa un chunk y devuelve (score, reason, judge_error).

    Con Structured Outputs, la API garantiza que la respuesta cumple el
    esquema de RelevanceJudgment (score en {0, 1, 2}, reason como string),
    así que aquí ya no hace falta parsear ni validar nada a mano.
    """

    prompt = f"""
    Your task is to evaluate the relevance of the RETRIEVED CHUNK
    for answering the QUESTION, using the GOLD ANSWER as reference.

    QUESTION:
    {question}

    GOLD ANSWER:
    {gold_answer}

    RETRIEVED CHUNK:
    {chunk_text}

    Compare the retrieved chunk directly with the gold answer.

    Assign:

    2 = The chunk contains the answer, or enough information to
    answer the question correctly.

    1 = The chunk contains some information from the gold
    answer or useful information toward answering the question,
    but the answer is incomplete.

    0 = The chunk contains no information useful for answering
    the question.

    IMPORTANT:
    If the chunk contains the same factual information as the
    gold answer, even using different words, assign 2.

    If the chunk contains only part of the gold answer,
    assign 1.

    Do NOT require exact wording.
    """

    try:

        judgment = generate_llm_response(
            prompt
        )

        return (
            judgment.score,
            judgment.reason,
            "",
        )

    except Exception as error:

        return (
            None,
            "",
            f"judge_error: {error}",
        )


# ==============================================================================
# 7. GUARDAR FILA EN CSV
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
# 8. EVALUAR LOS CHUNKS DE UNA PREGUNTA
# ==============================================================================

def judge_chunks(
    question_id,
    question,
    gold_answer,
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
        # Evaluar el chunk
        # ------------------------------------------------------

        score, reason, judge_error = (
            llm_judge_relevance(
                question=question,
                gold_answer=gold_answer,
                chunk_text=chunk_text,
            )
        )

        if judge_error:
            print(
                f"    ERROR rank {rank}: "
                f"{judge_error}"
            )

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
            f"score={score}"
        )

        print(
            f"    Reason: {reason}"
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
                "question_id":
                    question_id,

                "question":
                    question,

                "gold_answer":
                    gold_answer,

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

                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "chunk_index":
                    chunk.get(
                        "chunk_index"
                    ),

                "chunk_text":
                    chunk_text,

                "relevance_score":
                    score,

                "llm_reason":
                    reason,

                "judge_error":
                    judge_error,
            },
        )


# ==============================================================================
# 9. MAIN
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
    # 9.1 CARGAR CONFIGURACIÓN
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
    # 9.2 COMPROBAR RESULTADOS DE RETRIEVAL
    # ==========================================================================

    if not retrieval_results_file.exists():

        raise FileNotFoundError(
            "\nNo se ha encontrado:\n"
            f"{retrieval_results_file}\n\n"
            "Ejecuta primero run_retrieval.py."
        )


    # ==========================================================================
    # 9.3 CARGAR RETRIEVAL_RESULTS.JSON
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
    # 9.4 PREPARAR SALIDA
    # ==========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.restart and output_file.exists():

        output_file.unlink()

        print(
            "Se ha borrado el CSV anterior (--restart)."
        )

    # Reanudación: si ya existe un CSV de una ejecución anterior,
    # conservamos las preguntas ya evaluadas y solo repetimos la última
    # (por si el proceso se cortó a mitad de evaluarla) y las que falten.
    already_judged_question_ids = set()

    if output_file.exists():

        with output_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            existing_rows = list(
                csv.DictReader(file)
            )

        existing_question_ids = [
            row["question_id"]
            for row in existing_rows
            if row.get("question_id")
        ]

        if existing_question_ids:

            # La última pregunta puede haberse quedado a medias si el
            # proceso se interrumpió mientras se evaluaba (corte de luz,
            # reinicio de Windows, etc.), así que se descarta y se repite
            # entera por seguridad.
            last_question_id = existing_question_ids[-1]

            kept_rows = [
                row for row in existing_rows
                if row["question_id"] != last_question_id
            ]

            already_judged_question_ids = {
                row["question_id"] for row in kept_rows
            }

            with output_file.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                if kept_rows:

                    writer = csv.DictWriter(
                        file,
                        fieldnames=kept_rows[0].keys(),
                    )

                    writer.writeheader()
                    writer.writerows(kept_rows)

                else:

                    # Solo había filas de la pregunta descartada:
                    # dejamos el CSV vacío, se regenerará con la cabecera
                    # en la primera fila nueva que se escriba.
                    pass

            print(
                f"Reanudando: {len(already_judged_question_ids)} "
                "preguntas ya evaluadas se conservan del CSV anterior. "
                f"Se repite la última ({last_question_id!r}) por si "
                "quedó a medias."
            )


    # ==========================================================================
    # 9.5 RECORRER PREGUNTAS
    # ==========================================================================

    for query_number, result in enumerate(
        results,
        start=1,
    ):

        # ------------------------------------------------------
        # QUESTION ID
        # ------------------------------------------------------

        question_id = result.get(
            "question_id"
        )

        # ------------------------------------------------------
        # PREGUNTA
        # ------------------------------------------------------

        question = result.get(
            "question", "",
        )

        # ------------------------------------------------------
        # GOLD ANSWER
        # ------------------------------------------------------

        gold_answer = result.get(
            "gold_answer",
            "",
        )

        # ------------------------------------------------------
        # Mostrar pregunta
        # ------------------------------------------------------

        print()
        print(
            f"[{query_number}/{len(results)}]"
        )

        print(
            f"Pregunta: {question}"
        )

        # ------------------------------------------------------
        # Comprobar identificador de la pregunta
        # ------------------------------------------------------

        if not question_id:

            print(
                "  AVISO: no se encontró "
                "el identificador de la pregunta."
            )

            continue

        # ------------------------------------------------------
        # Reanudación: saltar preguntas ya evaluadas
        # ------------------------------------------------------

        if question_id in already_judged_question_ids:

            print(
                "  Ya evaluada en una ejecución anterior, se omite."
            )

            continue
        
        # ------------------------------------------------------
        # Comprobar pregunta
        # ------------------------------------------------------

        if not question:

            print(
                "  AVISO: no se encontró "
                "el texto de la pregunta."
            )

            continue

        # ------------------------------------------------------
        # Comprobar respuesta
        # ------------------------------------------------------

        if not gold_answer:

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
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
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
                    question_id=question_id,
                    question=question,
                    gold_answer=gold_answer,
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
    # 9.6 FINAL
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
# 10. EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()