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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import argparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Proteger la escritura al CSV cuando varios hilos terminan a la vez
_csv_lock = threading.Lock()

# Número de reintentos ante fallos transitorios de la API
# (rate limits, cortes de red, etc.), con espera creciente entre intentos
MAX_RETRIES = 5

class FatalAPIError(Exception):
    """
    Error de la API que no tiene sentido reintentar (sin crédito/cuota,
    clave inválida...). Detiene toda la ejecución en vez de generar
    miles de filas de error.
    """


def is_fatal_api_error(error: Exception) -> bool:
    """
    Detecta errores que reintentar no arregla: cuota agotada o
    autenticación fallida.
    """

    text = str(error).lower()

    return (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "invalid_api_key" in text
        or "incorrect api key" in text
        or type(error).__name__ == "AuthenticationError"
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

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help=(
            "Número de llamadas a la API en paralelo (por defecto: 10). "
            "Súbelo si va sobrado de rate limit, bájalo si empiezas "
            "a ver muchos reintentos por 429"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Número máximo de chunks a evaluar por pregunta/estrategia. "
            "Por defecto, se evalúan todos los que traiga el archivo de "
            "retrieval (que ya vienen recortados a su propio top_k desde "
            "que se generó). Solo hace falta si quieres evaluar menos de "
            "los que hay en el archivo."
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

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

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

        except Exception as error:
        
            if is_fatal_api_error(error):
                raise FatalAPIError(
                    str(error)
                ) from error
            
            last_error = error
            wait_seconds = 2 ** attempt

            print(
                    f"    Aviso: fallo en la llamada a la API "
                    f"(intento {attempt + 1}/{MAX_RETRIES}): {error}. "
                    f"Reintentando en {wait_seconds}s..."
                )

            time.sleep(wait_seconds)

    raise last_error


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
    así que no hace falta parsear ni validar nada a mano.
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

    except FatalAPIError:
        # Detener toda la ejecución si la API devuelve un error fatal
        raise
    
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

    Protegido con un lock: con varios hilos evaluando chunks en
    paralelo, dos escrituras a la vez podrían entrelazarse y dejar
    el CSV corrupto si no se serializan.
    """

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with _csv_lock:

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
# 8. CONSTRUIR TAREAS Y EVALUAR UN CHUNK
# ==============================================================================

def build_judge_tasks(
        results,
        top_k,
        done_keys,
):
    """
    Aplana todas las preguntas y sus chunks recuperados en una lista
    de tareas independientes (una por chunk a evaluar), para poder
    repartirlas entre varios hilos.

    Salta los chunks ya evaluados sin error (done keys) y las preguntas 
    sin identificador, texto o chunks reconocibles.
    """

    tasks = []

    for result in results:

        question_id = result.get(
            "question_id"
        )

        question = result.get(
            "question", "",
        )

        gold_answer = result.get(
            "gold_answer",
            "",
        )

        if not question_id or not question:
            continue

        if not gold_answer:

            print(
                f"  AVISO: {question_id} no tiene "
                "respuesta de referencia, se omite."
            )

            continue

        # --------------------------------------------------------
        # CASO 1: retrieval global
        # --------------------------------------------------------

        if "retrieved_chunks" in result:

            chunks_to_judge = result.get(
                "retrieved_chunks", [],
            )[:top_k]

            for rank, chunk in enumerate(
                chunks_to_judge,
                start=1,
            ):

                chunk_strategy = chunk.get("strategy")
                if (question_id, chunk_strategy or "", rank) in done_keys:
                    continue

                tasks.append({
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "chunk": chunk,
                    "rank": rank,
                    "strategy": chunk_strategy,
                })

        # --------------------------------------------------------
        # CASO 2: retrieval separado por estrategia
        # --------------------------------------------------------

        elif "strategies" in result:

            for strategy, retrieved_chunks in result.get(
                "strategies", {},
            ).items():

                for rank, chunk in enumerate(
                    retrieved_chunks[:top_k],
                    start=1,
                ):
                    if (question_id, strategy or "", rank) in done_keys:
                        continue

                    tasks.append({
                        "question_id": question_id,
                        "question": question,
                        "gold_answer": gold_answer,
                        "chunk": chunk,
                        "rank": rank,
                        "strategy": strategy,
                    })

        else:

            print(
                f"  AVISO: {question_id} no tiene "
                "chunks recuperados reconocibles, se omite."
            )

    return tasks


def judge_and_save_task(
    task,
    output_file,
):
    """
    Evalúa un único chunk (una tarea) y guarda la fila en el CSV.

    Es la unidad de trabajo que se reparte entre los hilos: cada
    llamada a la API ocurre aquí, de forma independiente del resto.
    """

    chunk = task["chunk"]

    chunk_text = chunk.get(
        "text", "",
    )

    # ------------------------------------------------------
    # Evaluar el chunk
    # ------------------------------------------------------
    
    score, reason, judge_error = (
        llm_judge_relevance(
            question=task["question"],
            gold_answer=task["gold_answer"],
            chunk_text=chunk_text,
        )
    )

    # ------------------------------------------------------
    # Strategy
    # ------------------------------------------------------
    
    chunk_strategy = (
        task["strategy"]
        or chunk.get("strategy")
    )

    # ------------------------------------------------------
    # Guardar resultado
    # ------------------------------------------------------
    
    append_judge_row(
        output_file,
        {
            "question_id": task["question_id"],
            "question": task["question"],
            "gold_answer": task["gold_answer"],
            "rank": task["rank"],
            "strategy": chunk_strategy,
            "document_id": chunk.get("document_id"),
            "document_type": chunk.get("document_type"),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_text": chunk_text,
            "relevance_score": score,
            "llm_reason": reason,
            "judge_error": judge_error,
        },
    )

    return (
        task["question_id"],
        task["rank"],
        score,
        judge_error,
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
        / f"gpt_judge_{retrieval_suffix}.csv"
    )

    print("=" * 70)
    print("EVALUACIÓN DE RETRIEVAL MEDIANTE LLM")
    print("=" * 70)

    print(
        f"Modelo juez: {MODEL_NAME}"
    )

    print()


    # ==========================================================================
    # 9.1 TOP_K
    # ==========================================================================

    # El archivo de retrieval ya trae los chunks recortados a su propio
    # top_k desde que se generó (Qdrant limitó la búsqueda al crearlo),
    # --top-k permite forzar un límite menor si se quiere evaluar menos
    # chunks de los que trae el archivo
    top_k = args.top_k

    print(
        f"Top K: {'todos los del archivo' if top_k is None else top_k}"
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

    # Reanudación por chunk: se conservan los chunks evaluados sin error y se
    # repiten los que fallaron o faltan
    
    done_keys = set()

    if output_file.exists():

        with output_file.open("r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))

        # Solo se conservan los chunks evaluados sin error
        ok_rows = [
            row for row in existing_rows
            if row.get("relevance_score") not in (None, "")
        ]

        done_keys = {
            (row["question_id"], row["strategy"] or "", int(row["rank"]))
            for row in ok_rows
        }

        if ok_rows:
            with output_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=ok_rows[0].keys())
                writer.writeheader()
                writer.writerows(ok_rows)
        else:
            output_file.unlink()

        print(
            f"Reanudando: {len(ok_rows)} chunks ya evaluados se conservan; "
            f"{len(existing_rows) - len(ok_rows)} con error se repetirán."
        )


    # ==========================================================================
    # 9.5 CONSTRUIR TAREAS Y EVALUAR EN PARALELO
    # ==========================================================================

    tasks = build_judge_tasks(
        results,
        top_k,
        done_keys,
    )

    total = len(tasks)

    print(
        f"Chunks a evaluar: {total}"
    )

    print(
        f"Llamadas en paralelo: {args.workers}"
    )

    print()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers,
    ) as executor:

        futures = [
            executor.submit(
                judge_and_save_task,
                task,
                output_file,
            )
            for task in tasks
        ]

        for future in as_completed(futures):

            completed += 1

            try:

                (
                    question_id,
                    rank,
                    score,
                    judge_error,
                ) = future.result()

            except FatalAPIError as error:

                # Sin cuota / clave inválida: reintentar no sirve.
                # Se cancelan las tareas pendientes y se para.
                print()
                print(
                    "ERROR FATAL de la API, se detiene la ejecución:"
                )
                print(f"  {error}")
                print(
                    "Lo ya evaluado está guardado en el CSV. Cuando se "
                    "resuelva (p.ej. se amplíe el límite), vuelve a "
                    "lanzar el mismo comando y continuará donde se quedó."
                )

                executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )

                sys.exit(1)
            
            except Exception as error:

                print(
                    f"  [{completed}/{total}] "
                    f"ERROR inesperado: {error}"
                )

                continue

            if judge_error:

                print(
                    f"  [{completed}/{total}] "
                    f"{question_id} rank={rank} "
                    f"ERROR: {judge_error}"
                )

            else:

                print(
                    f"  [{completed}/{total}] "
                    f"{question_id} rank={rank} "
                    f"score={score}"
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