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
import multiprocessing
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List

import argparse

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ==============================================================================
# 2. RUTAS Y CONFIGURACIÓN GENERAL
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

# Nombres de campo del CSV de salida
CSV_FIELDNAMES = [
    "question_id",
    "question",
    "gold_answer",
    "rank",
    "strategy",
    "document_id",
    "document_type",
    "chunk_id",
    "chunk_index",
    "chunk_text",
    "relevance_score",
    "llm_reason",
    "judge_error",
    "judge_version",
]

# Versión del juez, para poder re-evaluar si se cambia
# la rúbrica
JUDGE_VERSION = "local-qa-v2"

# Estado privado de cada proceso: cada worker carga su propio tokenizer y
# modelo en el dispositivo que le corresponde.
_worker_tokenizer = None
_worker_model = None
_worker_device = None
_worker_include_reason = False

# Configuración de argumentos de línea de comandos
def parse_args():
    """Define las opciones de ejecución y devuelve los argumentos validados."""
    parser = argparse.ArgumentParser(
        description="Evalúa los resultados de un retrieval mediante un LLM."
    )

    parser.add_argument(
        "retrieval_file",
        help="Nombre del archivo JSON generado por run_retrieval.py",
    )

    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help=(
            "Número máximo de chunks a evaluar por pregunta/estrategia. "
            "Por defecto, se evalúan todos los que traiga el archivo de "
            "retrieval."
        ),
    )
    # Por defecto se reanudan las evaluaciones pendientes; esta opción
    # solicita explícitamente descartar el CSV previo.
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Descarta el CSV anterior y empieza la evaluación desde cero.",
    )

    # Solo se puede elegir uno de estos modos. Si no se especifica ninguno,
    # iter_devices() usa cuda:0 cuando CUDA está disponible y, si no, CPU.
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        default=None,
        help=(
            "IDs CUDA visibles separados por comas (p. ej. 0,1). "
            "Cada GPU ejecuta una copia independiente del modelo. "
            "Por defecto se usa la GPU 0 si CUDA está disponible."
        ),
    )

    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Fuerza inferencia en CPU en lugar de usar GPU.",
    )

    # Cada GPU seleccionada ejecuta una copia completa del modelo; batch-size
    # limita cuántos chunks procesa esa copia en una llamada a generate().
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Chunks por llamada a generate() en cada worker (por defecto: 1).",
    )
    parser.add_argument(
        "--include-reason",
        action="store_true",
        help="Pide y guarda una breve justificación además del score.",
    )

    return parser.parse_args()


def _positive_int(value: str) -> int:
    """Valida opciones numéricas que deben ser mayores que cero."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero positivo."
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "El valor debe ser al menos 1."
        )
    return parsed


def _parse_gpu_ids(value: str) -> List[int]:
    """Convierte una lista como '0,2' en IDs CUDA únicos y no negativos."""
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Los IDs de GPU deben ser enteros separados por comas."
        ) from error
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError(
            "Indica al menos un ID de GPU no negativo."
        )
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError(
            "No se pueden repetir los IDs de GPU."
        )
    return gpu_ids


# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================

# Modelo instruct que asigna a cada chunk un score de relevancia entre 0 y 2.
MODEL_NAME = "Qwen/Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Sin --include-reason solo se necesita una respuesta JSON corta con el score.
MAX_NEW_TOKENS = 20

# ==============================================================================
# 4. GENERACIÓN CON EL MODELO
# ==============================================================================

def _initialize_judge_worker(
    device: str,
    include_reason: bool,
) -> None:
    """Carga una copia privada del tokenizer/modelo en el proceso worker."""

    global _worker_tokenizer, _worker_model, _worker_device
    global _worker_include_reason

    _worker_device = device
    _worker_include_reason = include_reason

    if device == "cpu":
        torch.set_num_threads(1)

    _worker_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _worker_tokenizer.padding_side = "left"
    if _worker_tokenizer.pad_token_id is None:
        if _worker_tokenizer.eos_token is None:
            raise RuntimeError(
                "El tokenizer no define pad_token ni eos_token para el padding."
            )
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token

    _worker_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map={"": device},
    )
    _worker_model.eval()


def _warmup_judge_worker() -> str:
    if _worker_model is None:
        raise RuntimeError("El modelo del worker no se ha inicializado.")
    if _worker_device.startswith("cuda:"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(
            int(_worker_device.split(":", maxsplit=1)[1])
        )
        return (
            f"{_worker_device} (VRAM libre tras cargar: "
            f"{free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB)"
        )
    return str(_worker_device)


def _format_chat_prompt(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a strict information retrieval evaluator.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return _worker_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ==============================================================================
# 5. EXTRAER JSON DE LA RESPUESTA
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
# 6. FUNCIÓN LLM JUDGE
# ==============================================================================

def build_judge_prompt(task: Dict[str, Any], include_reason: bool) -> str:
    prompt = f"""
    Your task is to evaluate the relevance of the RETRIEVED CHUNK
    for answering the QUESTION, using the GOLD ANSWER as reference.

    QUESTION:
    {task["question"]}

    GOLD ANSWER:
    {task["gold_answer"]}

    RETRIEVED CHUNK:
    {task["chunk"].get("text", "")}

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

    if include_reason:
        return prompt + """

    Return only a JSON object with the score as an integer (0, 1 or 2)
    and a brief reason:

    {"score": 0, "reason": "Brief reason"}
    """

    return prompt + """

    Return only a JSON object with the score as an integer (0, 1 or 2),
    with no explanation:

    {"score": 0}
    """


def _row_for_task(
    task: Dict[str, Any],
    score: int | None,
    reason: str,
    judge_error: str,
) -> Dict[str, Any]:
    chunk = task["chunk"]
    return {
        "question_id": task["question_id"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "rank": task["rank"],
        "strategy": task["strategy"] or chunk.get("strategy"),
        "document_id": chunk.get("document_id"),
        "document_type": chunk.get("document_type"),
        "chunk_id": chunk.get("chunk_id"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_text": chunk.get("text", ""),
        "relevance_score": score,
        "llm_reason": reason,
        "judge_error": judge_error,
        "judge_version": (
            f"{JUDGE_VERSION}-reason"
            if _worker_include_reason
            else f"{JUDGE_VERSION}-score-only"
        ),
    }


def _parse_judge_response(raw_response: str) -> tuple[int | None, str, str]:
    parsed, error = parse_llm_json(raw_response)
    if error:
        return None, "", error
    if not isinstance(parsed, dict):
        return None, "", "La respuesta JSON del modelo no es un objeto."

    score = parsed.get("score")
    if score is None:
        return None, "", "El JSON generado no contiene el campo 'score'."
    if isinstance(score, bool) or (
        isinstance(score, float) and not score.is_integer()
    ):
        return None, "", "El campo 'score' debe ser un entero."

    try:
        score = int(score)
    except (TypeError, ValueError):
        return None, "", "El campo 'score' no puede convertirse a entero."

    if score not in (0, 1, 2):
        return None, "", f"Score no válido: {score}"

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return score, reason, ""


def _generate_judge_batch(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts = [
        _format_chat_prompt(build_judge_prompt(task, _worker_include_reason))
        for task in tasks
    ]
    inputs = _worker_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(_worker_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = _worker_model.generate(
            **inputs,
            max_new_tokens=64 if _worker_include_reason else MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_worker_tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    rows = []
    for task, output in zip(tasks, outputs):
        raw_response = _worker_tokenizer.decode(
            output[prompt_length:],
            skip_special_tokens=True,
        )
        score, reason, error = _parse_judge_response(raw_response)
        rows.append(
            _row_for_task(
                task,
                score,
                reason if _worker_include_reason else "",
                error,
            )
        )
    return rows


def _judge_batch(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        return _generate_judge_batch(tasks)
    except torch.cuda.OutOfMemoryError as error:
        if len(tasks) > 1:
            print(
                f"AVISO: falta de VRAM en {_worker_device} con un batch "
                f"de {len(tasks)}; se dividirá en batches menores. {error}"
            )
            torch.cuda.empty_cache()
            midpoint = len(tasks) // 2
            return (
                _judge_batch(tasks[:midpoint])
                + _judge_batch(tasks[midpoint:])
            )
        failure = f"judge_error: CUDA out of memory: {error}"
    except Exception as error:
        failure = f"judge_error: {error}"

    return [
        _row_for_task(task, None, "", failure)
        for task in tasks
    ]


def _judge_worker_batch(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if _worker_model is None or _worker_tokenizer is None:
        raise RuntimeError("El modelo del worker no se ha inicializado.")
    return _judge_batch(tasks)


# ==============================================================================
# 7. CONSTRUIR TAREAS Y GUARDAR RESULTADOS
# ==============================================================================

def iter_judge_tasks(
    results: List[Dict[str, Any]],
    top_k: int | None,
    done_keys: set[tuple[str, str, int]],
    log_warnings: bool = False,
) -> Iterator[Dict[str, Any]]:
    for result in results:
        question_id = result.get("question_id")
        question = result.get("question", "")
        gold_answer = result.get("gold_answer", "")

        if not question_id or not question:
            if log_warnings:
                print("AVISO: pregunta sin question_id o texto; se omite.")
            continue
        if not gold_answer:
            if log_warnings:
                print(f"AVISO: {question_id} no tiene respuesta de referencia; se omite.")
            continue

        if "retrieved_chunks" in result:
            strategy_chunks = [(None, result.get("retrieved_chunks", []))]
        elif "strategies" in result:
            strategy_chunks = list(result.get("strategies", {}).items())
        else:
            if log_warnings:
                print(f"AVISO: {question_id} no tiene chunks reconocibles; se omite.")
            continue

        for strategy, retrieved_chunks in strategy_chunks:
            chunks = retrieved_chunks if top_k is None else retrieved_chunks[:top_k]
            for rank, chunk in enumerate(chunks, start=1):
                chunk_strategy = strategy or chunk.get("strategy") or ""
                if (str(question_id), chunk_strategy, rank) in done_keys:
                    continue
                yield {
                    "question_id": str(question_id),
                    "question": question,
                    "gold_answer": gold_answer,
                    "chunk": chunk,
                    "rank": rank,
                    "strategy": strategy,
                }


def batched(
    items: Iterator[Dict[str, Any]],
    batch_size: int,
) -> Iterator[List[Dict[str, Any]]]:
    while batch := list(islice(items, batch_size)):
        yield batch


def iter_devices(args) -> List[str]:
    if args.cpu:
        return ["cpu"]
    if not torch.cuda.is_available():
        if args.gpu_ids is not None:
            raise RuntimeError(
                "--gpu-ids especifica GPUs, pero CUDA no está disponible."
            )
        print("CUDA no está disponible; el juez se ejecutará en CPU.")
        return ["cpu"]

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else [0]
    gpu_count = torch.cuda.device_count()
    invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= gpu_count]
    if invalid_ids:
        raise ValueError(
            f"IDs CUDA no disponibles: {invalid_ids}. "
            f"El proceso solo ve {gpu_count} GPU(s)."
        )

    devices = []
    for gpu_id in gpu_ids:
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        print(
            f"GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)}): "
            f"{free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB libres."
        )
        devices.append(f"cuda:{gpu_id}")
    return devices


def prepare_resume(
    output_file: Path,
    restart: bool,
    include_reason: bool,
) -> set[tuple[str, str, int]]:
    if restart and output_file.exists():
        output_file.unlink()
        print("CSV anterior eliminado (--restart).")
        return set()

    if not output_file.exists():
        return set()

    with output_file.open("r", newline="", encoding="utf-8") as file:
        existing_rows = list(csv.DictReader(file))

    expected_version = (
        f"{JUDGE_VERSION}-reason"
        if include_reason
        else f"{JUDGE_VERSION}-score-only"
    )
    incompatible_rows = [
        row for row in existing_rows
        if row.get("judge_version") != expected_version
    ]
    if incompatible_rows:
        raise RuntimeError(
            "El CSV contiene resultados de otra versión del juez o una rúbrica "
            "anterior y no se ha modificado. Usa --restart para recalcularlos."
        )

    successful_rows = [
        row for row in existing_rows
        if row.get("relevance_score") in {"0", "1", "2"}
        and not row.get("judge_error")
        and row.get("judge_version") == expected_version
    ]
    done_keys = {
        (
            row.get("question_id", ""),
            row.get("strategy") or "",
            int(row["rank"]),
        )
        for row in successful_rows
        if row.get("rank", "").isdigit()
    }

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(successful_rows)

    print(
        f"Reanudando: se conservan {len(successful_rows)} chunks correctos; "
        f"{len(existing_rows) - len(successful_rows)} filas pendientes, fallidas "
        "o de otra versión se volverán a evaluar."
    )
    return done_keys


def create_executor(
    stack: ExitStack,
    device: str,
    include_reason: bool,
) -> ProcessPoolExecutor:
    return stack.enter_context(
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_judge_worker,
            initargs=(device, include_reason),
        )
    )


# ==============================================================================
# 8. MAIN
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
        / f"local_judge_{retrieval_suffix}.csv"
    )

    print("=" * 70)
    print("EVALUACIÓN DE RETRIEVAL MEDIANTE LLM")
    print("=" * 70)
    print(f"Modelo juez: {MODEL_NAME}")
    print(f"Top K: {'todos los del archivo' if args.top_k is None else args.top_k}")

    if not retrieval_results_file.exists():
        raise FileNotFoundError(
            "\nNo se ha encontrado:\n"
            f"{retrieval_results_file}\n\n"
            "Ejecuta primero run_retrieval.py."
        )

    with retrieval_results_file.open("r", encoding="utf-8") as file:
        results = json.load(file)

    print(f"Preguntas encontradas: {len(results)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done_keys = prepare_resume(
        output_file,
        args.restart,
        args.include_reason,
    )

    total = sum(
        1
        for _ in iter_judge_tasks(
            results,
            args.top_k,
            done_keys,
        )
    )
    print(f"Chunks a evaluar: {total}")

    if total == 0:
        print("No quedan chunks pendientes.")
        print(f"Resultados guardados en:\n{output_file}")
        return

    devices = iter_devices(args)
    print(
        f"Workers/modelos: {len(devices)}; dispositivos: {', '.join(devices)}; "
        f"batch por worker: {args.batch_size}; reason: "
        f"{'activado' if args.include_reason else 'desactivado'}"
    )

    task_batches = iter(
        batched(
            iter_judge_tasks(
                results,
                args.top_k,
                done_keys,
                log_warnings=True,
            ),
            args.batch_size,
        )
    )
    batch_count = (total + args.batch_size - 1) // args.batch_size
    start_time = time.perf_counter()
    completed_chunks = 0
    failed_chunks = 0
    next_batch_index = 0
    next_batch_to_write = 0
    buffered_results = {}
    active_futures = {}

    with ExitStack() as stack:
        executors = [
            create_executor(stack, device, args.include_reason)
            for device in devices
        ]

        warmup_futures = [
            executor.submit(_warmup_judge_worker)
            for executor in executors
        ]
        for device, future in zip(devices, warmup_futures):
            try:
                print(f"Modelo cargado en {future.result()}.")
            except Exception as error:
                raise RuntimeError(
                    f"No se pudo cargar {MODEL_NAME} en {device}; "
                    "comprueba la VRAM libre y la disponibilidad del modelo."
                ) from error

        with output_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDNAMES,
                extrasaction="ignore",
            )
            if file.tell() == 0:
                writer.writeheader()

            idle_workers = list(range(len(executors)))
            for worker_index in idle_workers[:]:
                batch = next(task_batches, None)
                if batch is None:
                    break
                future = executors[worker_index].submit(
                    _judge_worker_batch,
                    batch,
                )
                active_futures[future] = (worker_index, next_batch_index)
                next_batch_index += 1
            idle_workers = idle_workers[len(active_futures):]

            max_in_flight = max(1, len(executors) * 2)
            while active_futures:
                completed_futures, _ = wait(
                    active_futures,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed_futures:
                    worker_index, batch_index = active_futures.pop(future)
                    try:
                        buffered_results[batch_index] = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Worker {devices[worker_index]} ha fallado; "
                            "los resultados ya guardados se pueden reanudar."
                        ) from error
                    idle_workers.append(worker_index)

                while next_batch_to_write in buffered_results:
                    rows = buffered_results.pop(next_batch_to_write)
                    for row in rows:
                        writer.writerow(row)
                        completed_chunks += 1
                        if row["judge_error"]:
                            failed_chunks += 1
                            print(
                                f"ERROR question_id={row['question_id']} "
                                f"strategy={row['strategy']} rank={row['rank']}: "
                                f"{row['judge_error']}"
                            )
                    next_batch_to_write += 1
                    file.flush()
                    if completed_chunks % 100 < args.batch_size or completed_chunks == total:
                        elapsed = time.perf_counter() - start_time
                        print(
                            f"Chunks evaluados: {completed_chunks}/{total}; "
                            f"errores: {failed_chunks}; tiempo: {elapsed:.1f} s"
                        )

                while (
                    idle_workers
                    and next_batch_index < batch_count
                    and len(active_futures) + len(buffered_results) < max_in_flight
                ):
                    worker_index = idle_workers.pop()
                    batch = next(task_batches, None)
                    if batch is None:
                        break
                    future = executors[worker_index].submit(
                        _judge_worker_batch,
                        batch,
                    )
                    active_futures[future] = (worker_index, next_batch_index)
                    next_batch_index += 1

    elapsed = time.perf_counter() - start_time
    print()
    print("=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)
    print(
        f"Chunks evaluados: {completed_chunks}; errores guardados: {failed_chunks}; "
        f"tiempo total: {elapsed:.1f} s"
    )
    print(f"Resultados guardados en:\n{output_file}")


# ==============================================================================
# 9. EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()