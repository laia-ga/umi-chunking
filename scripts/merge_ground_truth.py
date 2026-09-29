################################################################################
# GROUND_TRUTH_QA.PY
#
# Une en un único JSONL las preguntas y respuestas de varias bases de
# datos, con un identificador por pregunta que indica su origen.
#
# Archivos de entrada (rutas relativas a la carpeta de este script):
#   - medmcqa/train.jsonl
#   - medmcqa/dev.jsonl
#   - s-medqa/S-MedQA_test.json
#   - s-medqa/S-MedQA_validation.json
#
# Campos de salida según la base de datos de origen:
#
#   MedMCQA  -> question_id   (MEDMCQA-000001, MEDMCQA-000002...)
#               question
#               gold_answer   (texto de la opción correcta según "cop")
#               specialty     ("subject_name")
#               sub-specialty ("topic_name")
#
#   S-MedQA  -> question_id   (SMEDQA-000001, SMEDQA-000002...)
#               question      ("Question")
#               gold_answer   ("Answer")
#               specialty     ("Specialty")
#               sub-specialty (vacío)
#
# Así, todas las preguntas tienen los mismos campos.
#
# La numeración es correlativa dentro de cada base de datos (continúa
# de un archivo al siguiente), así que ningún identificador se repite.
#
# Uso:
#   python ground_truth_qa.py
#   python ground_truth_qa.py --output mi_archivo.jsonl
################################################################################

import argparse
import json
from collections import Counter
from pathlib import Path


# ==============================================================================
# RUTAS
# ==============================================================================

# Carpeta donde está este script (scripts/)
SCRIPTS_DIR = Path(__file__).resolve().parent

# Raíz del proyecto
BASE_DIR = SCRIPTS_DIR.parent

# Carpeta con los datos de ground truth
DATA_DIR = BASE_DIR / "data" / "ground_truth"

MEDMCQA_FILES = [
    DATA_DIR / "medmcqa" / "train.jsonl",
    DATA_DIR / "medmcqa" / "dev.jsonl",
]

SMEDQA_FILES = [
    DATA_DIR / "s-medqa" / "S-MedQA_test.json",
    DATA_DIR / "s-medqa" / "S-MedQA_validation.json",
]

DEFAULT_OUTPUT = "ground_truth_qa.jsonl"


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Prefijos de los identificadores
MEDMCQA_PREFIX = "MEDMCQA"
SMEDQA_PREFIX = "SMEDQA"

# Opciones de MedMCQA en el orden que indica "cop" (1 = opa, 2 = opb...)
MEDMCQA_OPTIONS = ["opa", "opb", "opc", "opd"]


# ==============================================================================
# LECTURA
# ==============================================================================

def load_items(path):

    """
    Lee un archivo con objetos JSON en cualquiera de estos formatos:
      - JSONL (un objeto por línea)
      - objetos seguidos que ocupan varias líneas cada uno
      - una lista JSON [ {...}, {...} ]
    """

    text = path.read_text(encoding="utf-8").strip()

    if text.startswith("["):
        return json.loads(text)

    decoder = json.JSONDecoder()
    items = []
    position = 0

    while position < len(text):

        # Saltar espacios y saltos de línea entre objetos
        while position < len(text) and text[position].isspace():
            position += 1

        if position >= len(text):
            break

        item, position = decoder.raw_decode(text, position)
        items.append(item)

    return items


def clean_text(value):

    """
    Convierte a texto y quita espacios y comillas sobrantes al final
    (algunas preguntas de S-MedQA terminan con una comilla suelta).
    """

    if value is None:
        return ""

    return str(value).strip().rstrip('"').strip()


# ==============================================================================
# CONVERSIÓN DE CADA BASE DE DATOS
# ==============================================================================

def convert_medmcqa(files):

    """
    Convierte los archivos de MedMCQA. La numeración continúa entre
    archivos para que los identificadores no se repitan.
    """

    records = []
    skipped = 0

    for path in files:

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        items = load_items(path)
        before = len(records)

        for index, item in enumerate(items, start=1):

            cop = item.get("cop")

            if not isinstance(cop, int) or not 1 <= cop <= 4:
                print(
                    f"  AVISO: {path.name}, pregunta {index}: "
                    f"cop no válido ({cop!r}), se omite."
                )
                skipped += 1
                continue

            question = clean_text(item.get("question"))
            gold_answer = clean_text(item.get(MEDMCQA_OPTIONS[cop - 1]))

            if not question or not gold_answer:
                print(
                    f"  AVISO: {path.name}, pregunta {index}: "
                    "pregunta o respuesta vacía, se omite."
                )
                skipped += 1
                continue

            records.append({
                "question_id": f"{MEDMCQA_PREFIX}-{len(records) + 1:06d}",
                "question": question,
                "gold_answer": gold_answer,
                "specialty": item.get("subject_name"),
                "sub-specialty": item.get("topic_name") or "",
            })

        print(
            f"  {path.name}: {len(records) - before} de {len(items)} "
            "preguntas convertidas"
        )

    return records, skipped


def convert_smedqa(files):

    """
    Convierte los archivos de S-MedQA. La numeración continúa entre
    archivos para que los identificadores no se repitan.
    """

    records = []
    skipped = 0

    for path in files:

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        items = load_items(path)
        before = len(records)

        for index, item in enumerate(items, start=1):

            question = clean_text(item.get("Question"))
            gold_answer = clean_text(item.get("Answer"))

            if not question or not gold_answer:
                print(
                    f"  AVISO: {path.name}, pregunta {index}: "
                    "pregunta o respuesta vacía, se omite."
                )
                skipped += 1
                continue

            records.append({
                "question_id": f"{SMEDQA_PREFIX}-{len(records) + 1:06d}",
                "question": question,
                "gold_answer": gold_answer,
                "specialty": item.get("Specialty"),
                "sub-specialty": "",
            })

        print(
            f"  {path.name}: {len(records) - before} de {len(items)} "
            "preguntas convertidas"
        )

    return records, skipped


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Une MedMCQA y S-MedQA en un único JSONL."
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Nombre del archivo de salida (por defecto: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    output_file = DATA_DIR / args.output

    print("MedMCQA:")
    medmcqa_records, medmcqa_skipped = convert_medmcqa(MEDMCQA_FILES)

    print("\nS-MedQA:")
    smedqa_records, smedqa_skipped = convert_smedqa(SMEDQA_FILES)

    all_records = medmcqa_records + smedqa_records

    # Comprobación de seguridad: ningún identificador repetido
    duplicated = [
        qid for qid, n in Counter(r["question_id"] for r in all_records).items()
        if n > 1
    ]

    if duplicated:
        raise ValueError(f"Identificadores repetidos: {duplicated[:10]}")

    with open(output_file, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nResumen:")
    print(f"  MedMCQA: {len(medmcqa_records)} preguntas "
          f"({medmcqa_skipped} omitidas)")
    print(f"  S-MedQA: {len(smedqa_records)} preguntas "
          f"({smedqa_skipped} omitidas)")
    print(f"  Total:   {len(all_records)} preguntas")
    print(f"\nGuardado en: {output_file}")


if __name__ == "__main__":
    main()