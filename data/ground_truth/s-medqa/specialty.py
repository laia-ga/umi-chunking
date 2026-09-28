################################################################################
# COUNT_SMEDQA_SPECIALTIES.PY
#
# Lee uno o varios archivos de S-MedQA y muestra la lista de
# especialidades ("Specialty") con el número de preguntas de cada una.
#
# Uso (archivos en la misma carpeta que el script):
#   python count_smedqa_specialties.py S-MedQA_train.json
#   python count_smedqa_specialties.py S-MedQA_train.json S-MedQA_validation.json S-MedQA_test.json
################################################################################

import argparse
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def main():

    parser = argparse.ArgumentParser(
        description="Cuenta las preguntas por especialidad en S-MedQA."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Uno o varios archivos JSON de S-MedQA",
    )

    args = parser.parse_args()

    total_counts = Counter()
    total_questions = 0

    for name in args.files:

        path = DATA_DIR / name

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        with open(path, encoding="utf-8") as f:
            items = json.load(f)

        counts = Counter()

        for item in items:

            specialty = item.get("Specialty")

            # Por si alguna pregunta tiene varias especialidades en una lista
            specialties = specialty if isinstance(specialty, list) else [specialty]

            for s in specialties:
                counts[s if s else "(sin especialidad)"] += 1

        total_counts.update(counts)
        total_questions += len(items)

        print("=" * 60)
        print(f"{name}: {len(items)} preguntas, {len(counts)} especialidades")
        print("=" * 60)

        for specialty, n in counts.most_common():
            print(f"  {n:>6}  {specialty}")

        print()

    if len(args.files) > 1:

        print("=" * 60)
        print(
            f"TOTAL: {total_questions} preguntas, "
            f"{len(total_counts)} especialidades"
        )
        print("=" * 60)

        for specialty, n in total_counts.most_common():
            print(f"  {n:>6}  {specialty}")


if __name__ == "__main__":
    main()