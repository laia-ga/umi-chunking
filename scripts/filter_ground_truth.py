################################################################################
# FILTER_GROUND_TRUTH.PY
#
# Selecciona las preguntas de un archivo de ground truth cuyas
# combinaciones (specialty, sub-specialty) aparecen en un CSV.
#
# El CSV debe tener, al menos, las columnas:
#   specialty | sub-specialty
#
# Se admite tanto coma como punto y coma como separador (Excel en
# español suele guardar los CSV con punto y coma), y codificación
# UTF-8 o la de Windows (cp1252).
#
# - Para seleccionar preguntas sin sub-specialty, escribe en el CSV
#   "(sin sub-specialty)" o deja la celda vacía.
# - Si el CSV tiene columnas adicionales (por ejemplo, "category"
#   con tus categorías de urología), se añaden a cada pregunta
#   seleccionada. Si una misma combinación aparece en varias filas con
#   valores distintos, se unen con "; ".
# - La comparación ignora mayúsculas y espacios al principio y al
#   final, para tolerar pequeñas diferencias al copiar los nombres.
#
# Uso (desde la raíz del proyecto):
#   python scripts/filter_ground_truth.py
#   python scripts/filter_ground_truth.py --output urologia.jsonl
#   python scripts/filter_ground_truth.py --categories otra_lista.csv
################################################################################

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


# ==============================================================================
# RUTAS
# ==============================================================================

# Carpeta donde está este script (scripts/)
SCRIPTS_DIR = Path(__file__).resolve().parent

# Raíz del proyecto
BASE_DIR = SCRIPTS_DIR.parent

# Carpeta con los datos de ground truth
DATA_DIR = BASE_DIR / "data" / "ground_truth"

DEFAULT_JSONL = "ground_truth_MedMCQA_MedQA.jsonl"
DEFAULT_CATEGORIES = "ground_truth_categories.csv"
DEFAULT_OUTPUT = "ground_truth_filtered.jsonl"

# Texto que representa "sin sub-specialty" en el CSV
EMPTY_LABEL = "(sin sub-specialty)"


# ==============================================================================
# UTILIDADES
# ==============================================================================

def normalize(value):

    """
    Normaliza un valor para comparar: sin espacios sobrantes, en
    minúsculas, y con "(sin sub-specialty)" o vacío como "".
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip()

    if text.lower() == EMPTY_LABEL.lower():
        return ""

    return text.casefold()


def normalize_extra(value):

    """
    Limpia el valor de una columna adicional; devuelve "" si está vacía.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    return str(value).strip()


def read_categories(path):

    """
    Lee el CSV de combinaciones detectando el separador (coma o punto
    y coma) y probando las codificaciones UTF-8 y Windows (cp1252).
    Todas las columnas se leen como texto.
    """

    last_error = None

    for encoding in ("utf-8-sig", "cp1252"):

        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"No se ha podido leer {path} ni como UTF-8 ni como cp1252: {last_error}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Selecciona las preguntas cuyas combinaciones "
            "(specialty, sub-specialty) aparecen en un CSV."
        )
    )

    parser.add_argument("--jsonl", default=DEFAULT_JSONL,
                        help=f"Archivo de preguntas (por defecto: {DEFAULT_JSONL})")
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES,
                        help=f"CSV con las combinaciones (por defecto: {DEFAULT_CATEGORIES})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Archivo de salida (por defecto: {DEFAULT_OUTPUT})")

    args = parser.parse_args()

    jsonl_file = DATA_DIR / args.jsonl
    categories_file = DATA_DIR / args.categories
    output_file = DATA_DIR / args.output

    for path in (jsonl_file, categories_file):
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

    # --------------------------------------------------------------------------
    # Leer el CSV de combinaciones
    # --------------------------------------------------------------------------

    categories = read_categories(categories_file)
    categories.columns = [str(c).strip() for c in categories.columns]

    required = {"specialty", "sub-specialty"}
    missing = required - set(categories.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV: {sorted(missing)}. "
            f"Columnas encontradas: {list(categories.columns)}"
        )

    extra_columns = [c for c in categories.columns if c not in required]

    # Combinación normalizada -> {columna extra: set de valores}
    selected = {}
    original_names = {}

    for _, row in categories.iterrows():

        if normalize(row["specialty"]) == "":
            continue

        key = (normalize(row["specialty"]), normalize(row["sub-specialty"]))

        original_names[key] = (
            str(row["specialty"]).strip(),
            str(row["sub-specialty"]).strip()
            if normalize(row["sub-specialty"]) else EMPTY_LABEL,
        )

        extras = selected.setdefault(key, {c: set() for c in extra_columns})

        for column in extra_columns:
            value = normalize_extra(row[column])
            if value:
                extras[column].add(value)

    print(f"Combinaciones en el CSV: {len(selected)}")

    if extra_columns:
        print(f"Columnas adicionales que se añadirán: {extra_columns}")

    # --------------------------------------------------------------------------
    # Filtrar las preguntas
    # --------------------------------------------------------------------------

    matches = Counter()
    kept = []
    total = 0

    with open(jsonl_file, encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            key = (
                normalize(record.get("specialty")),
                normalize(record.get("sub-specialty")),
            )

            if key not in selected:
                continue

            for column, values in selected[key].items():
                record[column] = "; ".join(sorted(values))

            kept.append(record)
            matches[key] += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------------------
    # Resumen
    # --------------------------------------------------------------------------

    print(f"Preguntas leídas: {total}")
    print(f"Preguntas seleccionadas: {len(kept)}\n")

    print("Preguntas por combinación:")

    for key in sorted(selected, key=lambda k: -matches[k]):
        specialty, sub_specialty = original_names[key]
        print(f"  {matches[key]:>6}  {specialty} | {sub_specialty}")

    not_found = [original_names[k] for k in selected if matches[k] == 0]

    if not_found:
        print(
            f"\nAVISO: {len(not_found)} combinaciones del CSV no tienen "
            "ninguna pregunta (revisa si están bien escritas):"
        )
        for specialty, sub_specialty in not_found:
            print(f"  - {specialty} | {sub_specialty}")

    print(f"\nGuardado en: {output_file}")


if __name__ == "__main__":
    main()