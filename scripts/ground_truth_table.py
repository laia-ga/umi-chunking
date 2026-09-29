################################################################################
# COUNT_GROUND_TRUTH.PY
#
# Muestra el número de preguntas por "specialty" y "sub-specialty" de un
# archivo de ground truth, desglosado por base de datos de origen
# (el prefijo del question_id: MEDMCQA, SMEDQA...).
#
# Además de mostrarlo por consola, guarda la tabla en un CSV junto al
# archivo de entrada para poder revisarla y filtrarla en Excel.
#
# Uso (desde la raíz del proyecto):
#   python scripts/count_ground_truth.py
#   python scripts/count_ground_truth.py otro_archivo.jsonl
################################################################################

import argparse
import json
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

DEFAULT_FILE = "ground_truth_MedMCQA_MedQA.jsonl"

# Etiqueta para las preguntas sin sub-specialty
EMPTY = "(sin sub-specialty)"


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Cuenta preguntas por specialty y sub-specialty."
    )

    parser.add_argument(
        "file",
        nargs="?",
        default=DEFAULT_FILE,
        help=f"Archivo JSONL en data/ground_truth (por defecto: {DEFAULT_FILE})",
    )

    args = parser.parse_args()

    input_file = DATA_DIR / args.file

    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_file}")

    # --------------------------------------------------------------------------
    # Cargar
    # --------------------------------------------------------------------------

    with open(input_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows)

    df["source"] = df["question_id"].str.split("-").str[0]
    df["specialty"] = df["specialty"].fillna("(sin specialty)")
    df["sub-specialty"] = (
        df["sub-specialty"].fillna("").replace("", EMPTY)
    )

    # --------------------------------------------------------------------------
    # Tabla specialty × sub-specialty, con columnas por base de datos
    # --------------------------------------------------------------------------

    table = pd.crosstab(
        [df["specialty"], df["sub-specialty"]],
        df["source"],
    )

    table["total"] = table.sum(axis=1)

    specialty_totals = df["specialty"].value_counts()

    table = table.reset_index()
    table.insert(1, "specialty_total", table["specialty"].map(specialty_totals))

    table = table.sort_values(
        ["specialty_total", "specialty", "total"],
        ascending=[False, True, False],
    )

    # --------------------------------------------------------------------------
    # Mostrar por consola
    # --------------------------------------------------------------------------

    sources = sorted(df["source"].unique())

    print(f"Archivo: {input_file.name}")
    print(f"Total de preguntas: {len(df)}")

    for source in sources:
        print(f"  {source}: {int((df['source'] == source).sum())}")

    print(f"Specialties distintas: {df['specialty'].nunique()}")

    for specialty, n in specialty_totals.items():

        block = table[table["specialty"] == specialty]

        by_source = ", ".join(
            f"{s}: {int(block[s].sum())}" for s in sources if block[s].sum() > 0
        )

        print()
        print("=" * 70)
        print(f"{specialty}  ({n} preguntas; {by_source})")
        print("=" * 70)

        for _, row in block.iterrows():
            print(f"  {int(row['total']):>7}  {row['sub-specialty']}")

    # --------------------------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------------------------

    output_file = DATA_DIR / f"{input_file.stem}_specialties.csv"

    table.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nTabla guardada en: {output_file}")


if __name__ == "__main__":
    main()