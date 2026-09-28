################################################################################
# DOWNLOAD_MEDQA.PY
#
# Descarga la parte en inglés (USMLE) de MedQA desde Hugging Face,
# la guarda en JSONL (un archivo por split) y muestra los grupos
# disponibles.
#
# MedQA solo tiene un campo de clasificación:
#   meta_info: parte del examen USMLE ("step1" o "step2&3")
#     - step1:   ciencias básicas (fisiología, farmacología, patología...)
#     - step2&3: ciencias clínicas (diagnóstico, manejo, tratamiento)
#
# Para clasificar por especialidad clínica, usa S-MedQA
# (ver list_smedqa_specialties.py).
#
# Salida (en la misma carpeta que el script):
#   medqa_train.jsonl, medqa_dev.jsonl, medqa_test.jsonl
################################################################################

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATA_DIR = Path(__file__).resolve().parent

HF_DATASET = "GBaker/MedQA-USMLE-4-options"

# Nombres de split en Hugging Face -> nombre del archivo de salida
SPLITS = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


def main():

    dataset = load_dataset(HF_DATASET)

    print(f"Splits disponibles: {list(dataset.keys())}")

    frames = []

    for hf_split, name in SPLITS.items():

        if hf_split not in dataset:
            continue

        df = dataset[hf_split].to_pandas()
        df["split"] = name

        output_file = DATA_DIR / f"medqa_{name}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        print(f"  {name}: {len(df)} preguntas -> {output_file}")

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    print(f"\nColumnas: {list(data.columns)}")

    # ------------------------------------------------------------------
    # Grupos disponibles
    # ------------------------------------------------------------------

    print(f"\nTotal de preguntas: {len(data)}")

    if "meta_info" in data.columns:

        print("\nPreguntas por parte del examen (meta_info) y split:")
        print(
            pd.crosstab(data["meta_info"], data["split"], margins=True)
            .to_string()
        )

    # ------------------------------------------------------------------
    # Ejemplo
    # ------------------------------------------------------------------

    example = data.iloc[0]

    print("\nEjemplo:")
    print(f"  Pregunta: {str(example['question'])[:400]}...")
    print(f"  Respuesta: {example.get('answer')}")


if __name__ == "__main__":
    main()