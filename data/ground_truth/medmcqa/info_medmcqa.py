################################################################################
# LIST_SUBJECTS.PY
#
# Lista todas las materias (subject_name) de un archivo de MedMCQA y,
# dentro de cada una, todos sus temas (topic_name), con el número de
# preguntas de cada materia y de cada tema.
#
# Además de mostrarlo por consola, guarda la tabla materia × tema en
# un CSV para poder revisarla y filtrarla en Excel.
#
# Uso:
#   python list_subjects.py                 (usa dev.jsonl por defecto)
#   python list_subjects.py train.jsonl     (otro archivo de la misma carpeta)
################################################################################

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent

# Etiqueta para las preguntas sin tema asignado
NO_TOPIC = "(sin tema)"


def main():

    parser = argparse.ArgumentParser(
        description="Lista las subject_name y sus topic_name de un archivo de MedMCQA."
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="train.jsonl",
        help="Archivo JSONL de MedMCQA en la misma carpeta que el script "
             "(por defecto: dev.jsonl)",
    )

    args = parser.parse_args()

    input_file = DATA_DIR / args.file

    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_file}")

    df = pd.read_json(input_file, lines=True)

    df["topic_name"] = df["topic_name"].fillna(NO_TOPIC)

    # ------------------------------------------------------------------
    # Recuento por materia (para ordenar de mayor a menor)
    # ------------------------------------------------------------------

    subject_counts = df["subject_name"].value_counts()

    # ------------------------------------------------------------------
    # Recuento por materia × tema
    # ------------------------------------------------------------------

    topic_counts = (
        df.groupby(["subject_name", "topic_name"])
        .size()
        .reset_index(name="questions")
    )

    # ------------------------------------------------------------------
    # Mostrar por consola
    # ------------------------------------------------------------------

    print(f"Archivo: {input_file.name}")
    print(f"Total de preguntas: {len(df)}")
    print(f"Materias distintas: {df['subject_name'].nunique()}")
    print(
        f"Temas distintos: "
        f"{df.loc[df['topic_name'] != NO_TOPIC, 'topic_name'].nunique()}"
    )

    for subject, n_questions in subject_counts.items():

        topics = (
            topic_counts[topic_counts["subject_name"] == subject]
            .sort_values("questions", ascending=False)
        )

        n_topics = (topics["topic_name"] != NO_TOPIC).sum()

        print()
        print("=" * 70)
        print(f"{subject}  ({n_questions} preguntas, {n_topics} temas)")
        print("=" * 70)

        for _, row in topics.iterrows():
            print(f"  {row['questions']:>6}  {row['topic_name']}")

    # ------------------------------------------------------------------
    # Guardar CSV materia × tema
    # ------------------------------------------------------------------

    topic_counts["subject_total"] = topic_counts["subject_name"].map(subject_counts)

    topic_counts = topic_counts.sort_values(
        ["subject_total", "subject_name", "questions"],
        ascending=[False, True, False],
    )[["subject_name", "subject_total", "topic_name", "questions"]]

    output_file = DATA_DIR / f"{input_file.stem}_subjects_topics.csv"

    topic_counts.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nTabla materia × tema guardada en: {output_file}")


if __name__ == "__main__":
    main()