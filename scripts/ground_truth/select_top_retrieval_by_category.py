
"""Asigna categoría a cada pregunta y selecciona el máximo número equilibrado
de preguntas por categoría con score superior a un umbral.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize(value):
    return " ".join(str(value or "").strip().casefold().split())


def read_categories(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError("El CSV no contiene cabecera")
        fields = {normalize(x): x for x in reader.fieldnames}
        required = ("specialty", "sub-specialty", "category")
        missing = [x for x in required if x not in fields]
        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {missing}. "
                f"Encontradas: {reader.fieldnames}"
            )
        mapping = {}
        for line, row in enumerate(reader, 2):
            key = (
                normalize(row[fields["specialty"]]),
                normalize(row[fields["sub-specialty"]])
            )
            category = str(row[fields["category"]] or "").strip()
            if not key[0] or not category:
                raise ValueError(
                    f"Fila {line}: specialty y category deben tener valor "
                    "(sub-specialty puede estar vacío)"
                )
            if key in mapping and mapping[key] != category:
                raise ValueError(
                    f"Correspondencia contradictoria en fila {line}: {key}"
                )
            mapping[key] = category
    return mapping


def read_results(path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                "La salida JSON del retrieval debe ser una lista de preguntas"
            )
        yield from data


def best_score(question):
    hits = question.get("retrieved_chunks")
    if hits is None and isinstance(question.get("strategies"), dict):
        hits = [
            h
            for hs in question["strategies"].values()
            for h in hs
        ]
    if not hits:
        return None
    scores = [
        float(h["score"])
        for h in hits
        if h.get("score") is not None
    ]
    return max(scores) if scores else None


def main():
    parser = argparse.ArgumentParser(
        description="Selección equilibrada de preguntas por categoría y score"
    )
    parser.add_argument(
        "--retrieval", type=Path, required=True,
        help="JSON o JSONL de salida del retrieval"
    )
    parser.add_argument(
        "--categories", type=Path, required=True,
        help="CSV specialty, sub-specialty, category"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="JSONL de salida"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.60,
        help="Score mínimo exclusivo (por defecto: 0.60)"
    )
    args = parser.parse_args()

    mapping = read_categories(args.categories)
    categories = sorted(set(mapping.values()))

    if len(categories) != 4:
        raise ValueError(
            f"Se esperaban 4 categorías en el CSV; "
            f"encontradas {len(categories)}: {categories}"
        )

    grouped = defaultdict(list)
    unmatched = Counter()
    no_hits = Counter()
    below_threshold = Counter()
    seen_ids = set()

    for question in read_results(args.retrieval):
        qid = question.get("question_id")
        if not qid or qid in seen_ids:
            raise ValueError(
                f"ID de pregunta ausente o duplicado: {qid!r}"
            )
        seen_ids.add(qid)

        key = (
            normalize(question.get("specialty")),
            normalize(question.get("sub-specialty"))
        )
        category = mapping.get(key)

        if category is None:
            unmatched[key] += 1
            continue

        score = best_score(question)

        if score is None:
            no_hits[category] += 1
            continue

        # Solo preguntas con score estrictamente superior al umbral
        if score <= args.min_score:
            below_threshold[category] += 1
            continue

        updated = dict(question)
        updated.pop("specialty", None)
        updated.pop("sub-specialty", None)
        updated["category"] = category

        grouped[category].append((score, str(qid), updated))

    print(f"\nPreguntas con score > {args.min_score} por categoría:")

    for category in categories:
        print(
            f"  {category}: {len(grouped[category])} "
            f"(score insuficiente: {below_threshold[category]}, "
            f"sin hits: {no_hits[category]})"
        )

    if unmatched:
        print(
            f"\nCombinaciones sin correspondencia: "
            f"{sum(unmatched.values())} preguntas"
        )
        for key, count in unmatched.most_common(10):
            print(f"  {key}: {count}")

    # Máximo número de preguntas igual para todas las categorías
    per_category = min(len(grouped[c]) for c in categories)

    if per_category == 0:
        raise ValueError(
            f"Alguna categoría no tiene preguntas con score > "
            f"{args.min_score}. No se ha escrito ningún archivo."
        )

    print(f"\nPreguntas seleccionadas por categoría: {per_category}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for category in categories:
            selected = sorted(
                grouped[category],
                key=lambda x: (-x[0], x[1])
            )[:per_category]

            for _, _, question in selected:
                f.write(
                    json.dumps(question, ensure_ascii=False) + "\n"
                )

    total = len(categories) * per_category

    print(f"\nGuardadas {total} preguntas en {args.output}")


if __name__ == "__main__":
    main()

