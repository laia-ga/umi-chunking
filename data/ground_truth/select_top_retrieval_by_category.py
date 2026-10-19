#!/usr/bin/env python3
"""Asigna categoría a cada pregunta y selecciona top N por score de sus chunks."""
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
            raise ValueError(f"Faltan columnas en el CSV: {missing}. Encontradas: {reader.fieldnames}")
        mapping = {}
        for line, row in enumerate(reader, 2):
            key = (normalize(row[fields["specialty"]]), normalize(row[fields["sub-specialty"]]))
            category = str(row[fields["category"]] or "").strip()
            if not key[0] or not category:
                raise ValueError(f"Fila {line}: specialty y category deben tener valor (sub-specialty puede estar vacío)")
            if key in mapping and mapping[key] != category:
                raise ValueError(f"Correspondencia contradictoria en fila {line}: {key}")
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
            raise ValueError("La salida JSON del retrieval debe ser una lista de preguntas")
        yield from data


def best_score(question):
    hits = question.get("retrieved_chunks")
    if hits is None and isinstance(question.get("strategies"), dict):
        hits = [h for hs in question["strategies"].values() for h in hs]
    if not hits:
        return None
    scores = [float(h["score"]) for h in hits if h.get("score") is not None]
    return max(scores) if scores else None


def main():
    parser = argparse.ArgumentParser(description="Top N preguntas por categoría según mejor score de retrieval")
    parser.add_argument("--retrieval", type=Path, required=True, help="JSON o JSONL de salida del retrieval")
    parser.add_argument("--categories", type=Path, required=True, help="CSV specialty, sub-specialty, category")
    parser.add_argument("--output", type=Path, required=True, help="JSONL de salida")
    parser.add_argument("--per-category", type=int, default=1000)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category debe ser positivo")

    mapping = read_categories(args.categories)
    categories = sorted(set(mapping.values()))
    if len(categories) != 4:
        raise ValueError(f"Se esperaban 4 categorías en el CSV; encontradas {len(categories)}: {categories}")

    grouped = defaultdict(list)
    unmatched = Counter()
    no_hits = Counter()
    seen_ids = set()
    for question in read_results(args.retrieval):
        qid = question.get("question_id")
        if not qid or qid in seen_ids:
            raise ValueError(f"ID de pregunta ausente o duplicado: {qid!r}")
        seen_ids.add(qid)
        key = (normalize(question.get("specialty")), normalize(question.get("sub-specialty")))
        category = mapping.get(key)
        if category is None:
            unmatched[key] += 1
            continue
        score = best_score(question)
        if score is None:
            no_hits[category] += 1
            continue
        updated = dict(question)
        updated.pop("specialty", None)
        updated.pop("sub-specialty", None)
        updated["category"] = category
        grouped[category].append((score, str(qid), updated))

    print("Preguntas con score por categoría:")
    for category in categories:
        print(f"  {category}: {len(grouped[category])} (sin hits: {no_hits[category]})")
    if unmatched:
        print(f"Combinaciones sin correspondencia: {sum(unmatched.values())} preguntas")
        for key, count in unmatched.most_common(10):
            print(f"  {key}: {count}")

    insufficient = {c: len(grouped[c]) for c in categories if len(grouped[c]) < args.per_category}
    if insufficient:
        raise ValueError(f"No hay {args.per_category} preguntas con score en cada categoría: {insufficient}. No se ha escrito ningún archivo.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for category in categories:
            selected = sorted(grouped[category], key=lambda x: (-x[0], x[1]))[:args.per_category]
            for _, _, question in selected:
                f.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(f"Guardadas {len(categories) * args.per_category} preguntas en {args.output}")


if __name__ == "__main__":
    main()
