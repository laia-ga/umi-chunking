################################################################################
# CLEAN_QUESTIONS.PY
#
# Limpia el fichero de preguntas (ground truth) antes de lanzar el retrieval:
#   1. Quita preguntas "except / false / not true / incorrect" (sin opciones
#      visibles no se pueden responder desde un chunk) y respuestas "all of the
#      above".
#   2. Quita casi-duplicados (preguntas casi idénticas redactadas distinto):
#        - si las respuestas coinciden (o solo difieren por una errata), se
#          conserva la primera;
#        - si las respuestas son DISTINTAS (pregunta ambigua o inconsistente),
#          se descarta el grupo entero.
#   3. Elimina el campo "retrieved_chunks" (el top-1 usado para filtrar), para
#      que no se cuele en el retrieval ni en el juez.
#
# Uso:
#   python data\ground_truth\clean_questions.py --input data\ground_truth
#   \4n_questions.jsonl --output data\ground_truth\questions_clean.jsonl

################################################################################

import argparse
import collections
import difflib
import json
import re
from pathlib import Path

EXCEPT_PATTERN = re.compile(
    r"\bexcept\b|\bnot true\b|\bincorrect\b|\bfalse\b", re.IGNORECASE
)
DROP_FIELDS = {"retrieved_chunks"}


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).lower())
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def similar(a: str, b: str, threshold: float) -> bool:
    if abs(len(a) - len(b)) > max(40, 0.15 * max(len(a), len(b))):
        return False
    matcher = difflib.SequenceMatcher(None, a, b)
    return matcher.quick_ratio() > threshold and matcher.ratio() > threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-except", action="store_true",
                        help="No elimina las preguntas except/false.")
    parser.add_argument("--question-threshold", type=float, default=0.9,
                        help="Similitud mínima entre preguntas para considerarlas casi duplicadas.")
    parser.add_argument("--answer-threshold", type=float, default=0.85,
                        help="Similitud mínima entre respuestas para considerarlas la misma.")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Preguntas de entrada: {len(rows)}")

    # 1) except / false / all of the above
    kept = []
    removed_except = removed_above = 0
    for row in rows:
        if "above" in str(row.get("gold_answer", "")).lower():
            removed_above += 1
            continue
        if not args.keep_except and EXCEPT_PATTERN.search(row["question"]):
            removed_except += 1
            continue
        kept.append(row)
    print(f"Eliminadas por 'except/false/not true/incorrect': {removed_except}")
    print(f"Eliminadas por respuesta 'all of the above': {removed_above}")

    # 2) casi-duplicados dentro de cada categoría
    by_category = collections.defaultdict(list)
    for row in kept:
        by_category[row.get("category", "")].append(row)

    final = []
    dropped_same = dropped_conflict = conflict_groups = 0
    for category, items in by_category.items():
        questions = [normalize(r["question"]) for r in items]
        answers = [normalize(r["gold_answer"]) for r in items]
        used = set()
        for i in range(len(items)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(items)):
                if j not in used and similar(questions[i], questions[j], args.question_threshold):
                    group.append(j)
                    used.add(j)
            used.add(i)
            if len(group) == 1:
                final.append(items[i])
                continue
            same_answer = all(
                similar(answers[i], answers[k], args.answer_threshold) or answers[i] == answers[k]
                for k in group[1:]
            )
            if same_answer:
                final.append(items[i])
                dropped_same += len(group) - 1
            else:
                conflict_groups += 1
                dropped_conflict += len(group)
    print(f"Casi-duplicados con la misma respuesta eliminados: {dropped_same}")
    print(f"Grupos con respuestas contradictorias descartados: {conflict_groups} "
          f"({dropped_conflict} preguntas)")

    # 3) fuera el top-1 filtrado
    cleaned = [{k: v for k, v in r.items() if k not in DROP_FIELDS} for r in final]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as file:
        for row in cleaned:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nPreguntas finales: {len(cleaned)}")
    for category, count in sorted(collections.Counter(r.get("category", "") for r in cleaned).items()):
        print(f"  {category}: {count}")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()