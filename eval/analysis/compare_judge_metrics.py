"""Compare paired retrieval metrics from GPT and ATLaAI CSV exports."""
import argparse
import csv
from collections import Counter, defaultdict
from math import isfinite, log2
from pathlib import Path
from statistics import mean

CUTOFFS = (3, 5, 7, 10)
METRICS = ("nDCG", "HitRate", "MRR")
JUDGES = ("GPT", "ATLaAI")
BASE = Path("output/judge")


def load(path, judge, issues, cohort=None):
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        fields = reader.fieldnames or []
        missing = {"question_id", "chunk_id", "rank", "relevance_score"} - set(fields)
        if missing or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: cabecera inválida. Columnas faltantes: {sorted(missing)}")
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            if not qid:
                issues.append((judge, "", None, "question_id vacío", reader.line_num))
                continue
            if cohort is not None and qid not in cohort:
                continue
            ranks = groups.setdefault(qid, defaultdict(list))
            try:
                rank = float(row["rank"] or "")
                if not isfinite(rank) or not rank.is_integer() or rank < 1:
                    raise ValueError
                rank = int(rank)
            except ValueError:
                issues.append((judge, qid, None, "rank inválido", reader.line_num))
                ranks[0].append(("", None, False))
                continue
            chunk = (row["chunk_id"] or "").strip()
            error = (row.get("judge_error") or "").strip()
            try:
                score = float(row["relevance_score"] or "")
            except ValueError:
                score = None
            problems = []
            if not chunk:
                problems.append("chunk_id vacío")
            if error:
                problems.append(f"judge_error: {error}")
            if score not in (0, 1, 2):
                problems.append("relevance_score inválido")
            if None in row or any(value is None for value in row.values()):
                problems.append("fila CSV incompleta o con columnas extra")
            for problem in problems:
                issues.append((judge, qid, rank, problem, reader.line_num))
            ranks[rank].append((chunk, score, not problems))
    return groups


def metrics(scores):
    def dcg(values):
        return sum((2 ** score - 1) / log2(rank + 1)
                   for rank, score in enumerate(values, 1))

    ideal = dcg(sorted(scores, reverse=True))
    first = next((rank for rank, score in enumerate(scores, 1) if score == 2), None)
    return dcg(scores) / ideal if ideal else 0, int(first is not None), 1 / first if first else 0


def compare(gpt, local, issues):
    paired = {k: [] for k in CUTOFFS}
    detail = []
    for qid in sorted(gpt):
        failures = set()
        scores = [[], []]
        chunks = [{}, {}]
        rankings = (gpt[qid], local.get(qid, {}))
        for index, (judge, ranks) in enumerate(zip(JUDGES, rankings)):
            if 0 in ranks:
                failures.add(0)
            seen = set()
            for rank, rows in sorted(ranks.items()):
                if rank == 0:
                    continue
                if len(rows) > 1:
                    issues.append((judge, qid, rank, "rank duplicado", len(rows)))
                    failures.add(rank)
                for chunk, _, _ in rows:
                    if chunk and chunk in seen:
                        issues.append((judge, qid, rank, "chunk_id duplicado", chunk))
                        failures.add(rank)
                    seen.add(chunk)
            for rank in range(1, max(CUTOFFS) + 1):
                rows = ranks.get(rank, [])
                if not rows:
                    issues.append((judge, qid, rank, "rank faltante", None))
                    failures.add(rank)
                    continue
                chunk, score, valid = rows[0]
                chunks[index][rank] = chunk
                scores[index].append(score)
                if not valid:
                    failures.add(rank)
        for rank in chunks[0].keys() & chunks[1].keys():
            if chunks[0][rank] != chunks[1][rank]:
                issues.append(("Ambos", qid, rank, "chunk_id distinto", None))
                failures.add(rank)
        for k in CUTOFFS:
            invalid = sorted(rank for rank in failures if rank <= k)
            if invalid:
                detail.append((qid, k, False, ", ".join(
                    str(rank) if rank else "desconocido" for rank in invalid)))
                continue
            left, right = (metrics(values[:k]) for values in scores)
            delta = tuple(a - b for a, b in zip(left, right))
            paired[k].append((left, right, delta))
            detail.append((qid, k, True, "", *left, *right, *delta))
    return paired, detail


def report(out, paired, detail, issues, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xlsxwriter

    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for k, rows in paired.items():
        if rows:
            summaries[k] = [tuple(map(mean, zip(*group))) for group in zip(*rows)]
        else:
            summaries[k] = [(None,) * len(METRICS)] * 3

    with xlsxwriter.Workbook(out / "compare_judge_metrics.xlsx",
                             {"strings_to_formulas": False, "strings_to_urls": False}) as book:
        header = book.add_format({"bold": True, "bg_color": "#203850", "font_color": "#FFFFFF"})
        decimal = book.add_format({"num_format": "0.0000"})

        def sheet(name, columns, rows, first_metric=None):
            ws = book.add_worksheet(name)
            ws.write_row(0, 0, columns, header)
            for number, row in enumerate(rows, 1):
                for index, value in enumerate(row):
                    fmt = decimal if first_metric is not None and index >= first_metric else None
                    ws.write(number, index, value, fmt)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(rows), len(columns) - 1)
            ws.set_column(0, len(columns) - 1, 18)
            ws.set_column(0, 0, 25)
            return ws

        labels = ("k", "Preguntas GPT", "Pares válidos", "Excluidas")
        counts = [(k, total, len(rows), total - len(rows)) for k, rows in paired.items()]
        sheet("Resumen", (*labels, "Juez", *METRICS),
              [(*counts[i], judge, *summaries[k][j])
               for i, k in enumerate(CUTOFFS) for j, judge in enumerate(JUDGES)], 5)
        sheet("GPT menos ATLaAI", (*labels, *METRICS),
              [(*row, *summaries[row[0]][2]) for row in counts], 4)
        sheet("Por pregunta", ("question_id", "k", "Par válido", "Ranks con incidencias",
              *(f"{metric} {label}" for label in (*JUDGES, "delta") for metric in METRICS)),
              detail, 4).set_column(3, 3, 26)
        sheet("Incidencias", ("Juez", "question_id", "rank", "Incidencia", "Detalle"),
              issues).set_column(3, 4, 42)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for index, (metric, axis) in enumerate(zip(METRICS, axes)):
        for j, judge in enumerate(JUDGES):
            positions = [i + (j - 0.5) * 0.36 for i, k in enumerate(CUTOFFS) if paired[k]]
            values = [summaries[k][j][index] for k in CUTOFFS if paired[k]]
            axis.bar(positions, values, width=0.36, label=judge)
        axis.set(title=metric, ylim=(0, 1.05), xticks=range(len(CUTOFFS)),
                 xticklabels=[f"@{k}\nn={len(paired[k])}" for k in CUTOFFS])
    axes[0].legend()
    fig.savefig(out / "medias_pareadas.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout="constrained")
    for i, metric in enumerate(METRICS):
        for j, k in enumerate(CUTOFFS):
            axis = axes[i, j]
            values = [row[2][i] for row in paired[k]]
            if not values:
                axis.text(0.5, 0.5, "Sin pares válidos", ha="center", transform=axis.transAxes)
            elif metric == "HitRate":
                counts_by_delta = Counter(values)
                axis.bar((-1, 0, 1), [counts_by_delta[x] for x in (-1, 0, 1)], width=0.35)
            else:
                axis.hist(values, bins=[(n - 10.5) / 10 for n in range(22)])
            axis.set(title=f"{metric}@{k}", xlim=(-1.08, 1.08), xticks=(-1, 0, 1))
            if j == 0:
                axis.set_ylabel("Preguntas")
    fig.supxlabel("Diferencia por pregunta (GPT − ATLaAI)")
    fig.savefig(out / "distribuciones_delta.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, default=BASE / "all_bge_m3_200_gpt5_6/gpt_judge_bge_m3.csv")
    parser.add_argument("--local", type=Path, default=BASE / "all_bge_m3_atlaai_v1/local_judge_bge_m3.csv")
    parser.add_argument("--out-dir", type=Path, default=BASE / "compare_judge_metrics")
    args = parser.parse_args()
    issues = []
    try:
        gpt = load(args.gpt, "GPT", issues)
        local = load(args.local, "ATLaAI", issues, gpt)
        if len(gpt) != 200:
            issues.append(("GPT", "", None, "Se esperaban 200 question_id únicos", len(gpt)))
        paired, detail = compare(gpt, local, issues)
        report(args.out_dir, paired, detail, issues, len(gpt))
    except (OSError, ValueError, csv.Error, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Preguntas GPT: {len(gpt)}. Incidencias: {len(issues)}.")
    for k, rows in paired.items():
        print(f"k={k}: {len(rows)}/{len(gpt)} pares válidos")
    print(f"Resultados: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
