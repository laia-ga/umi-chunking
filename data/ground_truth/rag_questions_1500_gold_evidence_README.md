# RAG chunking benchmark — 1,500 questions with gold evidence

This benchmark contains 100 questions for each of the 15 supplied source JSON documents.
It is designed to evaluate whether a retriever returns chunks containing the information needed to answer each question.

## Primary file

`rag_questions_1500_gold_evidence.jsonl`

Each JSONL record includes:

- `question_id`: stable benchmark ID.
- `question`: query to send to the retriever.
- `source_file`: source document containing the answer.
- `target_section`: original logical target.
- `evaluation_tier`:
  - `strict`: precise evidence; recommended for the primary chunking metric.
  - `multi_evidence`: answer requires multiple evidence atoms or sections.
  - `section_scope`: broad/section-level probe; useful diagnostically, but not ideal for a strict Recall@K headline metric.
- `gold_source_paths`: canonical source paths in the source JSON.
- `gold_evidence`: flattened list of exact source-supported evidence atoms (`source_path`, `quote`).
- `gold_evidence_groups`: grouped evidence and the within-group operator.
- `retrieval_rule.across_groups_operator`: whether evidence groups are conjunctive (`all`) or not.
- `evidence_requirement`:
  - `any_atom`: one evidence atom is sufficient.
  - `all_atoms`: all atoms are required (typically enumerations).
  - `all_groups_any_atom`: at least one atom from every evidence group is required (cross-section queries).
- `gold_answer`: convenience rendering of the gold evidence; retrieval evaluation should primarily use `gold_evidence` / `gold_evidence_groups`.
- `gold_evidence_sha256`: hashes of normalized evidence text for reproducibility.

## Recommended retrieval scoring

Primary benchmark: filter to `evaluation_tier == "strict"`.

For each query and Top-K retrieved chunks:

1. Normalize whitespace/case in retrieved text and evidence.
2. `any_atom`: success if at least one gold evidence atom is covered by the Top-K chunks.
3. `all_atoms`: success if every gold atom is covered somewhere in the Top-K set.
4. `all_groups_any_atom`: success if each evidence group has at least one covered atom in the Top-K set.

For chunk boundaries, exact full-string matching can be overly strict. A practical alternative is token-level evidence coverage (for example >= 80% of an evidence atom) or a normalized substring/containment test in either direction, while keeping the threshold fixed across chunking strategies.

Useful metrics include Hit/Recall@1, @3, @5, MRR for `strict`, plus group recall for `multi_evidence`.

## Source-support audit

Every emitted gold evidence quote was validated as an exact substring of a scalar value in its corresponding source JSON.
Five prompts from the original query-only set were replaced because the original formulation targeted null/empty or semantically unsupported fields. Their original wording/path is retained in `original_question` / `original_target_section` on those records.

## Benchmark size

- Total records: 1,500
- Documents: 15
- Questions per document: 100
- `strict`: 1,300
- `multi_evidence`: 109
- `section_scope`: 91
