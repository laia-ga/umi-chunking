# umi-chunking

Pipeline para chunking, indexación y evaluación de recuperación de
información con enfoque médico.

## Proyecto

Este repositorio incluye:

- chunking sobre documentos JSON y Markdown;
- configuración para indexación y recuperación;
- scripts de evaluación y métricas;
- generación de ground truth reproducible para benchmarking RAG.

## Ground truth

La parte de ground truth permite:

- seleccionar documentos equilibrados por especialidad;
- mantener una muestra reproducible con semilla fija;
- generar preguntas con OpenAI a partir de esos documentos;
- exportar una versión limpia y compartible del dataset.

### Archivos clave

- [data/ground_truth/select_documents_for_question_generation.py](data/ground_truth/select_documents_for_question_generation.py)
- [data/ground_truth/generate_questions_from_selected_documents.py](data/ground_truth/generate_questions_from_selected_documents.py)
- [data/ground_truth/four_specialty_groups.json](data/ground_truth/four_specialty_groups.json)

## Uso rápido

```bash
python data/ground_truth/select_documents_for_question_generation.py \
  --docs-per-specialty 5 \
  --question-total 200 \
  --seed 42

python data/ground_truth/generate_questions_from_selected_documents.py --dry-run

python data/ground_truth/generate_questions_from_selected_documents.py
```

## Estructura principal

- [chunking](chunking)
- [configs](configs)
- [data](data)
- [eval](eval)
- [scripts](scripts)
