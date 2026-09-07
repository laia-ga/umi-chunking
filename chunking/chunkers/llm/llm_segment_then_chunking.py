from typing import Callable, Dict, List, Optional

import nltk

from ..base import BaseChunker, Chunk
from .llm_client import LLMClient


class LLMSegmentThenChunker(BaseChunker):
    """
    Segmenta primero el documento en unidades semánticas y después
    convierte esas unidades en chunks con límite de tamaño.

    Estrategia:
    1. Segmentación estructural por párrafos.
    2. Refinamiento con LLM de los segmentos grandes.
    3. División adicional si una unidad supera max_chars.
    4. Aplicación opcional de overlap.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        llm: Optional[LLMClient] = None,
        overlap_tokens: int = 75,
        enable_llm_refinement: bool = True,
        max_tokens: int = 512,
    ):
        if max_tokens <= 0:
            raise ValueError(
                "max_chars debe ser mayor que 0."
            )

        if overlap_tokens < 0:
            raise ValueError(
                "overlap no puede ser negativo."
            )

        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap debe ser menor que max_tokens."
            )

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función."
            )

        self.token_counter = token_counter
        self.llm = llm or LLMClient()

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.enable_llm_refinement = enable_llm_refinement

        self.llm_max_tokens = max_tokens

    # ======================================================
    # API pública
    # ======================================================

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        text = self._normalize(text)

        # Primera segmentación estructural
        base_segments = self._heuristic_segments(text)

        # Refinamiento opcional con LLM
        semantic_segments = self._refine_segments(
            base_segments
        )

        # Convertir segmentos semánticos en chunks
        chunk_parts: List[Dict] = []

        for segment in semantic_segments:
            parts = self._split_to_max(
                segment["text"]
            )

            for part_index, part in enumerate(parts):
                part = part.strip()

                if not part:
                    continue

                chunk_parts.append(
                    {
                        "text": part,
                        "unit_type": segment["type"],
                        "source_segment_index": (
                            segment["segment_index"]
                        ),
                        "segment_part_index": part_index,
                        "used_llm_refinement": segment[
                            "used_llm_refinement"
                        ],
                    }
                )

        chunk_parts = self._apply_overlap(
            chunk_parts
        )

        chunks = []

        for index, part in enumerate(chunk_parts):
            chunk_text = part["text"].strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "llm_segment_then_chunking"
                        ),
                        "max_tokens": self.max_tokens,
                        "overlap_tokens": self.overlap_tokens,
                        "unit_type": part["unit_type"],
                        "source_segment_index": part[
                            "source_segment_index"
                        ],
                        "segment_part_index": part[
                            "segment_part_index"
                        ],
                        "used_llm_refinement": part[
                            "used_llm_refinement"
                        ],
                        "chunk_index": index,
                        "token_count": self.token_counter(chunk_text),
                        "character_count": len(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{index:04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks

    # ======================================================
    # Normalización
    # ======================================================

    def _normalize(self, text: str) -> str:
        """
        Normaliza saltos de línea y conserva la separación
        entre párrafos.
        """

        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        lines = [
            line.rstrip()
            for line in text.split("\n")
        ]

        text = "\n".join(lines)

        while "\n\n\n" in text:
            text = text.replace(
                "\n\n\n",
                "\n\n",
            )

        return text.strip()

    # ======================================================
    # Segmentación estructural inicial
    # ======================================================

    def _heuristic_segments(
        self,
        text: str,
    ) -> List[Dict]:
        """
        Segmenta por párrafos.
        """

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        segments = []

        for index, paragraph in enumerate(paragraphs):
            segments.append(
                {
                    "text": paragraph,
                    "type": self._detect_unit_type(
                        paragraph
                    ),
                    "segment_index": index,
                    "used_llm_refinement": False,
                }
            )

        return segments

    def _detect_unit_type(
        self,
        text: str,
    ) -> str:
        """
        Clasificación estructural sencilla del segmento.
        """

        stripped = text.strip()

        if stripped.startswith("```"):
            return "code"

        if stripped.startswith("#"):
            return "heading_block"

        lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip()
        ]

        if lines and all(
            line.startswith(("-", "*"))
            or self._starts_with_numbered_item(line)
            for line in lines
        ):
            return "list"


        if ":" in stripped:
            return "structured_paragraph"

        return "paragraph"

    def _starts_with_numbered_item(
        self,
        text: str,
    ) -> bool:
        """
        Detecta líneas que comienzan con elementos como:
        1. Texto
        2. Texto
        """

        first_token = text.split(
            maxsplit=1
        )[0]

        if not first_token.endswith("."):
            return False

        return first_token[:-1].isdigit()

    # ======================================================
    # Refinamiento con LLM
    # ======================================================

    def _refine_segments(
        self,
        segments: List[Dict],
    ) -> List[Dict]:
        """
        Envía al LLM únicamente los segmentos que superan
        llm_refine_threshold.
        """

        refined_segments = []

        for segment in segments:
            text = segment["text"]

            should_refine = (
                self.enable_llm_refinement
            )

            if not should_refine:
                refined_segments.append(segment)
                continue

            llm_parts = self._ask_llm_for_segments(
                text
            )

            if not llm_parts:
                refined_segments.append(segment)
                continue

            for part_index, part in enumerate(llm_parts):
                refined_segments.append(
                    {
                        "text": part["text"],
                        "type": part.get(
                            "type",
                            segment["type"],
                        ),
                        "segment_index": (
                            segment["segment_index"]
                        ),
                        "llm_segment_index": part_index,
                        "used_llm_refinement": True,
                    }
                )

        return refined_segments

    def _ask_llm_for_segments(
        self,
        text: str,
    ) -> Optional[List[Dict]]:
        """
        Solicita al LLM límites entre frases.

        Se utilizan índices de frases para evitar que el modelo
        tenga que calcular posiciones exactas de caracteres.
        """

        sentences = self._sentence_tokenize(text)

        if len(sentences) < 2:
            return None

        numbered_sentences = "\n".join(
            f"{index}: {sentence}"
            for index, sentence in enumerate(sentences)
        )

        prompt = f"""
Eres un sistema de segmentación de documentación clínica.

Divide las frases numeradas en unidades semánticas coherentes.

Reglas:
- Mantén exactamente el orden original.
- No elimines ni reescribas ninguna frase.
- Cada unidad debe contener frases consecutivas.
- Separa cuando cambie claramente la temática o la sección.
- Evita unidades excesivamente pequeñas.
- Intenta que cada unidad no supere aproximadamente
  {self.max_tokens} tokens.
- No incluyas el índice de la última frase como límite.
- Devuelve únicamente JSON válido.

Frases:
{numbered_sentences}

Devuelve los índices de las frases después de las cuales
debe terminar una unidad semántica.

Formato exacto:
{{
  "boundary_sentence_indices": [int, ...],
  "segment_types": [
    "structured_paragraph",
    "paragraph",
    "list",
    "heading_block",
    "code",
    "qa",
    "other"
  ]
}}
""".strip()

        try:
            response = self.llm.complete_json(
                prompt
            )

        except Exception:
            return None

        return self._build_segments_from_response(
            response=response,
            sentences=sentences,
        )

    def _build_segments_from_response(
        self,
        response,
        sentences: List[str],
    ) -> Optional[List[Dict]]:
        """
        Valida la respuesta del LLM y reconstruye los segmentos
        utilizando las frases originales.
        """

        if not isinstance(response, dict):
            return None

        raw_boundaries = response.get(
            "boundary_sentence_indices"
        )

        if not isinstance(raw_boundaries, list):
            return None

        boundaries = []

        for boundary in raw_boundaries:
            if isinstance(boundary, (int, float)):
                boundary = int(boundary)

                if 0 <= boundary < len(sentences) - 1:
                    boundaries.append(boundary)

        boundaries = sorted(set(boundaries))

        # El LLM puede decidir que no hace falta subdividir.
        if not boundaries:
            return None

        raw_types = response.get(
            "segment_types",
            [],
        )

        allowed_types = {
            "structured_paragraph",
            "paragraph",
            "list",
            "heading_block",
            "code",
            "qa",
            "other",
        }

        output = []
        start_index = 0

        all_end_indices = boundaries + [
            len(sentences) - 1
        ]

        for segment_index, end_index in enumerate(
            all_end_indices
        ):
            segment_sentences = sentences[
                start_index:end_index + 1
            ]

            if not segment_sentences:
                start_index = end_index + 1
                continue

            segment_type = "paragraph"

            if (
                isinstance(raw_types, list)
                and segment_index < len(raw_types)
                and raw_types[segment_index]
                in allowed_types
            ):
                segment_type = raw_types[
                    segment_index
                ]

            segment_text = " ".join(
                segment_sentences
            ).strip()

            output.append(
                {
                    "text": segment_text,
                    "type": segment_type,
                }
            )

            start_index = end_index + 1

        return output or None

    # ======================================================
    # Separación en frases
    # ======================================================

    def _sentence_tokenize(
        self,
        text: str,
    ) -> List[str]:
        try:
            sentences = nltk.sent_tokenize(
                text,
                language="spanish",
            )

        except LookupError:
            nltk.download("punkt_tab")

            sentences = nltk.sent_tokenize(
                text,
                language="spanish",
            )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ======================================================
    # Conversión de segmentos a chunks
    # ======================================================

    def _split_to_max(
        self,
        text: str,
    ) -> List[str]:
        """
        Divide un segmento cuando supera max_tokens.

        Prioridad de corte:
        2. frase;
        3. palabra.
        """

        text = text.strip()

        if not text:
            return []


        if self.token_counter(text) <= self.max_tokens:
            return [text]

        sentences = self._sentence_tokenize(text)

        if not sentences:
            return self._split_by_words(text)

        parts = []
        current_sentences = []

        for sentence in sentences:
            sentence_token_count = self.token_counter(sentence)
            # Una sola frase es demasiado larga
            if sentence_token_count > self.max_tokens:
                if current_sentences:
                    parts.append(
                        " ".join(
                            current_sentences
                        ).strip()
                    )

                    current_sentences = []

                parts.extend(
                    self._split_by_words(sentence)
                )
                continue

            candidate_text = " ".join(
                current_sentences + [sentence]
            )

            candidate_token_count = self.token_counter(candidate_text)

            if current_sentences and candidate_token_count > self.max_tokens:
                parts.append(
                    " ".join(
                        current_sentences
                    ).strip()
                )

                current_sentences = [sentence]

        if current_sentences:
            parts.append(
                " ".join(
                    current_sentences
                ).strip()
            )

        return [
            part 
            for part in parts
            if part
        ]


    def _split_by_words(
        self,
        text: str,
    ) -> List[str]:
        """
        Fallback para frases individuales que superan max_chars.
        """

        words = text.split()

        parts = []
        current_words = []

        for word in words:
            candidate_text = " ".join(
                current_words + [word]
            )

            candidate_token_count = self.token_counter(candidate_text)
           

            if (
                current_words
                and candidate_token_count > self.max_tokens
            ):
                parts.append(
                    " ".join(current_words)
                )

                current_words = [word]

            else:
                current_words.append(word)

        if current_words:
            parts.append(
                " ".join(current_words)
            )

        return parts

    # ======================================================
    # Overlap
    # ======================================================

    def _apply_overlap(
        self,
        parts: List[Dict],
    ) -> List[Dict]:
        """
        Añade al inicio de cada chunk el final del anterior.
        """

        if self.overlap <= 0 or not parts:
            return parts

        output = []

        for index, part in enumerate(parts):
            new_part = dict(part)

            if index == 0:
                new_part["text"] = part["text"]
                output.append(new_part)
                continue

            previous_words = parts[index - 1]["text"].split()

            overlap_words = []

            for word in reversed(previous_words):
                candidate_words = [
                    word,
                    *overlap_words
                ]

                candidate_text = " ".join(candidate_words)

                if self.token_counter(candidate_text) > self.overlap_tokens:
                    break

                overlap_words = candidate_words

            overlap_text = " ".join(overlap_words).strip()

            if overlap_text:
                new_part["text"] = (
                    f"{overlap_text}\n\n"
                    f"{part['text']}"
                )
            else:
                new_part["text"] = part["text"]

            output.append(new_part)

        return output