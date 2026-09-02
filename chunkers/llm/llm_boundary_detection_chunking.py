from typing import List, Optional
import re
import statistics

import nltk

from ..base import BaseChunker, Chunk
from .llm_client import LLMClient


class LLMBoundaryDetectionChunker(BaseChunker):
    """
    Divide el texto mediante límites heurísticos y, opcionalmente,
    utiliza un LLM para refinar esos límites.

    Flujo:
    1. Separa el texto respetando párrafos y frases.
    2. Genera chunks de hasta max_chars caracteres.
    3. Si los tamaños están muy desequilibrados, pide al LLM
    que proponga mejores límites.
    4. Ajusta las propuestas del LLM a límites reales de frase
    o párrafo.
    5. Añade overlap entre chunks.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_chars: int = 1500,
        overlap: int = 80,
        enable_llm_refinement: bool = True,
        llm_timeout_sec: float = 8.0,
        llm_max_tokens: int = 128,
        llm_temperature: float = 0.0,
    ):
        if max_chars <= 0:
            raise ValueError(
                "max_chars debe ser mayor que 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap no puede ser negativo."
            )

        if overlap >= max_chars:
            raise ValueError(
                "overlap debe ser menor que max_chars."
            )

        self.llm = llm or LLMClient()

        self.max_chars = max_chars
        self.overlap = overlap
        self.enable_llm_refinement = enable_llm_refinement

        self.llm_timeout_sec = llm_timeout_sec
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        text = self._normalize(text)

        # Primera segmentación sin LLM
        slices = self._heuristic_chunk(text)

        used_llm = False

        if (
            self.enable_llm_refinement
            and self._needs_llm_refinement(slices)
        ):
            refined_slices = self._refine_boundaries_llm(
                text=text,
                current_chunks=slices,
            )

            if refined_slices:
                slices = refined_slices
                used_llm = True

        slices = self._apply_overlap(slices)

        chunks = []

        for index, chunk_text in enumerate(slices):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                    metadata={
                        "chunker": (
                            "llm_boundary_detection_chunking"
                        ),
                        "max_chars": self.max_chars,
                        "overlap": self.overlap,
                        "character_count": len(chunk_text),
                        "used_llm_refinement": used_llm,
                        "chunk_index": index,
                    },
                )
            )

        return chunks

    # ======================================================
    # Normalización
    # ======================================================

    def _normalize(self, text: str) -> str:
        """
        Normaliza los saltos de línea sin eliminar
        la separación entre párrafos.
        """
        return (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    # ======================================================
    # Segmentación heurística inicial
    # ======================================================

    def _split_into_units(self, text: str) -> List[str]:
        """
        Convierte el documento en unidades de frase.

        Se conservan los párrafos mediante una separación
        adicional entre las últimas y primeras frases
        de párrafos consecutivos.
        """

        try:
            nltk.sent_tokenize(
                "Texto de prueba.",
                language="spanish",
            )

        except LookupError:
            nltk.download("punkt_tab")

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        units = []

        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(
                paragraph,
                language="spanish",
            )

            sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            if sentences:
                units.extend(sentences)

        return units

    def _heuristic_chunk(self, text: str) -> List[str]:
        """
        Agrupa frases completas sin superar max_chars.

        Si una frase individual supera max_chars,
        se divide mediante _split_long_unit().
        """

        units = self._split_into_units(text)

        if not units:
            return [text]

        chunks = []
        current_units = []
        current_length = 0

        for unit in units:

            # Una sola frase supera la longitud máxima
            if len(unit) > self.max_chars:
                if current_units:
                    chunks.append(
                        " ".join(current_units).strip()
                    )
                    current_units = []
                    current_length = 0

                chunks.extend(
                    self._split_long_unit(unit)
                )
                continue

            separator_length = 1 if current_units else 0

            candidate_length = (
                current_length
                + separator_length
                + len(unit)
            )

            if (
                current_units
                and candidate_length > self.max_chars
            ):
                chunks.append(
                    " ".join(current_units).strip()
                )

                current_units = [unit]
                current_length = len(unit)

            else:
                current_units.append(unit)
                current_length = candidate_length

        if current_units:
            chunks.append(
                " ".join(current_units).strip()
            )

        return chunks

    def _split_long_unit(self, text: str) -> List[str]:
        """
        Divide una frase o unidad excepcionalmente larga.

        Intenta cortar por palabras y evita cortar
        directamente en mitad de una palabra.
        """

        words = text.split()

        fragments = []
        current_words = []
        current_length = 0

        for word in words:
            separator_length = 1 if current_words else 0

            candidate_length = (
                current_length
                + separator_length
                + len(word)
            )

            if (
                current_words
                and candidate_length > self.max_chars
            ):
                fragments.append(
                    " ".join(current_words)
                )

                current_words = [word]
                current_length = len(word)

            else:
                current_words.append(word)
                current_length = candidate_length

        if current_words:
            fragments.append(
                " ".join(current_words)
            )

        return fragments

    # ======================================================
    # Decisión de utilizar el LLM
    # ======================================================

    def _needs_llm_refinement(
        self,
        chunks: List[str],
    ) -> bool:
        """
        Solicita refinamiento cuando:

        - hay varios chunks muy pequeños, o
        - los tamaños están muy desequilibrados.
        """

        if len(chunks) < 3:
            return False

        sizes = [
            len(chunk)
            for chunk in chunks
        ]

        small_chunk_limit = self.max_chars * 0.35

        small_chunks = sum(
            size < small_chunk_limit
            for size in sizes
        )

        if small_chunks > len(chunks) * 0.30:
            return True

        try:
            mean_size = statistics.mean(sizes)
            standard_deviation = statistics.pstdev(sizes)

        except statistics.StatisticsError:
            return False

        if mean_size == 0:
            return False

        coefficient_of_variation = (
            standard_deviation / mean_size
        )

        return coefficient_of_variation > 0.50

    # ======================================================
    # Refinamiento mediante LLM
    # ======================================================

    def _refine_boundaries_llm(
        self,
        text: str,
        current_chunks: List[str],
    ) -> Optional[List[str]]:
        """
        Pide al LLM que elija límites entre las frases del texto.

        El LLM devuelve índices de frase, no posiciones arbitrarias
        de caracteres. Esto evita cortar palabras o frases.
        """

        sentences = self._split_into_units(text)

        if len(sentences) < 2:
            return None

        numbered_sentences = "\n".join(
            f"{index}: {sentence}"
            for index, sentence in enumerate(sentences)
        )

        current_sizes = [
            len(chunk)
            for chunk in current_chunks
        ]

        prompt = f"""
Eres un sistema de segmentación de documentos clínicos.

Debes dividir las siguientes frases en chunks coherentes y equilibrados.

Reglas:
- Conserva el orden original.
- No modifiques el texto.
- Cada límite debe colocarse después de una frase completa.
- Intenta que cada chunk tenga como máximo {self.max_chars} caracteres.
- Evita chunks demasiado pequeños.
- Separa cuando cambie claramente el tema.
- Devuelve únicamente JSON válido.

Tamaños actuales de los chunks:
{current_sizes}

Frases numeradas:
{numbered_sentences}

Devuelve los índices de las frases después de las cuales debe terminar
un chunk. No incluyas el índice de la última frase.

Formato exacto:
{{"boundary_sentence_indices": [int, ...]}}
""".strip()

        try:
            response = self.llm.complete_json(
                prompt,
                timeout=self.llm_timeout_sec,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
            )

        except Exception:
            return None

        if (
            not response
            or "boundary_sentence_indices" not in response
        ):
            return None

        boundaries = []

        for boundary in response[
            "boundary_sentence_indices"
        ]:
            if isinstance(boundary, (int, float)):
                boundary = int(boundary)

                if 0 <= boundary < len(sentences) - 1:
                    boundaries.append(boundary)

        boundaries = sorted(set(boundaries))

        if not boundaries:
            return None

        refined_chunks = []
        start_index = 0

        for boundary in boundaries:
            chunk_sentences = sentences[
                start_index:boundary + 1
            ]

            if chunk_sentences:
                refined_chunks.append(
                    " ".join(chunk_sentences)
                )

            start_index = boundary + 1

        remaining_sentences = sentences[start_index:]

        if remaining_sentences:
            refined_chunks.append(
                " ".join(remaining_sentences)
            )

        # No aceptar la propuesta si genera chunks
        # excesivamente grandes
        if any(
            len(chunk) > self.max_chars * 1.20
            for chunk in refined_chunks
        ):
            return None

        return refined_chunks

    # ======================================================
    # Overlap
    # ======================================================

    def _apply_overlap(
        self,
        chunks: List[str],
    ) -> List[str]:
        """
        Añade al chunk actual el final del chunk anterior.

        El inicio del overlap se ajusta al siguiente espacio
        para evitar comenzar en mitad de una palabra.
        """

        if self.overlap <= 0 or not chunks:
            return chunks

        output = [chunks[0]]

        for index in range(1, len(chunks)):
            previous_text = chunks[index - 1]

            overlap_start = max(
                0,
                len(previous_text) - self.overlap,
            )

            overlap_text = previous_text[overlap_start:]

            # Evitar comenzar en mitad de palabra
            if overlap_start > 0 and " " in overlap_text:
                overlap_text = overlap_text.split(
                    " ",
                    1,
                )[1]

            overlap_text = overlap_text.strip()

            if overlap_text:
                combined_text = (
                    f"{overlap_text}\n\n{chunks[index]}"
                )
            else:
                combined_text = chunks[index]

            output.append(combined_text)

        return output