from typing import Callable, List

import nltk

from ..base import BaseChunker, Chunk


class SentenceGroupChunker(BaseChunker):
    """
    Divide el texto en grupos de frases.

    El número de frases por chunk y el solapamiento
    son configurables.

    También registra el tamaño de cada chunk en tokens.
    """

    def __init__(
        self,
        sentences_per_chunk: int,
        token_counter: Callable[[str], int],
        overlap: int = 0,
    ):
        if sentences_per_chunk <= 0:
            raise ValueError(
                "sentences_per_chunk debe ser "
                "mayor que 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap no puede ser negativo."
            )

        if overlap >= sentences_per_chunk:
            raise ValueError(
                "overlap debe ser menor que "
                "sentences_per_chunk."
            )

        if not callable(token_counter):
            raise TypeError(
                "token_counter debe ser una función."
            )

        self.sentences_per_chunk = (
            sentences_per_chunk
        )
        self.overlap = overlap
        self.token_counter = token_counter

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

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

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks = []
        start = 0
        chunk_counter = 0

        step = (
            self.sentences_per_chunk
            - self.overlap
        )

        while start < len(sentences):
            end = min(
                start + self.sentences_per_chunk,
                len(sentences),
            )

            chunk_sentences = sentences[
                start:end
            ]

            chunk_text = " ".join(
                chunk_sentences
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "sentence_group_chunking"
                        ),
                        "sentences_per_chunk": (
                            self.sentences_per_chunk
                        ),
                        "overlap_sentences": (
                            self.overlap
                        ),
                        "start_sentence_index": start,
                        "end_sentence_index": end,
                        "sentence_count": len(
                            chunk_sentences
                        ),
                        "token_count": (
                            self.token_counter(
                                chunk_text
                            )
                        ),
                        "character_count": len(
                            chunk_text
                        ),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_"
                        f"{chunk_counter:04d}"
                    ),
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1

            # Evita crear un último chunk formado
            # únicamente por frases ya solapadas.
            if end == len(sentences):
                break

            start += step

        return chunks