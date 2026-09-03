from typing import Callable, List
import nltk
from ..base import BaseChunker, Chunk

class LengthAwareChunker(BaseChunker):
    """
    Agrupa frases completas intentando mantener cada chunk
    dentro de un rango de longitud en tokens
    """

    def __init__(
        self, 
        target_length: int, 
        tolerance: int,
        token_counter: Callable[[str], int],
    ):
        if target_length <= 0:
            raise ValueError("target_length debe ser mayor que 0.")

        if tolerance < 0:
            raise ValueError("tolerance no puede ser negativa.")

        if tolerance >= target_length:
            raise ValueError(
                "tolerance debe ser menor que target_length."
            )

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función"
            )

        self.target_length = target_length
        self.tolerance = tolerance
        self.token_counter = token_counter

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        try:
            sentences = nltk.sent_tokenize(
                text,
                language="spanish"
            )
        except LookupError:
            nltk.download("punkt_tab")

            sentences = nltk.sent_tokenize(
                text,
                language="spanish"
            )

        chunks = []
        current_sentences = []

        min_length = self.target_length - self.tolerance
        max_length = self.target_length + self.tolerance

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Construir temporalmente el chunk que resultaría de añadir la frase actual
            candidate_text = " ".join(
                current_sentences + [sentence]
            )

            candidate_length = self.token_counter(candidate_text)

            # Si añadir la frase supera el máximo de tokens,
            # se cierra el chunk actual
            if current_sentences and candidate_length > max_length:
                chunk_text = " ".join(current_sentences)

                token_count = self.token_counter(chunk_text)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": "length_aware_chunking",
                            "target_length": self.target_length,
                            "tolerance": self.tolerance,
                            "min_length": min_length,
                            "max_length": max_length,
                            "token_count": token_count,
                            "character_count": len(chunk_text),
                            "sentence_count": len(current_sentences),
                        },
                        chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                        doc_id=doc_id,
                    )
                )

                current_sentences = []

            current_sentences.append(sentence)

        # Guardar el último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)

            token_count = self.token_counter(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "length_aware_chunking",
                        "target_length_tokens": self.target_length,
                        "tolerance_tokens": self.tolerance,
                        "min_length_tokens": min_length,
                        "max_length_tokens": max_length,
                        "token_count": token_count,
                        "character_count": len(chunk_text),
                        "sentence_count": len(current_sentences),
                    },
                    chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                    doc_id=doc_id,
                )
            )

        return chunks