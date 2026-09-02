from typing import List
import nltk
from ..base import BaseChunker, Chunk

class LengthAwareChunker(BaseChunker):
    """
    Agrupa frases completas intentando mantener cada chunk
    dentro de un rango de longitud en caracteres.
    """

    def __init__(self, target_length: int, tolerance: int):
        if target_length <= 0:
            raise ValueError("target_length debe ser mayor que 0.")

        if tolerance < 0:
            raise ValueError("tolerance no puede ser negativa.")

        if tolerance >= target_length:
            raise ValueError(
                "tolerance debe ser menor que target_length."
            )

        self.target_length = target_length
        self.tolerance = tolerance

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
        current_length = 0

        min_length = self.target_length - self.tolerance
        max_length = self.target_length + self.tolerance

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Se añade 1 carácter por el espacio entre frases
            separator_length = 1 if current_sentences else 0

            candidate_length = (
                current_length
                + separator_length
                + len(sentence)
            )

            # Si añadir la frase supera el máximo,
            # se cierra el chunk actual
            if current_sentences and candidate_length > max_length:
                chunk_text = " ".join(current_sentences)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": "length_aware_chunking",
                            "target_length": self.target_length,
                            "tolerance": self.tolerance,
                            "min_length": min_length,
                            "max_length": max_length,
                            "length": len(chunk_text),
                            "sentence_count": len(current_sentences),
                        },
                        chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                        doc_id=doc_id,
                    )
                )

                current_sentences = []
                current_length = 0

            separator_length = 1 if current_sentences else 0

            current_sentences.append(sentence)
            current_length += separator_length + len(sentence)

        # Guardar el último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "length_aware_chunking",
                        "target_length": self.target_length,
                        "tolerance": self.tolerance,
                        "min_length": min_length,
                        "max_length": max_length,
                        "length": len(chunk_text),
                        "sentence_count": len(current_sentences),
                    },
                    chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                    doc_id=doc_id,
                )
            )

        return chunks