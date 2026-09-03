from typing import Callable, List
import nltk
from ..base import BaseChunker, Chunk


class ContentDensityAdaptiveChunker(BaseChunker):
    """
    Divide el texto en chunks de tamaño variable según
    la densidad léxica de las frases.

    - Texto más denso y con más palabras diferentes:
    chunks más pequeños.

    - Texto más repetitivo:
    chunks más grandes.
    """

    def __init__(
            self, 
            base_chunk_size: int,
            token_counter: Callable[[str], int],
        ) -> None:
        """
        Parameters
        ----------
        base_chunk_size:
            Tamaño base del chunk expresado en tokens.

            El límite real se adapta según la densidad léxica.

        token_counter:
            Función que recibe un texto y devuelve su número de
            tokens utilizando el tokenizador configurado para
            todo el proyecto.
        """
        if base_chunk_size <= 0:
            raise ValueError(
                "base_chunk_size debe ser mayor que 0."
            )
        
        if not callable(token_counter):
            raise TypeError(
                "token_counter debe ser una función."
            )

        self.base_chunk_size = base_chunk_size
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

            sentences = nltk.sent_tokenize( # tokenizador utilizado para separar palabras
                text,
                language="spanish",
            )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_densities = []

        for sentence in sentences:
            try:
                tokens = nltk.word_tokenize(
                    sentence,
                    language="spanish",
                )

            except LookupError:
                nltk.download("punkt_tab")

                tokens = nltk.word_tokenize(
                    sentence,
                    language="spanish",
                )

            # Conservar únicamente tokens que contengan
            # al menos un carácter alfanumérico
            words = [
                token.lower()
                for token in tokens
                if any(character.isalnum() for character in token)
            ]

            if not words:
                continue

            unique_words = set(words)

            density = len(unique_words) / len(words)

            # Densidad alta -> factor menor -> chunks pequeños
            # Densidad baja -> factor mayor -> chunks grandes
            factor = 1.5 - density

            effective_max_length = int(
                self.base_chunk_size * factor
            )

            # Crear temporalmente el texto que tendría el chunk si se añadiera la frase actual
            candidate_text = " ".join(
                current_sentences + [sentence]
            )

            # El tamaño se calcula con el tokenizador configurado para todo el proyecto
            candidate_length = self.token_counter(candidate_text) 

            if (
                current_sentences
                and candidate_length > effective_max_length
            ):
                chunk_text = " ".join(current_sentences)

                mean_density = (
                    sum(current_densities)
                    / len(current_densities)
                )

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": (
                                "content_density_adaptive_chunking"
                            ),
                            "base_chunk_size": self.base_chunk_size,
                            "mean_density": round(
                                mean_density,
                                4,
                            ),
                            "mean_density_factor": round(
                                1.5 - mean_density,
                                4,
                            ),
                            "sentence_count": len(
                                current_sentences
                            ),
                            "token_count": self.token_counter(chunk_text),
                        },

                        chunk_id=(
                            f"{doc_id}_chunk_{len(chunks):04d}"
                        ),
                        doc_id=doc_id,
                    )
                )

                current_sentences = []
                current_densities = []

            current_sentences.append(sentence)
            current_densities.append(density)

        # Guardar el último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)

            mean_density = (
                sum(current_densities)
                / len(current_densities)
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "content_density_adaptive_chunking"
                        ),
                        "base_chunk_size": self.base_chunk_size,
                        "mean_density": round(
                            mean_density,
                            4,
                        ),
                        "mean_density_factor": round(
                            1.5 - mean_density,
                            4,
                        ),
                        "sentence_count": len(
                            current_sentences
                        ),
                        "token_count": self.token_counter(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks