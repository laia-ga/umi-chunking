from typing import Callable, List
import nltk
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from ..base import BaseChunker, Chunk


class SemanticVarianceAdaptiveChunker(BaseChunker):
    """
    Divide el texto cuando la similitud semántica entre dos frases
    consecutivas cae de forma significativa respecto a la media reciente.
    """

    def __init__(
        self,
        sensitivity: float, # cuánto debe caer la similitud respecto a la media para crear un corte
        token_counter: Callable[[str], int], # función que devuelve el número de tokens de un texto
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        window_size: int = 3, # número de frases anteriores para calcular la media
    ):
        """
        Parameters
        ----------
        sensitivity:
            Cuánto debe caer la similitud respecto a la media
            reciente para crear un corte.

        token_counter:
            Función general utilizada para contar los tokens.

        embedding_model:
            Modelo utilizado para calcular los embeddings de
            las frases.

        window_size:
            Número de similitudes anteriores utilizadas para
            calcular la media reciente.
        """
        if sensitivity < 0:
            raise ValueError(
                "sensitivity no puede ser negativa."
            )

        if window_size <= 0:
            raise ValueError(
                "window_size debe ser mayor que 0."
            )

        self.sensitivity = sensitivity
        self.token_counter = token_counter
        self.embedding_model = embedding_model
        self.window_size = window_size

        # Modelo para calcular la similitud semántica entre frases consecutivas
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

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

        if not sentences:
            return []

        # Caso de una única frase
        if len(sentences) == 1:
            sentence = sentences[0]

            return [
                Chunk(
                    text=sentence,
                    metadata={
                        "chunker": (
                            "semantic_variance_adaptive_chunking"
                        ),
                        "embedding_model": self.embedding_model,
                        "sensitivity": self.sensitivity,
                        "window_size": self.window_size,
                        "sentence_count": 1,
                        "character_count": len(sentence),
                    },
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                )
            ]

        # Generar embeddings de las frases
        sentence_embeddings = (
            self.embeddings.embed_documents(sentences)
        )

        # Calcular similitudes entre frases consecutivas
        similarities = []

        for index in range(1, len(sentences)):
            similarity = cosine_similarity(
                [sentence_embeddings[index - 1]],
                [sentence_embeddings[index]],
            )[0][0]

            similarities.append(float(similarity))

        chunks = []
        current_sentences = [sentences[0]]
        current_internal_similarities = []
        chunk_start_index = 0

        for index in range(1, len(sentences)):
            current_similarity = similarities[index - 1]

            recent_start = max(
                0,
                index - 1 - self.window_size,
            )

            recent_similarities = similarities[
                recent_start:index - 1
            ]

            should_split = False
            recent_mean = None

            if recent_similarities:
                recent_mean = float(
                    np.mean(recent_similarities)
                )

                should_split = (
                    current_similarity
                    < recent_mean - self.sensitivity
                )

            if should_split:
                chunk_text = " ".join(current_sentences)

                mean_internal_similarity = (
                    float(
                        np.mean(
                            current_internal_similarities
                        )
                    )
                    if current_internal_similarities
                    else None
                )

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": (
                                "semantic_variance_adaptive_chunking"
                            ),
                            "embedding_model": (
                                self.embedding_model
                            ),
                            "sensitivity": self.sensitivity,
                            "window_size": self.window_size,
                            "split_similarity": round(
                                current_similarity,
                                4,
                            ),
                            "recent_mean_similarity": round(
                                recent_mean,
                                4,
                            ),
                            "mean_internal_similarity": (
                                round(
                                    mean_internal_similarity,
                                    4,
                                )
                                if mean_internal_similarity
                                is not None
                                else None
                            ),
                            "start_sentence_index": (
                                chunk_start_index
                            ),
                            "end_sentence_index": index,
                            "sentence_count": len(
                                current_sentences
                            ),
                            "character_count": len(chunk_text),
                            "token_count": self.token_counter(chunk_text),
                        },
                        chunk_id=(
                            f"{doc_id}_chunk_{len(chunks):04d}"
                        ),
                        doc_id=doc_id,
                    )
                )

                current_sentences = []
                current_internal_similarities = []
                chunk_start_index = index

            else:
                current_internal_similarities.append(
                    current_similarity
                )

            current_sentences.append(sentences[index])

        # Guardar el último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)

            mean_internal_similarity = (
                float(
                    np.mean(current_internal_similarities)
                )
                if current_internal_similarities
                else None
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "semantic_variance_adaptive_chunking"
                        ),
                        "embedding_model": self.embedding_model,
                        "sensitivity": self.sensitivity,
                        "window_size": self.window_size,
                        "split_similarity": None,
                        "recent_mean_similarity": None,
                        "mean_internal_similarity": (
                            round(
                                mean_internal_similarity,
                                4,
                            )
                            if mean_internal_similarity is not None
                            else None
                        ),
                        "start_sentence_index": (
                            chunk_start_index
                        ),
                        "end_sentence_index": len(sentences),
                        "sentence_count": len(
                            current_sentences
                        ),
                        "character_count": len(chunk_text),
                        "token_count": self.token_counter(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks