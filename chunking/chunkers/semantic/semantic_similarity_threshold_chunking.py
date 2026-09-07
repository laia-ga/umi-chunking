from typing import Callable, List

import nltk
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from ..base import BaseChunker, Chunk


class SemanticSimilarityThresholdChunker(BaseChunker):
    """
    Agrupa frases consecutivas mientras su similitud semántica
    sea igual o superior al umbral configurado.

    Cuando la similitud entre dos frases consecutivas cae
    por debajo del umbral, comienza un nuevo chunk.
    """

    def __init__(
        self,
        threshold: float,
        token_counter: Callable[[str], int],
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold debe estar entre 0 y 1."
            )

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función."
            )

        self.threshold = threshold
        self.token_counter = token_counter
        self.embedding_model = embedding_model

        # Modelo utilizado para calcular similitud semántica
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

        # Caso especial: una sola frase
        if len(sentences) == 1:
            sentence = sentences[0]

            return [
                Chunk(
                    text=sentence,
                    metadata={
                        "chunker": (
                            "semantic_similarity_threshold_chunking"
                        ),
                        "threshold": self.threshold,
                        "embedding_model": self.embedding_model,
                        "sentence_count": 1,
                        "mean_internal_similarity": None,
                        "token_count": self.token_counter(sentence),
                        "character_count": len(sentence),
                    },
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                )
            ]

        # Generar embeddings de todas las frases
        sentence_embeddings = (
            self.embeddings.embed_documents(sentences)
        )

        chunks = []
        current_sentences = [sentences[0]]
        current_similarities = []

        for index in range(1, len(sentences)):
            similarity = cosine_similarity(
                [sentence_embeddings[index - 1]],
                [sentence_embeddings[index]],
            )[0][0]

            similarity = float(similarity)

            if similarity >= self.threshold:
                current_sentences.append(sentences[index])
                current_similarities.append(similarity)

            else:
                chunk_text = " ".join(current_sentences)

                mean_similarity = (
                    sum(current_similarities)
                    / len(current_similarities)
                    if current_similarities
                    else None
                )

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": (
                                "semantic_similarity_threshold_chunking"
                            ),
                            "threshold": self.threshold,
                            "embedding_model": self.embedding_model,
                            "sentence_count": len(
                                current_sentences
                            ),
                            "mean_internal_similarity": (
                                round(mean_similarity, 4)
                                if mean_similarity is not None
                                else None
                            ),
                            "token_count": self.token_counter(chunk_text),
                            "character_count": len(chunk_text),
                        },
                        chunk_id=(
                            f"{doc_id}_chunk_{len(chunks):04d}"
                        ),
                        doc_id=doc_id,
                    )
                )

                current_sentences = [sentences[index]]
                current_similarities = []

        # Guardar el último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)

            mean_similarity = (
                sum(current_similarities)
                / len(current_similarities)
                if current_similarities
                else None
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "semantic_similarity_threshold_chunking"
                        ),
                        "threshold": self.threshold,
                        "embedding_model": self.embedding_model,
                        "sentence_count": len(
                            current_sentences
                        ),
                        "mean_internal_similarity": (
                            round(mean_similarity, 4)
                            if mean_similarity is not None
                            else None
                        ),
                        "token_count": self.token_counter(chunk_text),
                        "character_count": len(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks