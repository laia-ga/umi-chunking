import re
from typing import Callable, List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from ..base import BaseChunker, Chunk


class SemanticEmbeddingChunker(BaseChunker):
    """
    Divide un texto según cambios de significado entre frases.

    El método:
    1. separa el texto en frases;
    2. genera embeddings de grupos de frases;
    3. calcula las diferencias semánticas entre grupos consecutivos;
    4. crea un nuevo chunk cuando detecta una ruptura semántica.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1, # cuántas frases vecinas se añaden alrededor para calcular embedding
    ):
        if buffer_size < 0:
            raise ValueError(
                "buffer_size no puede ser negativo."
            )

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función."
            )

        valid_threshold_types = {
            "percentile",
            "standard_deviation",
            "interquartile",
            "gradient",
        }

        if breakpoint_threshold_type not in valid_threshold_types:
            raise ValueError(
                "breakpoint_threshold_type debe ser uno de: "
                f"{sorted(valid_threshold_types)}"
            )

        self.token_counter = token_counter
        self.embedding_model = embedding_model
        self.breakpoint_threshold_type = (
            breakpoint_threshold_type
        )
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
        )
        self.buffer_size = buffer_size

        # Modelo utilizado para detectar fronteras semánticas
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=(
                breakpoint_threshold_type
            ),
            breakpoint_threshold_amount=(
                breakpoint_threshold_amount
            ),
        )

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        text = text.strip()

        # Evita posibles errores del SemanticChunker
        # cuando el documento solo contiene una frase.
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text,
                    metadata={
                        "chunker": (
                            "semantic_embedding_based_chunking"
                        ),
                        "embedding_model": self.embedding_model,
                        "breakpoint_threshold_type": (
                            self.breakpoint_threshold_type
                        ),
                        "breakpoint_threshold_amount": (
                            self.breakpoint_threshold_amount
                        ),
                        "buffer_size": self.buffer_size,
                        "token_count": self.token_counter(text),
                        "character_count": len(text),
                        "single_sentence_document": True,
                    },
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                )
            ]

        chunk_texts = self.splitter.split_text(text)

        chunks = []

        for chunk_text in chunk_texts:
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "semantic_embedding_based_chunking"
                        ),
                        "embedding_model": self.embedding_model,
                        "breakpoint_threshold_type": (
                            self.breakpoint_threshold_type
                        ),
                        "breakpoint_threshold_amount": (
                            self.breakpoint_threshold_amount
                        ),
                        "buffer_size": self.buffer_size,
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