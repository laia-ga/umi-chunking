import re
from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from ..base import BaseChunker, Chunk


class SemanticBoundaryChunker(BaseChunker):
    """
    Divide el texto detectando cambios semánticos bruscos
    entre frases consecutivas.

    Utiliza embeddings y el criterio de corte "gradient".
    """

    def __init__(
        self,
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1, # cuántas frases vecinas se consideran para calcular el embedding
    ):
        if buffer_size < 0:
            raise ValueError(
                "buffer_size no puede ser negativo."
            )

        self.embedding_model = embedding_model
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
        )
        self.buffer_size = buffer_size

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
            breakpoint_threshold_type="gradient",
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

        # Control para textos con una sola frase
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
                            "semantic_boundary_detection"
                        ),
                        "embedding_model": self.embedding_model,
                        "breakpoint_threshold_type": "gradient",
                        "breakpoint_threshold_amount": (
                            self.breakpoint_threshold_amount
                        ),
                        "buffer_size": self.buffer_size,
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
                            "semantic_boundary_detection"
                        ),
                        "embedding_model": self.embedding_model,
                        "breakpoint_threshold_type": "gradient",
                        "breakpoint_threshold_amount": (
                            self.breakpoint_threshold_amount
                        ),
                        "buffer_size": self.buffer_size,
                        "character_count": len(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks