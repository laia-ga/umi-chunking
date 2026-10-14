import re
from typing import Callable, List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from ..base import BaseChunker, Chunk


class SemanticBoundaryChunker(BaseChunker):
    """
    Divide el texto detectando cambios semánticos bruscos
    entre frases consecutivas.

    Utiliza embeddings del modelo all-MiniLM-L6-L2 y el criterio de corte "gradient".
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1, # cuántas frases vecinas se consideran para calcular el embedding
        max_chunk_tokens: int | None = 512,
        device: str = "cpu",
    ):
        """
        Parámetros:
        ----------
        token_counter:
            Función utilizada para contar los tokens.

        embedding_model:
            Modelo utilizado para generar los embeddings
            necesarios para detectar fronteras semánticas.

        breakpoint_threshold_amount:
            Umbral utilizado por SemanticChunker.

        buffer_size:
            Número de frases vecinas utilizadas para
            calcular cada embedding.
        
        max_chunk_tokens:
            Tope de seguridad: si un chunk supera este número de tokens
            (porque el criterio semántico no encontró un buen punto de
            corte), se vuelve a partir por frases. None desactiva el tope.
        """
        if buffer_size < 0:
            raise ValueError(
                "buffer_size no puede ser negativo."
            )

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función."
            )

        self.token_counter = token_counter
        self.embedding_model = embedding_model
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
        )
        self.buffer_size = buffer_size
        self.max_chunk_tokens = max_chunk_tokens
        self.device = device

        # Modelo para detectar los cambios semánticos
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": device,
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
            chunks = [
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
                        "token_count": self.token_counter(text),
                        "character_count": len(text),
                        "single_sentence_document": True,
                    },
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                )
            ]
            return self.split_oversized_chunks(
                chunks, self.token_counter, self.max_chunk_tokens, doc_id
            )

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
                        "token_count": self.token_counter(chunk_text),
                        "character_count": len(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return self.split_oversized_chunks(
            chunks, self.token_counter, self.max_chunk_tokens, doc_id
        )