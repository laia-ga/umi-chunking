from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..base import BaseChunker, Chunk


class RecursiveTokenChunker(BaseChunker):
    """
    Divide el texto recursivamente usando tokens para medir
    el tamaño de los chunks.

    Intenta respetar primero párrafos y frases. Si una unidad
    sigue siendo demasiado grande, continúa dividiendo hasta
    ajustarse al límite de tokens.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size debe ser mayor que 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap no puede ser negativo."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap debe ser menor que chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

        self.splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",  # Párrafos
                    "\n",    # Saltos de línea
                    ". ",    # Frases
                    " ",     # Palabras
                    "",      # Último recurso
                ],
            )
        )

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        chunk_texts = self.splitter.split_text(text)

        chunks = []

        for chunk_text in chunk_texts:
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            token_count = len(
                self.encoding.encode(chunk_text)
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": (
                            "recursive_token_fallback_chunking"
                        ),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "encoding_name": self.encoding_name,
                        "token_count": token_count,
                        "character_count": len(chunk_text),
                    },
                    chunk_id=(
                        f"{doc_id}_chunk_{len(chunks):04d}"
                    ),
                    doc_id=doc_id,
                )
            )

        return chunks