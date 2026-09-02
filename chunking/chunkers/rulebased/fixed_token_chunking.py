from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class FixedTokenChunker(BaseChunker):
    """
    Divide el texto en chunks de un número fijo de tokens,
    con un solapamiento configurable.
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int = 0,
        encoding_name: str = "cl100k_base"
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que 0.")

        if overlap < 0:
            raise ValueError("overlap no puede ser negativo.")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap debe ser menor que chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)

        chunks = []
        start = 0
        chunk_counter = 0
        step = self.chunk_size - self.overlap

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "fixed_token_chunking",
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        "start_token": start,
                        "end_token": end,
                        "token_count": len(chunk_tokens),
                    },
                    chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1
            start += step

        return chunks