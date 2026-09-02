from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class OverlappingTokenChunker(BaseChunker):
    """
    Divide el texto en chunks de tokens solapados.
    """
    def __init__(
        self,
        chunk_size: int,
        overlap_size: int,
        encoding_name: str = "cl100k_base"
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que 0.")

        if overlap_size < 0:
            raise ValueError("overlap_size no puede ser negativo.")

        if overlap_size >= chunk_size:
            raise ValueError(
                "overlap_size debe ser menor que chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)

        chunks = []
        start = 0
        chunk_counter = 0
        step = self.chunk_size - self.overlap_size

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "overlapping_token_chunking",
                        "chunk_size": self.chunk_size,
                        "overlap_size": self.overlap_size,
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