from typing import Callable, List
from ..base import BaseChunker, Chunk

class FixedCharacterChunker(BaseChunker):
    """
    Divide el texto en chunks de un número fijo de caracteres,
    con un solapamiento configurable.
    """

    def __init__(
            self, 
            chunk_size: int, 
            token_counter: Callable[[str], int],
            overlap: int = 0):
        
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que 0.")

        if overlap < 0:
            raise ValueError("overlap no puede ser negativo.")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap debe ser menor que chunk_size."
            )

        if not callable(token_counter): 
            raise ValueError(
                "token_counter debe ser una función"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = token_counter

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        chunk_counter = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "fixed_character_chunking",
                        "chunk_size_characters": self.chunk_size,
                        "overlap_characters": self.overlap,
                        "start_char": start,
                        "end_char": end,
                        "character_count": len(chunk_text),
                        "token_count": self.token_counter(chunk_text),
                    },
                    chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1

            if end == len(text):
                break
            
            start += step

        return chunks