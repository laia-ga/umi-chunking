from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class SlidingWindowTokenChunker(BaseChunker):
    """
    Divide el texto mediante una ventana deslizante de tokens.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int,
        encoding_name: str = "cl100k_base"
    ):
        if window_size <= 0:
            raise ValueError("window_size debe ser mayor que 0.")

        if step_size <= 0:
            raise ValueError("step_size debe ser mayor que 0.")

        if step_size > window_size:
            raise ValueError(
                "step_size no debería ser mayor que window_size, "
                "porque quedarían partes del texto sin cubrir."
            )

        self.window_size = window_size
        self.step_size = step_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)

        chunks = []
        start = 0
        chunk_counter = 0

        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "sliding_window_token_chunking",
                        "window_size": self.window_size,
                        "step_size": self.step_size,
                        "overlap_size": max(
                            0,
                            self.window_size - self.step_size
                        ),
                        "start_token": start,
                        "end_token": end,
                        "token_count": len(chunk_tokens),
                    },
                    chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1
            start += self.step_size

        return chunks