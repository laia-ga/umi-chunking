from typing import Callable, List
from ..base import BaseChunker, Chunk

class ParagraphBasedChunker(BaseChunker):
    """
    Divide el texto en párrafos individuales.

    Se considera que los párrafos están separados
    por dos saltos de línea: "\\n\\n".

    No aplica ningún límite de tamaño
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
    ):
        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función"
            )
        self.token_counter = token_counter

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        for index, paragraph in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    text=paragraph,
                    metadata={
                        "chunker": "paragraph_based_chunking",
                        "paragraph_index": index,
                        "token_count": self.token_counter(paragraph),
                        "character_count": len(paragraph),
                    },
                    chunk_id=f"{doc_id}_chunk_{index:04d}",
                    doc_id=doc_id,
                )
            )

        return chunks