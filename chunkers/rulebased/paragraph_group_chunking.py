from typing import List
from ..base import BaseChunker, Chunk

class ParagraphGroupChunker(BaseChunker):
    """
    Divide el texto en grupos de párrafos, con un número configurable
    de párrafos por chunk y solapamiento entre chunks.
    """

    def __init__(
        self,
        paragraphs_per_chunk: int,
        overlap: int = 0
    ):
        if paragraphs_per_chunk <= 0:
            raise ValueError(
                "paragraphs_per_chunk debe ser mayor que 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap no puede ser negativo."
            )

        if overlap >= paragraphs_per_chunk:
            raise ValueError(
                "overlap debe ser menor que paragraphs_per_chunk."
            )

        self.paragraphs_per_chunk = paragraphs_per_chunk
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        start = 0
        chunk_counter = 0
        step = self.paragraphs_per_chunk - self.overlap

        while start < len(paragraphs):
            end = min(
                start + self.paragraphs_per_chunk,
                len(paragraphs)
            )

            chunk_paragraphs = paragraphs[start:end]
            chunk_text = "\n\n".join(chunk_paragraphs)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "paragraph_group_chunking",
                        "paragraphs_per_chunk": self.paragraphs_per_chunk,
                        "overlap": self.overlap,
                        "start_paragraph_index": start,
                        "end_paragraph_index": end,
                        "paragraph_count": len(chunk_paragraphs),
                        "character_count": len(chunk_text),
                    },
                    chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1
            start += step

        return chunks