from typing import List
from ..base import BaseChunker, Chunk

class ParagraphBasedChunker(BaseChunker):
    """
    Divide el texto en párrafos individuales.

    Se considera que los párrafos están separados
    por dos saltos de línea: "\\n\\n".
    """

    def __init__(self):
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        for i, paragraph in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    text=paragraph,
                    metadata={
                        "chunker": "paragraph_based_chunking",
                        "paragraph_index": i,
                        "character_count": len(paragraph),
                    },
                    chunk_id=f"{doc_id}_chunk_{i:04d}",
                    doc_id=doc_id,
                )
            )

        return chunks