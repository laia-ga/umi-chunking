from typing import List
import nltk
from ..base import BaseChunker, Chunk

class SentenceBasedChunker(BaseChunker):
    """
    Divide el texto en frases individuales.
    Cada frase se convierte en un chunk.
    """

    def __init__(self):
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        try:
            sentences = nltk.sent_tokenize(
                text,
                language="spanish"
            )
        except LookupError:
            nltk.download("punkt_tab")

            sentences = nltk.sent_tokenize(
                text,
                language="spanish"
            )

        chunks = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            chunks.append(
                Chunk(
                    text=sentence,
                    metadata={
                        "chunker": "sentence_based_chunking",
                        "sentence_index": i,
                        "character_count": len(sentence),
                    },
                    chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                    doc_id=doc_id,
                )
            )

        return chunks