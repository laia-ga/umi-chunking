from typing import List
import nltk
from ..base import BaseChunker, Chunk

class SentenceGroupChunker(BaseChunker):
    """
    Divide el texto en grupos de frases, con un número configurable
    de frases por chunk y solapamiento entre chunks.
    """

    def __init__(
        self,
        sentences_per_chunk: int,
        overlap: int = 0
    ):
        if sentences_per_chunk <= 0:
            raise ValueError(
                "sentences_per_chunk debe ser mayor que 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap no puede ser negativo."
            )

        if overlap >= sentences_per_chunk:
            raise ValueError(
                "overlap debe ser menor que sentences_per_chunk."
            )

        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

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

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks = []
        start = 0
        chunk_counter = 0
        step = self.sentences_per_chunk - self.overlap

        while start < len(sentences):
            end = min(
                start + self.sentences_per_chunk,
                len(sentences)
            )

            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "sentence_group_chunking",
                        "sentences_per_chunk": self.sentences_per_chunk,
                        "overlap": self.overlap,
                        "start_sentence_index": start,
                        "end_sentence_index": end,
                        "sentence_count": len(chunk_sentences),
                        "character_count": len(chunk_text),
                    },
                    chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                    doc_id=doc_id,
                )
            )

            chunk_counter += 1
            start += step

        return chunks