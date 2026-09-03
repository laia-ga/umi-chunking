from typing import List
from transformers import PreTrainedTokenizerBase
from ..base import BaseChunker, Chunk

class OverlappingTokenChunker(BaseChunker):
    """
    Divide el texto en chunks de tokens solapados.

    Utiliza el tokenizador general indicado en tokenizer_config.json
    """
    def __init__(
        self,
        chunk_size: int,
        overlap_size: int,
        tokenizer: PreTrainedTokenizerBase,
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
        self.tokenizer = tokenizer

    def chunk(
            self, 
            text: str, 
            doc_id: str
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        # Convertir el documento en tokens
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        start = 0
        chunk_counter = 0

        step = self.chunk_size - self.overlap_size

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if chunk_text: 
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": "overlapping_token_chunking",
                            "chunk_size_tokens": self.chunk_size,
                            "overlap_size": self.overlap_size,
                            "start_token": start,
                            "end_token": end,
                            "token_count": len(chunk_tokens),
                            "character_count": len(chunk_text),
                            "tokenizer": getattr(
                                self.tokenizer,
                                "name_or_path",
                                "unknown"
                            )
                        },
                        chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
                        doc_id=doc_id,
                    )
                )

            chunk_counter += 1

            if end == len(tokens):
                break
            
            start += step

        return chunks