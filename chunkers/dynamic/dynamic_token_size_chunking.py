from typing import List

import tiktoken

from ..base import BaseChunker, Chunk


class DynamicTokenSizeChunker(BaseChunker):
    """
    Divide el texto en chunks de tamaño variable medido en tokens.

    Para cada chunk:
    - intenta alcanzar como máximo max_chunk_size;
    - busca hacia atrás un final razonable entre min_chunk_size
      y max_chunk_size;
    - prioriza cortes en final de frase o salto de línea;
    - si no encuentra ninguno, corta en max_chunk_size.
    """

    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        encoding_name: str = "cl100k_base",
    ):
        if min_chunk_size <= 0:
            raise ValueError(
                "min_chunk_size debe ser mayor que 0."
            )

        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size debe ser mayor que 0."
            )

        if min_chunk_size >= max_chunk_size:
            raise ValueError(
                "min_chunk_size debe ser menor que max_chunk_size."
            )

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        chunks = []
        start = 0

        while start < total_tokens:
            target_end = min(
                start + self.max_chunk_size,
                total_tokens,
            )

            min_end = min(
                start + self.min_chunk_size,
                total_tokens,
            )

            best_end = target_end
            split_reason = "max_token_fallback"

            # Si el texto restante ya es menor que el mínimo,
            # se guarda completo como último chunk.
            if target_end == total_tokens:
                best_end = total_tokens
                split_reason = "end_of_document"

            else:
                # Buscar hacia atrás desde el máximo hasta el mínimo.
                # Se intenta cortar en puntuación o salto de línea.
                for token_index in range(
                    target_end,
                    min_end,
                    -1,
                ):
                    token_text = self.encoding.decode(
                        [tokens[token_index - 1]]
                    )

                    if "\n" in token_text:
                        best_end = token_index
                        split_reason = "newline"
                        break

                    if any(
                        punctuation in token_text
                        for punctuation in [".", "?", "!"]
                    ):
                        best_end = token_index
                        split_reason = "sentence_end"
                        break

            chunk_tokens = tokens[start:best_end]
            chunk_text = self.encoding.decode(
                chunk_tokens
            ).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": (
                                "dynamic_token_size_chunking"
                            ),
                            "min_chunk_size": (
                                self.min_chunk_size
                            ),
                            "max_chunk_size": (
                                self.max_chunk_size
                            ),
                            "encoding_name": self.encoding_name,
                            "start_token": start,
                            "end_token": best_end,
                            "token_count": len(chunk_tokens),
                            "character_count": len(chunk_text),
                            "split_reason": split_reason,
                        },
                        chunk_id=(
                            f"{doc_id}_chunk_{len(chunks):04d}"
                        ),
                        doc_id=doc_id,
                    )
                )

            start = best_end

        return chunks