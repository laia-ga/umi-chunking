from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..base import BaseChunker, Chunk


class ParentChildChunker(BaseChunker):
    """
    Divide el texto en chunks parent y child.

    Los parents contienen más contexto.
    Los children son fragmentos más pequeños asociados
    a su parent mediante parent_id.
    """

    def __init__(
        self,
        parent_chunk_size: int,
        child_chunk_size: int,
        chunk_overlap: int = 0,
    ):
        if parent_chunk_size <= 0:
            raise ValueError(
                "parent_chunk_size debe ser mayor que 0."
            )

        if child_chunk_size <= 0:
            raise ValueError(
                "child_chunk_size debe ser mayor que 0."
            )

        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                "child_chunk_size debe ser menor que "
                "parent_chunk_size."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap no puede ser negativo."
            )

        if chunk_overlap >= child_chunk_size:
            raise ValueError(
                "chunk_overlap debe ser menor que "
                "child_chunk_size."
            )

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

        separators = [
            "\n\n",  # Párrafos
            "\n",    # Saltos de línea
            ". ",    # Frases
            " ",     # Palabras
            "",      # Último recurso: caracteres
        ]

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False,
        )

    def chunk(
        self,
        text: str,
        doc_id: str,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        parent_texts = self.parent_splitter.split_text(text)

        chunks = []

        for parent_index, parent_text in enumerate(parent_texts):
            parent_text = parent_text.strip()

            if not parent_text:
                continue

            parent_chunk_id = (
                f"{doc_id}_parent_{parent_index:04d}"
            )

            # Chunk parent
            chunks.append(
                Chunk(
                    text=parent_text,
                    metadata={
                        "chunker": "parent_child_chunking",
                        "chunk_type": "parent",
                        "parent_chunk_size": (
                            self.parent_chunk_size
                        ),
                        "child_chunk_size": (
                            self.child_chunk_size
                        ),
                        "chunk_overlap": self.chunk_overlap,
                        "character_count": len(parent_text),
                    },
                    chunk_id=parent_chunk_id,
                    doc_id=doc_id,
                )
            )

            # Chunks child generados a partir del parent
            child_texts = self.child_splitter.split_text(
                parent_text
            )

            for child_index, child_text in enumerate(child_texts):
                child_text = child_text.strip()

                if not child_text:
                    continue

                child_chunk_id = (
                    f"{parent_chunk_id}_child_"
                    f"{child_index:04d}"
                )

                chunks.append(
                    Chunk(
                        text=child_text,
                        metadata={
                            "chunker": "parent_child_chunking",
                            "chunk_type": "child",
                            "parent_id": parent_chunk_id,
                            "parent_index": parent_index,
                            "child_index": child_index,
                            "parent_chunk_size": (
                                self.parent_chunk_size
                            ),
                            "child_chunk_size": (
                                self.child_chunk_size
                            ),
                            "chunk_overlap": self.chunk_overlap,
                            "character_count": len(child_text),
                        },
                        chunk_id=child_chunk_id,
                        doc_id=doc_id,
                    )
                )

        return chunks