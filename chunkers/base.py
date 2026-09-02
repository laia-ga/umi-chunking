from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Representa un único chunk de texto.
    """

    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str


class BaseChunker(ABC):
    """
    Clase base abstracta para todas las estrategias de chunking.
    """

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Divide un texto en una lista de objetos Chunk.
        """
        pass