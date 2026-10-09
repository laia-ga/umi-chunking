# ============================================================
# PIPELINE DEL RAG
# ============================================================

"""
Este módulo conecta el retrieval y la generación.

Flujo:

1. Recibe la pregunta del usuario.
2. Recupera los chunks más relevantes desde Qdrant.
3. Envía la pregunta y los chunks al LLM mediante vLLM.
4. Devuelve la respuesta generada y los chunks utilizados.
"""


# ============================================================
# IMPORTS
# ============================================================

from typing import Any, Dict, List, Optional, Tuple

from rag.retriever import (
    create_embeddings,
    connect_to_qdrant,
    retrieve_chunks,
)

from rag.generator import generate_answer


# ============================================================
# PIPELINE RAG
# ============================================================

class RAGPipeline:

    """
    Pipeline completo del sistema RAG.

    Los modelos de embeddings y la conexión con Qdrant
    se inicializan una única vez al crear el pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):

        self.config = config


        # ----------------------------------------------------
        # Configuración
        # ----------------------------------------------------

        self.qdrant_config = config[
            "qdrant"
        ]

        self.embeddings_config = config[
            "embeddings"
        ]

        self.retrieval_config = config[
            "retrieval"
        ]

        self.generation_config = config[
            "generation"
        ]


        # ----------------------------------------------------
        # Cargar modelo de embeddings
        # ----------------------------------------------------

        self.embeddings = create_embeddings(
            self.embeddings_config
        )


        # ----------------------------------------------------
        # Conectar con Qdrant
        # ----------------------------------------------------

        self.qdrant_client = connect_to_qdrant(
            self.qdrant_config
        )


    # ========================================================
    # RESPONDER PREGUNTA
    # ========================================================

    def answer(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        """
        Ejecuta el pipeline RAG completo para una pregunta.

        Returns
        -------
        answer:
            Respuesta generada por el LLM.

        chunks:
            Chunks recuperados desde Qdrant y utilizados
            como contexto para generar la respuesta.
        """

        # ----------------------------------------------------
        # 1. Recuperar chunks
        # ----------------------------------------------------

        chunks = retrieve_chunks(
            question=question,
            embeddings=self.embeddings,
            client=self.qdrant_client,
            collection_name=self.qdrant_config[
                "collection_name"
            ],
            top_k=self.retrieval_config[
                "top_k"
            ],
            strategies=self.retrieval_config[
                "strategies"
            ],
        )


        # ----------------------------------------------------
        # 2. Generar respuesta
        # ----------------------------------------------------

        answer = generate_answer(
            question=question,
            chunks=chunks,
            generation_config=self.generation_config,
            history=history,
        )


        # ----------------------------------------------------
        # 3. Devolver respuesta y chunks
        # ----------------------------------------------------

        return answer, chunks