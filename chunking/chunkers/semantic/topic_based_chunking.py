from typing import Callable, List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import AgglomerativeClustering
import nltk
from ..base import BaseChunker, Chunk

class TopicBasedChunker(BaseChunker):
    """
    Agrupa las frases de un texto en chunks según su similitud semántica,
    a partir del modelo all-MiniLM-L6-v2 de sentence-transformers.

    Las frases se representan mediante embeddings y posteriormente se agrupan 
    mediante clustering jerárquico aglomerativo.

    Las frases de un mismo cluster no tienen por qué ser adyacentes en el 
    texto original.
    """

    def __init__(
            self, 
            token_counter: Callable[[str], int],
            distance_threshold: float, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            max_chunk_tokens: int | None = 512,
            device: str = "cpu",
    ):

        if not callable(token_counter):
            raise ValueError(
                "token_counter debe ser una función."
            )

        # Comprobación de la distancia coseno en rango [0,2]:
        if not 0 <= distance_threshold <= 2:
            raise ValueError(
                "distance_threshold debe estar en el rango [0, 2]."
            )

        self.token_counter = token_counter
        self.distance_threshold = distance_threshold
        self.embedding_model = embedding_model
        self.max_chunk_tokens = max_chunk_tokens
        self.device = device
        # Modelo utilizado para los embeddings de las frases
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
        )
        

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        try:
            sentences = nltk.sent_tokenize(text)

        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
            
        if len(sentences) < 2:
             # Just return one chunk
            chunks = [Chunk(
                text=sentences[0],
                metadata={
                    "chunker": "topic_based_chunking",
                    "topic_label": 0,
                    "sentence_count": 1,
                    "token_count": self.token_counter(sentences[0]),
                    "distance_threshold": self.distance_threshold,
                    "embedding_model": self.embedding_model
                },
                chunk_id=f"{doc_id}_chunk_0000",
                doc_id=doc_id
            )]
            return self.split_oversized_chunks(
                chunks, self.token_counter, self.max_chunk_tokens, doc_id
            )
        
        # Calculamos un embedding para cada frase
        embeddings = self.embeddings.embed_documents(sentences)
        X = np.array(embeddings)
        
        # Agrupamos las frases según su similitud semántica.
        # No imponemos ninguna restricción de adyacencia, por lo que
        # frases alejadas entre sí pueden pertenecer al mismo chunk.
        
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold,
            metric='cosine', 
            linkage='average'
        )
        labels = clustering.fit_predict(X)
        
        # Guardamos las frases pertenecientes a cada cluster
        topic_sentences = {}

        for sentence, label in zip(sentences, labels):
            label = int(label)

            if label not in topic_sentences:
                topic_sentences[label] = []

            topic_sentences[label].append(sentence)

        # Ordenamos los clusters según la posición de su primera frase
        # en el documento original
        topic_order = []

        for label in labels:
            label = int(label)

            if label not in topic_order:
                topic_order.append(label)

        chunks = []

        # Creamos un chunk por cada cluster temático
        for label in topic_order:
            sentences_in_topic = topic_sentences[label]

            # Las frases mantienen su orden relativo dentro del texto original
            chunk_text = " ".join(sentences_in_topic)

            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "topic_based_chunking",
                        "topic_label": label,
                        "sentence_count": len(sentences_in_topic),
                        "token_count": self.token_counter(
                            chunk_text
                        ),
                        "distance_threshold": self.distance_threshold,
                        "embedding_model": self.embedding_model,
                    },
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                )
            )

        return self.split_oversized_chunks(
            chunks, self.token_counter, self.max_chunk_tokens, doc_id
        )