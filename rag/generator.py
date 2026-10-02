# ============================================================
# GENERADOR DE RESPUESTAS DEL CHATBOT
# ============================================================

"""
Este módulo se encarga de generar la respuesta final del RAG.

Flujo:

1. Recibe la pregunta del usuario.
2. Recibe los chunks recuperados por retriever.py.
3. Construye el contexto con esos chunks.
4. Construye el prompt para el LLM.
5. Envía la petición a la API de vLLM.
6. Devuelve la respuesta generada por el modelo.
"""


# ============================================================
# IMPORTS
# ============================================================

from typing import Any, Dict, List, Optional

import requests


# ============================================================
# CONSTRUIR CONTEXTO
# ============================================================

def build_context(
    chunks: List[Dict[str, Any]],
) -> str:

    """
    Construye el contexto que recibirá el LLM a partir
    de los chunks recuperados por Qdrant.

    Solo se envía al modelo el texto de cada chunk.
    Los demás metadatos se conservan fuera del prompt.
    """

    context_parts = []

    for chunk in chunks:

        text = chunk.get("text")

        if text:
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    return context


# ============================================================
# CONSTRUIR MENSAJES
# ============================================================

def build_messages(
    question: str,
    context: str,
) -> List[Dict[str, str]]:

    """
    Construye los mensajes que se enviarán al modelo.

    El modelo recibe:
    - instrucciones del sistema;
    - contexto recuperado;
    - pregunta del usuario.
    """

    system_prompt = (
        "Eres un asistente especializado en responder preguntas "
        "utilizando información recuperada de una base documental. "
        "Responde utilizando únicamente la información proporcionada "
        "en el contexto. "
        "No inventes información ni utilices conocimiento externo. "
        "Si el contexto no contiene información suficiente para "
        "responder a la pregunta, indícalo claramente. "
        "Responde de forma clara, precisa y concisa."
    )

    user_prompt = (
        f"CONTEXTO:\n"
        f"{context}\n\n"
        f"PREGUNTA:\n"
        f"{question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return messages


# ============================================================
# GENERAR RESPUESTA
# ============================================================

def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
) -> str:

    """
    Genera una respuesta utilizando la API de vLLM.

    Parameters
    ----------
    question:
        Pregunta realizada por el usuario.

    chunks:
        Chunks recuperados previamente por retriever.py.

    generation_config:
        Configuración de la API y de la generación.

    api_key:
        Token de autenticación opcional.

    Returns
    -------
    str
        Respuesta generada por el modelo.
    """

    # --------------------------------------------------------
    # 1. Comprobar que existen chunks
    # --------------------------------------------------------

    if not chunks:
        return (
            "No se ha encontrado información relevante "
            "para responder a la pregunta."
        )


    # --------------------------------------------------------
    # 2. Construir contexto
    # --------------------------------------------------------

    context = build_context(
        chunks
    )


    # --------------------------------------------------------
    # 3. Construir mensajes
    # --------------------------------------------------------

    messages = build_messages(
        question=question,
        context=context,
    )


    # --------------------------------------------------------
    # 4. Construir petición
    # --------------------------------------------------------

    base_url = generation_config[
        "base_url"
    ].rstrip("/")

    endpoint = (
        f"{base_url}/chat/completions"
    )

    payload = {
        "model": generation_config[
            "model"
        ],
        "messages": messages,
        "max_tokens": generation_config.get(
            "max_tokens",
            500,
        ),
        "temperature": generation_config.get(
            "temperature",
            0.0,
        ),
    }

    # --------------------------------------------------------
    # 6. Enviar petición a vLLM
    # --------------------------------------------------------

    response = requests.post(
        endpoint,
        json=payload,
        timeout=generation_config.get(
            "timeout",
            120,
        ),
    )

    response.raise_for_status()


    # --------------------------------------------------------
    # 7. Leer respuesta
    # --------------------------------------------------------

    result = response.json()

    answer = result[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ]

    return answer.strip()