# ============================================================
# EJECUCIÓN DEL CHATBOT RAG
# ============================================================

"""
Script para ejecutar el chatbot RAG desde terminal.

Flujo:

1. Carga chatbot_config.json.
2. Define la URL de vLLM y el nombre del modelo.
3. Inicializa el pipeline RAG.
4. Permite introducir preguntas desde terminal.
5. Recupera los chunks relevantes desde Qdrant.
6. Genera una respuesta mediante vLLM.
7. Muestra la respuesta y las fuentes recuperadas.
"""


# ============================================================
# IMPORTS
# ============================================================

import json
import sys
from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent

CONFIG_FILE = (
    BASE_DIR
    / "configs"
    / "chatbot_config.json"
)


# ============================================================
# CONFIGURACIÓN DE vLLM
# ============================================================

VLLM_BASE_URL = "http://localhost:8000/v1/chat/completions"

VLLM_MODEL = "qwen3.6-27b-nvfp4"


# ============================================================
# AÑADIR RAÍZ DEL PROYECTO AL PATH
# ============================================================

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )


# Importar después de añadir BASE_DIR al path
from rag.pipeline import RAGPipeline


# ============================================================
# CARGAR CONFIGURACIÓN
# ============================================================

def load_config(
    config_file: Path,
) -> dict:

    """
    Carga la configuración del chatbot desde JSON.
    """

    if not config_file.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de configuración: "
            f"{config_file}"
        )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# MOSTRAR DOCUMENTOS RECUPERADOS
# ============================================================

def print_sources(
    chunks: list,
) -> None:

    """
    Muestra los documentos de los que proceden los chunks
    recuperados.

    Los documentos repetidos se muestran una sola vez.
    """

    document_ids = []

    for chunk in chunks:

        document_id = chunk.get(
            "document_id"
        )

        if (
            document_id
            and document_id not in document_ids
        ):
            document_ids.append(
                document_id
            )

    if not document_ids:
        return

    print("\nFuentes recuperadas:")

    for document_id in document_ids:
        print(
            f"- {document_id}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Cargar configuración
    # --------------------------------------------------------

    config = load_config(
        CONFIG_FILE
    )


    # --------------------------------------------------------
    # 2. Añadir configuración de vLLM
    # --------------------------------------------------------

    config["generation"]["base_url"] = (
        VLLM_BASE_URL
    )

    config["generation"]["model"] = (
        VLLM_MODEL
    )


    # --------------------------------------------------------
    # 3. Inicializar pipeline RAG
    # --------------------------------------------------------

    print(
        "\nInicializando chatbot..."
    )

    pipeline = RAGPipeline(
        config=config,
    )


    # --------------------------------------------------------
    # 4. Mostrar información inicial
    # --------------------------------------------------------

    print(
        "\n"
        "========================================\n"
        "UMI RAG\n"
        "========================================\n"
        "\n"
        "Escribe una pregunta para consultar "
        "la base documental.\n"
        "Escribe 'salir' para cerrar el chatbot.\n"
    )


    # --------------------------------------------------------
    # 5. Bucle del chatbot
    # --------------------------------------------------------

    while True:

        try:

            question = input(
                "Pregunta: "
            ).strip()


            # ------------------------------------------------
            # Salir
            # ------------------------------------------------

            if question.lower() in {
                "salir",
                "exit",
                "quit",
            }:

                print(
                    "\nChatbot cerrado."
                )

                break


            # ------------------------------------------------
            # Ignorar preguntas vacías
            # ------------------------------------------------

            if not question:
                continue


            # ------------------------------------------------
            # Ejecutar RAG
            # ------------------------------------------------

            print(
                "\nBuscando información..."
            )

            answer, chunks = pipeline.answer(
                question
            )


            # ------------------------------------------------
            # Mostrar respuesta
            # ------------------------------------------------

            print(
                "\nRespuesta:\n"
            )

            print(
                answer
            )


            # ------------------------------------------------
            # Mostrar fuentes
            # ------------------------------------------------

            print_sources(
                chunks
            )

            print()


        # ----------------------------------------------------
        # Ctrl + C
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print(
                "\n\nChatbot cerrado."
            )

            break


        # ----------------------------------------------------
        # Error durante una consulta
        # ----------------------------------------------------

        except Exception as error:

            print(
                f"\nError al procesar la pregunta: "
                f"{error}\n"
            )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()