# ============================================================
# CHATBOT RAG - INTERFAZ DE TERMINAL
# ============================================================

"""
Este script permite utilizar el chatbot RAG desde la terminal.

La inicialización del pipeline se realiza mediante
rag/service.py, que se encarga de:

1. Cargar la configuración del chatbot.
2. Obtener la URL y el modelo de vLLM.
3. Inicializar el modelo de embeddings.
4. Conectar con Qdrant.
5. Crear el pipeline RAG.

Flujo:

Pregunta del usuario
        ↓
RAGPipeline
        ↓
Retriever
        ↓
Qdrant
        ↓
Chunks relevantes
        ↓
Generator
        ↓
vLLM
        ↓
Respuesta
"""


# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

# Directorio donde se encuentra este script:
# umi-chunking/scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent

# Raíz del proyecto:
# umi-chunking/
BASE_DIR = SCRIPTS_DIR.parent

# Permite importar los módulos del proyecto cuando
# ejecutamos:
#
# python scripts/run_chatbot.py
#
sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORTS DEL RAG
# ============================================================

from rag.service import create_pipeline


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Inicializar pipeline
    # --------------------------------------------------------

    print("Inicializando chatbot...")

    pipeline = create_pipeline()

    print("Chatbot listo.")
    print("Escribe 'salir' para terminar.\n")


    # --------------------------------------------------------
    # 2. Bucle de conversación
    # --------------------------------------------------------

    while True:

        question = input(
            "Pregunta: "
        ).strip()


        # ----------------------------------------------------
        # Salir del chatbot
        # ----------------------------------------------------

        if question.lower() in {
            "salir",
            "exit",
            "quit",
        }:
            print("Cerrando chatbot.")
            break


        # ----------------------------------------------------
        # Ignorar preguntas vacías
        # ----------------------------------------------------

        if not question:
            continue


        # ----------------------------------------------------
        # Ejecutar pipeline RAG
        # ----------------------------------------------------

        try:

            answer, chunks = pipeline.answer(
                question
            )


            # ------------------------------------------------
            # Mostrar respuesta
            # ------------------------------------------------

            print(
                f"\nRespuesta:\n"
                f"{answer}\n"
            )


        # ----------------------------------------------------
        # Control de errores
        # ----------------------------------------------------

        except Exception as exc:

            print(
                f"\nError al procesar la pregunta: "
                f"{exc}\n"
            )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()