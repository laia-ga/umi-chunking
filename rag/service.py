import json
import os
from pathlib import Path

from rag.pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "chatbot_config.json"


def create_pipeline() -> RAGPipeline:

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    config["generation"]["base_url"] = os.environ[
        "VLLM_URL"
    ]

    config["generation"]["model"] = os.environ[
        "VLLM_MODEL"
    ]

    return RAGPipeline(config)