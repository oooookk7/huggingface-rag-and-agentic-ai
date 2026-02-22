"""Module for interfacing with Hugging Face LLMs and embeddings."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import config

logger = logging.getLogger(__name__)

# Load .env from parent folder (02-build-rag-applications/.env)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def create_hf_embedding() -> HuggingFaceEmbedding:
    """Create a Hugging Face embedding model for vector representation."""
    embedding_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_ID)
    logger.info(f"Created Hugging Face embedding model: {config.EMBEDDING_MODEL_ID}")
    return embedding_model

def create_hf_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
) -> HuggingFaceInferenceAPI:
    """Create a Hugging Face LLM for response generation."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in ../.env.")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

    # HuggingFaceInferenceAPI validates temperature as strictly > 0.
    safe_temperature = max(float(temperature), 0.01)

    provider = config.HF_PROVIDER if isinstance(config.HF_PROVIDER, str) and config.HF_PROVIDER.strip() else "auto"

    hf_kwargs = {
        "model_name": config.LLM_MODEL_ID,
        "token": token,
        "provider": provider,
        "task": "conversational",
        "temperature": safe_temperature,
        "top_p": config.TOP_P,
    }

    hf_llm = HuggingFaceInferenceAPI(**hf_kwargs)

    logger.info(f"Created Hugging Face LLM model: {config.LLM_MODEL_ID}")
    return hf_llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use."""
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
