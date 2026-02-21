import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Flask runtime
    host: str = os.getenv("FLASK_HOST", "127.0.0.1")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # App behavior
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are an AI assistant helping with customer inquiries. "
        "Provide a helpful and concise response.",
    )

    # Hugging Face auth
    huggingface_api_token: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Model IDs and provider config
    glm5_model_id: str = os.getenv("GLM5_MODEL_ID", "zai-org/GLM-5")
    glm5_provider: str = os.getenv("GLM5_PROVIDER", "novita")
    minimax_model_id: str = os.getenv("MINIMAX_MODEL_ID", "MiniMaxAI/MiniMax-M2.5")
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-235B-A22B")


settings = Settings()

