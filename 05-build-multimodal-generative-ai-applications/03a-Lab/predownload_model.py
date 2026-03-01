"""
One-time model predownload utility for local/offline startup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import config


def main():
    load_dotenv()

    project_dir = Path(__file__).resolve().parent
    hf_home = Path(os.getenv("HF_HOME", str(project_dir / ".hf_cache")))
    local_model_dir = Path(
        os.getenv(
            "LOCAL_MODEL_DIR",
            str(project_dir / ".models" / config.HF_MODEL_ID.replace("/", "--")),
        )
    )
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")

    local_model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.HF_MODEL_ID,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"Model cached at: {local_model_dir}")
    print(f"HF cache at: {hf_home}")


if __name__ == "__main__":
    main()
