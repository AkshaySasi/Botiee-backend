import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    logger.info("Pre-downloading model for offline use...")
    model_path = snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./models/all-MiniLM-L6-v2",
        local_dir_use_symlinks=False
    )
    logger.info(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()
