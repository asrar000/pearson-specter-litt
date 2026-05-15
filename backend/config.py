import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/psl.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "6"))

    # LLM
    MODEL: str = "claude-sonnet-4-20250514"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()

# Ensure runtime directories exist
for d in [settings.UPLOAD_DIR, settings.CHROMA_DB_PATH]:
    Path(d).mkdir(parents=True, exist_ok=True)
