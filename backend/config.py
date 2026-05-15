import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class Settings:
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/psl.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "6"))

    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEYS: list[str] = _dedupe(
        [GROQ_API_KEY] + _csv(os.getenv("GROQ_API_KEYS", ""))
    )
    GROQ_API_BASE_URL: str = os.getenv(
        "GROQ_API_BASE_URL",
        "https://api.groq.com/openai/v1",
    ).rstrip("/")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_MODEL_FALLBACKS: list[str] = _csv(
        os.getenv("GROQ_MODEL_FALLBACKS", "llama-3.1-8b-instant")
    )
    GROQ_RETRY_PER_MODEL: int = int(os.getenv("GROQ_RETRY_PER_MODEL", "1"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "420"))
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "45"))
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
    GROQ_RETRY_BACKOFF_BASE_SEC: float = float(
        os.getenv("GROQ_RETRY_BACKOFF_BASE_SEC", "3")
    )
    GROQ_RETRY_BACKOFF_MAX_SEC: float = float(
        os.getenv("GROQ_RETRY_BACKOFF_MAX_SEC", "45")
    )

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()

# Ensure runtime directories exist
for d in [settings.UPLOAD_DIR, settings.CHROMA_DB_PATH]:
    Path(d).mkdir(parents=True, exist_ok=True)
