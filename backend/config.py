from pathlib import Path
from typing import Optional, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(".env"))

root = Path(__file__).parent.parent


class Settings(BaseSettings):
    # Gemini settings
    GOOGLE_API_KEY: Optional[str] = ""
    GEMINI_LLM_MODEL: Optional[str] = "gemini-3.1-flash-lite-preview"

    LANGCHAIN_VERBOSE: bool = False

    # Document Ingestion
    DATASET_PATH: Optional[str] = str(root / "dataset" / "jobs.csv")
    CHROMA_DB_PATH: Optional[str] = str(root / "chroma")
    CHROMA_COLLECTION: Optional[str] = "jobs"
    EMBEDDINGS_MODEL: Optional[str] = "gemini-embedding-001"

    # Remote chroma store (used to fetch chroma/ on startup if not present
    # locally, e.g. on Render, since the store is too large to commit to git)
    GCS_BUCKET_NAME: Optional[str] = ""
    GCS_CHROMA_BLOB: Optional[str] = "chroma.zip"


settings = Settings()
