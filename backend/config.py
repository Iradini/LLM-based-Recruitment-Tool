from pathlib import Path
from typing import Optional, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(".env"))

root = Path(__file__).parent.parent
print("root", root)


class Settings(BaseSettings):
    # Gemini settings
    GOOGLE_API_KEY: Optional[str] = ""
    GEMINI_LLM_MODEL: Optional[str] = "gemini-3.1-flash-lite-preview"

    LANGCHAIN_VERBOSE: bool = False

    # Document Ingestion
    DATASET_PATH: Optional[str] = f"{root}/dataset/jobs.csv"
    CHROMA_DB_PATH: Optional[str] = f"{root}/chroma"
    CHROMA_COLLECTION: Optional[str] = "jobs"
    EMBEDDINGS_MODEL: Optional[str] = "paraphrase-MiniLM-L6-v2"


settings = Settings()
