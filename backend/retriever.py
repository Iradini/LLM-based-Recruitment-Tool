import zipfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from backend.config import settings


def _ensure_chroma_store() -> None:
    """
    Downloads and extracts the chroma store from Google Cloud Storage if it
    isn't already present locally (e.g. on a fresh Render deploy, since the
    store is too large to commit to git).
    """
    chroma_path = Path(settings.CHROMA_DB_PATH)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        return
    if not settings.GCS_BUCKET_NAME:
        return

    from google.cloud import storage

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(settings.GCS_BUCKET_NAME)
    blob = bucket.blob(settings.GCS_CHROMA_BLOB)

    zip_path = chroma_path.parent / "chroma_download.zip"
    blob.download_to_filename(str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(chroma_path.parent)
    zip_path.unlink()


def load_vector_store() -> Chroma:
    """Build a vector base on Chroma. As an embedding function, we use Google's embedding API"""
    _ensure_chroma_store()
    return Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDINGS_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        ),
    )


class Retriever:
    """Retriever class to search jobs into a Chroma vector store."""

    def __init__(self):
        self.vector_store = load_vector_store()

    def search(self, query: str, k: int = 4) -> List[Document]:
        kits = self.vector_store.similarity_search(query=query, k=k)

        return kits
