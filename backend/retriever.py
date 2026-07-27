import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from backend.config import settings


def _is_populated(chroma_path: Path) -> bool:
    """
    Checks whether chroma_path holds a collection that actually has
    documents in it, as opposed to an empty/partial store left behind by a
    previously failed or incomplete download.
    """
    if not chroma_path.exists():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(settings.CHROMA_COLLECTION)
        return collection.count() > 0
    except Exception:
        return False


def _ensure_chroma_store() -> None:
    """
    Downloads and extracts the chroma store from Google Cloud Storage if it
    isn't already present locally (e.g. on a fresh Render deploy, since the
    store is too large to commit to git). Extracts to a temp directory first
    and swaps it into place atomically, so a failed/partial download never
    leaves behind a broken store that a later check could mistake for a
    valid one.
    """
    chroma_path = Path(settings.CHROMA_DB_PATH)
    if _is_populated(chroma_path):
        return
    if not settings.GCS_BUCKET_NAME:
        return

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.GCS_BUCKET_NAME)
    blob = bucket.blob(settings.GCS_CHROMA_BLOB)
    blob.reload()  # populate .size from GCS metadata
    expected_size = blob.size
    print(
        f"[chroma] downloading gs://{settings.GCS_BUCKET_NAME}/{settings.GCS_CHROMA_BLOB} "
        f"({expected_size / 1024 / 1024:.1f} MB)...",
        flush=True,
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        tmp_dir = Path(tempfile.mkdtemp(prefix="chroma_dl_"))
        try:
            zip_path = tmp_dir / "chroma.zip"
            blob.download_to_filename(str(zip_path))

            actual_size = zip_path.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"downloaded {actual_size} bytes, expected {expected_size} "
                    "- transfer looks incomplete."
                )

            with zipfile.ZipFile(zip_path) as zf:
                bad_entry = zf.testzip()
                if bad_entry is not None:
                    raise IOError(f"zip integrity check failed on entry '{bad_entry}' - download is corrupted.")

                for info in zf.infolist():
                    # Defensive: zips built on Windows can store paths with
                    # backslashes, which zipfile only treats as a directory
                    # separator on Windows itself. Normalize so extraction
                    # is correct regardless of platform either way.
                    normalized_name = info.filename.replace("\\", "/")
                    target = tmp_dir / normalized_name
                    if normalized_name.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, open(target, "wb") as dest:
                        shutil.copyfileobj(source, dest)

            extracted_root = tmp_dir / chroma_path.name
            if not extracted_root.exists():
                raise RuntimeError(
                    f"expected '{chroma_path.name}/' inside the downloaded zip, "
                    f"found: {[p.name for p in tmp_dir.iterdir()]}"
                )
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            shutil.move(str(extracted_root), str(chroma_path))
            break
        except Exception as e:
            print(f"[chroma] attempt {attempt}/{max_attempts} failed: {e}", flush=True)
            shutil.rmtree(chroma_path, ignore_errors=True)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"[chroma] failed to download a valid store after {max_attempts} attempts."
                ) from e
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not _is_populated(chroma_path):
        raise RuntimeError(
            f"[chroma] extraction completed but the store at {chroma_path} has no "
            f"documents in collection '{settings.CHROMA_COLLECTION}'."
        )
    print(f"[chroma] store ready at {chroma_path}.", flush=True)


def load_vector_store() -> Chroma:
    """Build a vector base on Chroma. As an embedding function, we use Google's embedding API"""
    _ensure_chroma_store()
    return Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDINGS_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=settings.EMBEDDINGS_DIMENSIONS,
        ),
    )


class Retriever:
    """Retriever class to search jobs into a Chroma vector store."""

    def __init__(self):
        self.vector_store = load_vector_store()

    def search(self, query: str, k: int = 4) -> List[Document]:
        kits = self.vector_store.similarity_search(query=query, k=k)

        return kits
