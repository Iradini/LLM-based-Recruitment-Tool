import time
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm

from backend.config import settings


class ETLProcessor:
    """
    This class is responsible for performing an Extract-Transform-Load (ETL)
    process for document embedding.
    """

    def __init__(
        self,
        batch_size: int,
        chunk_size: int,
        chunk_overlap: int,
        dataset_path: Optional[str] = settings.DATASET_PATH,
        embedding_model: Optional[str] = settings.EMBEDDINGS_MODEL,
        collection_name: Optional[str] = settings.CHROMA_COLLECTION,
        persist_directory: Optional[str] = settings.CHROMA_DB_PATH,
    ):
        """
        Initializes the ETLProcessor object with a specified batch_size.

        Parameters
        ----------
        batch_size : int
            Number of documents to process in each batch, e.g. 100.

        chunk_size : int
            Size of chunks to be split into, e.g. 500.

        chunk_overlap : int
            Number of characters to overlap between chunks, e.g. 20.

        dataset_path : str, optional
            Path to csv file containing the dataset.

        embedding_model : str, optional
            Name of the Google embedding model to be used, e.g.
            "gemini-embedding-001".

        collection_name : str, optional
            Name of the collection to be used in the vector store.

        persist_directory : str, optional
            Directory to persist the vector store.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.embedding = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        self.collection_name = collection_name
        self.persist_directory = persist_directory        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def load_data(self) -> pd.DataFrame:
        """
        Loads the jobs descriptions from a csv file.

        Returns
        -------
        df : pd.DataFrame
            Jobs descriptions with extra metadata from the dataset.
        """
        columns = [
            "description", "Employment type", "Seniority level", "company", "location", "post_url", "title"
        ]
        
        df = pd.read_csv(self.dataset_path)
        df = df[columns].dropna(subset=columns)
        return df

    def create_documents(self, descriptions: pd.DataFrame) -> List[Document]:
        """
        Creates a list of Document objects from given descriptions.

        Parameters
        ----------
        descriptions : pd.DataFrame
            Job descriptions to be converted to Document objects.

        Returns
        -------
        List[Document]
            List of Document (langchain.docstore.document.Document) objects.
        """
        output_documents = []
        for idx, row in descriptions.iterrows():
            metadata = {
                "employment_type": row["Employment type"],
                "seniority_level": row["Seniority level"],
                "company": row["company"],
                "location": row["location"],
                "post_url": row["post_url"],
                "title": row["title"],
                "id": idx,
            }
            doc = Document(page_content=row["description"], metadata=metadata)
            output_documents.append(doc)

        return output_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks using the pre-defined
        text_splitter.

        Parameters
        ----------
        documents : List[Document]
            List of Document objects to be split.

        Returns
        -------
        List[Document]
            List of split Document objects.
        """
        return self.text_splitter.split_documents(documents)

    # Error statuses that retrying can't fix: bad model name, bad/revoked key,
    # missing permissions, or a malformed request. Everything else (rate
    # limits, dropped connections, timeouts, transient 5xx errors) is retried.
    _PERMANENT_ERROR_STATUSES = (
        "NOT_FOUND",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "INVALID_ARGUMENT",
        "UNIMPLEMENTED",
    )

    def _add_batch_with_retry(self, batch: List[Document], max_wait: int = 90) -> None:
        """
        Embeds and stores a single batch, retrying with exponential backoff
        on transient failures (rate limits, network errors, etc). Permanent
        errors (e.g. an invalid model name or API key) are raised immediately.
        """
        delay = 5
        while True:
            try:
                Chroma.from_documents(
                    batch,
                    embedding=self.embedding,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                )
                return
            except Exception as e:
                if any(status in str(e) for status in self._PERMANENT_ERROR_STATUSES):
                    raise
                tqdm.write(f"Transient error ({e}); retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_wait)

    def process_batches(self, splits: List[Document], start_index: int = 0) -> None:
        """
        Processes documents in batches, creating Chroma vector stores for
        each batch.

        Parameters
        ----------
        splits : List[Document]
            List of Document objects to be processed.

        start_index : int, optional
            Number of already-processed chunks to skip, for resuming an
            interrupted run. Default is 0 (process everything).

        Returns
        -------
        None
        """
        remaining = splits[start_index:]
        for i in tqdm(
            range(0, len(remaining), self.batch_size), desc="Processing batches"
        ):
            batch = remaining[i: i + self.batch_size]
            self._add_batch_with_retry(batch)

    def run_etl(self, start_index: int = 0) -> None:
        """
        Executes the ETL process: extract data from a source, transform it,
        and load into a new storage.

        Parameters
        ----------
        start_index : int, optional
            Number of already-processed chunks to skip, for resuming an
            interrupted run. Default is 0 (process everything).
        """
        job_descriptions = self.load_data()
        docs = self.create_documents(job_descriptions)
        splits = self.split_documents(docs)
        self.process_batches(splits, start_index=start_index)


if __name__ == "__main__":
    import chromadb

    etl_processor = ETLProcessor(
        batch_size=32,
        chunk_size=500,
        chunk_overlap=100,
    )

    # Resume support: if a previous run was interrupted, skip the chunks
    # already embedded in the persisted collection instead of starting over.
    start_index = 0
    try:
        client = chromadb.PersistentClient(path=etl_processor.persist_directory)
        collection = client.get_collection(etl_processor.collection_name)
        start_index = collection.count()
        if start_index:
            print(f"Resuming: {start_index} chunks already embedded, skipping them.")
    except Exception:
        pass  # no existing collection yet, start from the beginning

    etl_processor.run_etl(start_index=start_index)
