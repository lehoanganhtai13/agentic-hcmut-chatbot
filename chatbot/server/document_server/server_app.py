from typing import Optional

import json
import uvicorn
import traceback
from contextlib import asynccontextmanager
from minio import Minio
from pydantic import BaseModel
from fastapi import FastAPI
from loguru import logger

from chatbot.config.system_config import SETTINGS
from chatbot.core.model_clients import BM25Client
from chatbot.core.model_clients.load_model import init_embedder
from chatbot.core.retriever import DocumentRetriever, FAQRetriever
from chatbot.core.retriever.base_class import DocumentRetrievalResult, FAQRetrievalResult
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients import VectorDatabase

# ------------------- Global Model Classes -------------------

class DocumentRetrievalRequest(BaseModel):
    """Class to store document retrieval request data."""
    query: str
    top_k: int = 2


class DocumentRetrievalOutput(BaseModel):
    """Class to store document retrieval response data."""

    status: str
    results: Optional[DocumentRetrievalResult] = None
    message: Optional[str] = None


# ------------------- Server API -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever

    logger.info("Starting up Document Retrieval server...")
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        models_config = ModelsConfig.from_dict(models_config)

    # Initialize the embedder
    embedder = init_embedder(models_conf=models_config)

    # Initialize the MinIO client for loading BM25 state dicts
    minio_client = Minio(
        endpoint=SETTINGS.MINIO_URL.replace("http://", "").replace("https://", ""),
        access_key=SETTINGS.MINIO_ACCESS_KEY_ID,
        secret_key=SETTINGS.MINIO_SECRET_ACCESS_KEY,
        secure=False
    )
    bm25_client = BM25Client(
        storage=minio_client,
        bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
        init_without_load=False,
        remove_after_load=True
    )

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host=SETTINGS.MILVUS_URL.replace("http://", "").split(":")[0],
        port=int(SETTINGS.MILVUS_URL.split(":")[-1]),
        run_async=False
    )
    
    retriever = DocumentRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
        embedder=embedder,
        bm25_client=bm25_client,
        vector_db=vector_db
    )

    yield
    logger.info("Shutting down Document Retrieval server...")
    del retriever


app = FastAPI(lifespan=lifespan)


@app.post("/retrieve", response_model=DocumentRetrievalOutput)
async def retrieve(request: DocumentRetrievalRequest):
    """Retrieve relevant documents."""
    try:
        # Retrieve relevant documents
        results = retriever.retrieve_documents(
            query=request.query,
            top_k=request.top_k
        )
        return DocumentRetrievalOutput(status="success", results=results)
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {e}")
        logger.error(traceback.format_exc())
        return DocumentRetrievalOutput(status="error", message=str(e))


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("chatbot.document_server.server_app:app", host="0.0.0.0", port=8000)
