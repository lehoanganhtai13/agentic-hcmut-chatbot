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
from chatbot.core.retriever import FAQRetriever
from chatbot.core.retriever.base_class import FAQRetrievalResult
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients import VectorDatabase

# ------------------- Global Model Classes -------------------

class FAQRetrievalRequest(BaseModel):
    """Class to store FAQ retrieval request data."""
    query: str
    top_k: int = 2


class FAQRetrievalOutput(BaseModel):
    """Class to store FAQ retrieval response data."""

    status: str
    results: Optional[FAQRetrievalResult] = None
    message: Optional[str] = None


# ------------------- Server API -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = None

    yield
    logger.info("Shutting down FAQ Retrieval server...")
    del retriever


app = FastAPI(lifespan=lifespan)


@app.post("/load")
async def load():
    """Load the FAQ retrieval model."""
    global retriever 
    try:
        # Load the FAQ retrieval model
        logger.info("Starting up FAQ Retrieval server...")
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
            bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
            init_without_load=False,
            remove_after_load=True
        )

        # Initialize the vector database client
        vector_db = VectorDatabase(
            host=SETTINGS.MILVUS_URL.replace("http://", "").split(":")[0],
            port=int(SETTINGS.MILVUS_URL.split(":")[-1]),
            run_async=False
        )

        # Initialize the FAQ retriever
        retriever = FAQRetriever(
            collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
            embedder=embedder,
            bm25_client=bm25_client,
            vector_db=vector_db
        )
        logger.info("FAQ retriever initialized successfully.")
        return {"status": "success", "message": "FAQ retrieval model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.post("/retrieve", response_model=FAQRetrievalOutput)
async def retrieve(request: FAQRetrievalRequest):
    """Retrieve relevant FAQs."""
    try:
        # Retrieve relevant FAQs
        results = retriever.retrieve_faqs(
            query=request.query,
            top_k=request.top_k
        )
        return FAQRetrievalOutput(status="success", results=results)
    except Exception as e:
        logger.error(f"Error retrieving relevant FAQs: {e}")
        logger.error(traceback.format_exc())
        return FAQRetrievalOutput(status="error", message=str(e))


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("chatbot.faq_server.server_app:app", host="0.0.0.0", port=8000)
