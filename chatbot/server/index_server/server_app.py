import json
import io
import os
import tempfile
import traceback
import uuid
from typing import List

import asyncio
import httpx
import uvicorn
from contextlib import asynccontextmanager
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from minio import Minio
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from loguru import logger

from chatbot.config.system_config import SETTINGS
from chatbot.core.model_clients import BM25Client
from chatbot.core.model_clients.load_model import init_embedder, init_llm
from chatbot.indexing.context_document.base_class import PreprocessingConfig
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.workflow.build_index import DataIndex
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients import VectorDatabase

# ------------------- Global Model Classes -------------------

class InitializationMode(Enum):
    NEW = "new"
    LOAD_EXISTING = "load existing"


class IndexingMode(Enum):
    FULL_INDEX = "full index"
    INSERT = "insert"


class IndexingOutput(BaseModel):
    """Class to store indexing operation response data."""

    status: str
    message: str
    file_count: int = 0
    indexed_documents: int = 0


# ------------------- Server API -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexer, minio_client, llm, embedder, vector_db, preprocessing_config

    logger.info("Starting up Data Indexing server...")
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        models_config = ModelsConfig.from_dict(models_config)

    # Initialize the MinIO client for storing BM25 state dicts
    minio_client = Minio(
        endpoint=SETTINGS.MINIO_URL.replace("http://", "").replace("https://", ""),
        access_key=SETTINGS.MINIO_ACCESS_KEY_ID,
        secret_key=SETTINGS.MINIO_SECRET_ACCESS_KEY,
        secure=False
    )

    # Initialize the LLM and embedder clients
    llm = init_llm(
        task="indexing_llm",
        models_conf=models_config
    )
    embedder = init_embedder(models_conf=models_config)

    # Initialize the configuration for preprocessing before chunking
    preprocessing_config = PreprocessingConfig()

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host=SETTINGS.MILVUS_URL.replace("http://", "").split(":")[0],
        port=int(SETTINGS.MILVUS_URL.split(":")[-1]),
        run_async=False
    )
    
    indexer = None

    yield
    logger.info("Shutting down Data Indexing server...")
    del indexer


app = FastAPI(lifespan=lifespan)


@app.post("/load")
async def load(
    mode: InitializationMode
):
    """
    Load the indexer model.
    
    Args:
        mode (InitializationMode): Mode of initialization (`NEW` and `LOAD_EXISTING`)
    """
    global indexer
    try:
        if mode == InitializationMode.NEW:
            logger.info("Initializing new indexer model...")

            # Initialize new BM25 models
            document_bm25_client = BM25Client(
                storage=minio_client,
                bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
                overwrite_minio_bucket=True
            )
            faq_bm25_client = BM25Client(
                storage=minio_client,
                bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
                overwrite_minio_bucket=True
            )

            # Initialize a new indexer model
            indexer = DataIndex(
                llm=llm,
                embedder=embedder,
                document_bm25_client=document_bm25_client,
                faq_bm25_client=faq_bm25_client,
                preprocessing_config=preprocessing_config,
                vector_db=vector_db
            )
            return {"status": "success", "message": "New indexer model initialized."}
        
        elif mode == InitializationMode.LOAD_EXISTING:
            logger.info("Loading existing indexer model...")

            # Load the existing BM25 models
            document_bm25_client = BM25Client(
                storage=minio_client,
                bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
                init_without_load=False,
                remove_after_load=True
            )
            faq_bm25_client = BM25Client(
                storage=minio_client,
                bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
                init_without_load=False,
                remove_after_load=True
            )

            # Load existing indexer model
            indexer = DataIndex(
                llm=llm,
                embedder=embedder,
                document_bm25_client=document_bm25_client,
                faq_bm25_client=faq_bm25_client,
                preprocessing_config=preprocessing_config,
                vector_db=vector_db
            )
            return {"status": "success", "message": "Existing indexer model loaded."}
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to initialize indexer: {str(e)}")
    

@app.post("/load_weight")
async def load_weight(files: List[UploadFile] = File(...)):
    """
    Load the weight of the indexer model.
    
    Args:
        files (List[UploadFile]): List of files to upload (including `faq_state_dict.json` and `document_state_dict.json`)
    """
    global indexer
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No file provided for upload")
        
        required_files = ["faq_state_dict.json", "document_state_dict.json"]
        # Check if the uploaded files are the expected ones
        if not all(file.filename in required_files for file in files):
            raise HTTPException(status_code=400, detail="Uploaded files must be 'faq_state_dict.json' and 'document_state_dict.json'.")
        
        # Upload the files to MinIO
        for file in files:
            # Determine the bucket name based on the file name
            bucket_name = ""
            if file.filename == "faq_state_dict.json":
                bucket_name = SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME
            elif file.filename == "document_state_dict.json":
                bucket_name = SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME

            # Read the file content
            file_content = await file.read()

            # Clean the bucket if it already exists
            BM25Client(
                storage=minio_client,
                bucket_name=bucket_name,
                overwrite_minio_bucket=True
            )

            minio_client.put_object(
                bucket_name=bucket_name,
                object_name="bm25/state_dict.json",
                data=io.BytesIO(file_content),
                length=len(file_content)
            )
            logger.info(f"Uploaded {file.filename} to MinIO.")

        # Initialize new BM25 models
        document_bm25_client = BM25Client(
            storage=minio_client,
            bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
            init_without_load=False,
            remove_after_load=True
        )
        faq_bm25_client = BM25Client(
            storage=minio_client,
            bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
            init_without_load=False,
            remove_after_load=True
        )

        # Initialize a new indexer model
        indexer = DataIndex(
            llm=llm,
            embedder=embedder,
            document_bm25_client=document_bm25_client,
            faq_bm25_client=faq_bm25_client,
            preprocessing_config=preprocessing_config,
            vector_db=vector_db
        )
        return {"status": "success", "message": "Weights loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load weights: {str(e)}")

@app.post("/upload_index", response_model=IndexingOutput)
async def upload_and_index(
    files: List[UploadFile] = File(default=[]),
    web_urls: List[str] = [],
    indexing_mode: IndexingMode = IndexingMode.FULL_INDEX
):
    """
    Upload files and index them into the database.
    
    Args:
        files (List[UploadFile]): List of files to upload and index.
        web_urls (List[str]): List of web URLs to index.
        indexing_mode (IndexingMode): Mode of indexing (`FULL_INDEX` and `INSERT`)
    """
    if not files and not web_urls:
        raise HTTPException(status_code=400, detail="No files or URLs provided for upload")
    
    temp_files = []
    documents = []
    FAQs = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Process uploaded files based on their type
            for file in files:
                # Create temp file
                suffix = Path(file.filename).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    # Write content to temp file
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)
                    
                # Process based on file type
                if suffix == ".txt":
                    # Process as document
                    logger.info(f"Processing {file.filename} as document")
                    with open(temp_file_path, "r", encoding="utf-8") as doc_file:
                        document_content = doc_file.read()
                        documents.append(document_content)
                        
                elif suffix == ".csv":
                    # Process as FAQ
                    logger.info(f"Processing {file.filename} as FAQ")
                    try:
                        faq_df = pd.read_csv(temp_file_path)
                        
                        # Verify the required columns exist
                        if "query" not in faq_df.columns or "answer" not in faq_df.columns:
                            raise HTTPException(
                                status_code=400, 
                                detail=f"CSV file {file.filename} must contain 'query' and 'answer' columns"
                            )
                        
                        # Convert rows to FAQDocument objects
                        for _, row in faq_df.iterrows():
                            FAQs.append(
                                FAQDocument(
                                    id=str(uuid.uuid4()),
                                    question=row["query"],
                                    answer=row["answer"]
                                )
                            )
                            
                    except pd.errors.EmptyDataError:
                        logger.warning(f"Empty CSV file: {file.filename}")
                    except pd.errors.ParserError as e:
                        logger.error(f"Error parsing CSV file {file.filename}: {str(e)}")
                        raise HTTPException(status_code=400, detail=f"Invalid CSV format in {file.filename}")
                else:
                    logger.warning(f"Skipping unsupported file type: {file.filename}")

            # Process web URLs
            for url in web_urls:
                    logger.info(f"Requesting content for URL: {url}")
                    try:
                        # Call API of web crawler
                        response = await client.get(
                            url="http://web-crawler-server:8000/crawl",
                            params={"url": url}
                        )
                        response.raise_for_status() # Raises an exception if the status code is not 2xx

                        crawler_data = response.json()

                        # Check if the crawler returned an error
                        if crawler_data.get("error"):
                            logger.error(f"Crawler API returned error for {url}: {crawler_data['error']}")
                            # Can decide to stop or skip this URL
                            continue # Skip URL with error

                        # Get content and add to the documents list
                        content = crawler_data.get("content")
                        if content:
                            documents.append(content)
                            logger.info(f"Successfully extracted content for URL: {url} (Length: {len(content)})")
                        else:
                            logger.warning(f"No content extracted by crawler for URL: {url}. Message: {crawler_data.get('message')}")

                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTP error calling crawler for {url}: Status {e.response.status_code} - {e.response.text}")
                        # Can decide to stop or skip this URL
                        continue # Skip URL with error
                    except httpx.RequestError as e:
                        logger.error(f"Network error calling crawler for {url}: {str(e)}")
                        # Can decide to stop or skip this URL
                        continue # Skip URL with error
                    except Exception as e:
                        logger.error(f"Unexpected error processing URL {url}: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Can decide to stop or skip this URL
                        continue # Skip URL with error
            
            # Run indexing
            await run_indexing(temp_files, documents, FAQs, indexing_mode)
            
            return IndexingOutput(
                status="success",
                message=f"Files uploaded and indexed successfully.",
                file_count=len(files),
                indexed_documents=len(documents) + len(FAQs)
            )
        
        except Exception as e:
            # Clean up temp files in case of error
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

async def run_indexing(
    temp_files: List[str],
    documents: List[str] = [],
    FAQs: List[FAQDocument] = [],
    indexing_mode: IndexingMode = IndexingMode.FULL_INDEX
):
    global indexer # Ensure the index is loaded
    if indexer is None:
        logger.error("Indexer is not loaded. Please call /load or /load_weight first.")
        # Don't raise HTTPException here as it's not a direct endpoint
        # Instead, the calling function (upload_and_index) will catch the error if needed
        raise ValueError("Indexer not loaded")
    
    try:
        if indexing_mode == IndexingMode.FULL_INDEX:
            result = await asyncio.to_thread( # Run in a separate thread to avoid blocking
                indexer.run_index,
                documents=documents,
                faqs=FAQs,
                document_collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
                faq_collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME
            )
            logger.info(f"Indexing completed successfully: {len(documents)} documents, {len(FAQs)} FAQs")
        elif indexing_mode == IndexingMode.INSERT:
            result = await asyncio.to_thread( # Run in a separate thread to avoid blocking
                indexer.run_insert,
                documents=documents,
                faqs=FAQs,
                document_collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
                faq_collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME
            )
            logger.info(f"Inserted {len(documents)} documents, {len(FAQs)} FAQs into the index")

        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("chatbot.index_server.server_app:app", host="0.0.0.0", port=8000)
