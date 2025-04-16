import json
import os
import tempfile
import traceback
import uuid
from typing import List, Optional

import uvicorn
from contextlib import asynccontextmanager
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

class IndexingOutput(BaseModel):
    """Class to store indexing operation response data."""

    status: str
    message: str
    file_count: int = 0
    indexed_documents: int = 0


# ------------------- Server API -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexer

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

    # Initialize the configuration for preprocessing before chunking
    preprocessing_config = PreprocessingConfig()

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host=SETTINGS.MILVUS_URL.replace("http://", "").split(":")[0],
        port=int(SETTINGS.MILVUS_URL.split(":")[-1]),
        run_async=False
    )
    
    indexer = DataIndex(
        llm=llm,
        embedder=embedder,
        document_bm25_client=document_bm25_client,
        faq_bm25_client=faq_bm25_client,
        preprocessing_config=preprocessing_config,
        vector_db=vector_db
    )

    yield
    logger.info("Shutting down Data Indexing server...")
    del indexer


app = FastAPI(lifespan=lifespan)


@app.post("/upload_index", response_model=IndexingOutput)
async def upload_and_index(files: List[UploadFile] = File(...)):
    """
    Upload files and index them into the database.
    
    Args:
        files (List[UploadFile]): List of files to upload and index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    temp_files = []
    documents = []
    FAQs = []

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
        
        # Run indexing
        await run_indexing(temp_files, documents, FAQs)
        
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

async def run_indexing(temp_files: List[str], documents: List[str], FAQs: Optional[List[FAQDocument]] = None):
    try:
        result = indexer.run(
            documents=documents,
            faqs=FAQs,
            document_collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
            faq_collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME
        )
        logger.info(f"Indexing completed successfully: {len(documents)} documents, {len(FAQs)} FAQs")

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
