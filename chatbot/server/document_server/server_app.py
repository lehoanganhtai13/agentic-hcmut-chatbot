from typing import Optional

import json
import traceback
from contextlib import asynccontextmanager
from minio import Minio
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from loguru import logger

from chatbot.config.system_config import SETTINGS
from chatbot.core.model_clients import BM25Client
from chatbot.core.model_clients.load_model import init_embedder
from chatbot.core.retriever import DocumentRetriever
from chatbot.core.retriever.base_class import DocumentRetrievalResult
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients import VectorDatabase

# ------------------- Global Model Classes -------------------

class DocumentRetrievalOutput(BaseModel):
    """Class to store document retrieval response data."""
    status: str = Field(..., description="The status of the retrieval operation.")
    results: Optional[DocumentRetrievalResult] = Field(None, description="The retrieved documents.")
    message: Optional[str] = Field(None, description="An error message if the operation failed.")


# -------------------Init Server API -------------------

async def load():
    """Load the document retrieval model."""
    global retriever
    try:
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
        logger.info("Document retrieval model loaded successfully.")
        return {"status": "success", "message": "Document retrieval model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading document retrieval model: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@asynccontextmanager
async def lifespan(app: FastMCP):
    global retriever
    retriever = None
    try:
        await load()  # Initialize the retriever
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())
        return

    yield
    logger.info("Shutting down Document Retrieval server...")
    del retriever


mcp = FastMCP(
    name="Document Retrieval Server",
    lifespan=lifespan,
    instructions="This is a server for retrieving relevant documents about HCMUT.",
    tags=["document"]
)

# ------------------- API Endpoints -------------------


@mcp.tool(
    name="faq_document_tool",
    description="Retrieve top K relevant documents based on the query.",
    tags=["document"]
)
async def retrieve(
    query: str = Field(..., description="The query string for document retrieval."),
    top_k: int = Field(5, description="The number of top documents to retrieve.")
):
    try:
        # Retrieve relevant documents
        results = retriever.retrieve_documents(
            query=query,
            top_k=top_k
        )
        return DocumentRetrievalOutput(status="success", results=results)
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {e}")
        logger.error(traceback.format_exc())
        return DocumentRetrievalOutput(status="error", message=str(e))
    
# ------------------- MCP Server Check -------------------

async def check_mcp(mcp: FastMCP):
    # List the components that were created
    tools = await mcp.get_tools()
    resources = await mcp.get_resources()
    templates = await mcp.get_resource_templates()

    data_log = f"""
    Tools: {len(tools)} Tool(s): {', '.join([t.name for t in tools.values()])}
    Resources: {len(resources)} Resource(s): {', '.join([r.name for r in resources.values()])}
    Templates: {len(templates)} Resource Template(s): {', '.join([t.name for t in templates.values()])}
    """
    logger.info(data_log)

if __name__ == "__main__":
    import asyncio

    # Run quick check on the MCP server
    asyncio.run(check_mcp(mcp))

    # Start the FastMCP server
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
