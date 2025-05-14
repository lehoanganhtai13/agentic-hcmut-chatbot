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
from chatbot.core.retriever import FAQRetriever
from chatbot.core.retriever.base_class import FAQRetrievalResult
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients import VectorDatabase

# ------------------- Global Model Classes -------------------

class FAQRetrievalOutput(BaseModel):
    """Class to store FAQ retrieval response data."""
    status: str = Field(..., description="The status of the retrieval operation.")
    results: Optional[FAQRetrievalResult] = Field(None, description="The retrieved FAQs.")
    message: Optional[str] = Field(None, description="An error message if the operation failed.")


# ------------------- Init Server API -------------------

mcp = FastMCP(
    name="FAQ Retrieval Server",
    instructions="This is a server for retrieving relevant FAQs about HCMUT.",
    tags=["faq"]
)

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
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())

# ------------------- API Endpoints -------------------

@mcp.tool(
    name="faq_retrieval_tool",
    description="Retrieve top K relevant FAQs based on the query.",
    tags=["faq"]
)
async def retrieve(
    query: str = Field(..., description="The query string for FAQ retrieval."),
    top_k: int = Field(5, description="The number of top FAQs to retrieve.")
) -> FAQRetrievalOutput:
    try:
        # Retrieve relevant FAQs
        results = retriever.retrieve_faqs(
            query=query,
            top_k=top_k
        )
        return FAQRetrievalOutput(status="success", results=results)
    except Exception as e:
        logger.error(f"Error retrieving relevant FAQs: {e}")
        logger.error(traceback.format_exc())
        return FAQRetrievalOutput(status="error", message=str(e))
    
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
    asyncio.run(
        mcp.run_async(
            transport="sse",
            host="0.0.0.0",
            port=8000,
            uvicorn_config={"workers": 4}
        )
    )
