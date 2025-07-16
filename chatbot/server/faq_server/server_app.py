import os
from typing import Optional

import asyncio
import json
import traceback
from asyncio import Lock
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import JSONResponse

from chatbot.config.system_config import SETTINGS
from chatbot.core.model_clients import BM25Client
from chatbot.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from chatbot.core.model_clients.reranker.vllm import VLLMRerankerConfig, VLLMReranker
from chatbot.core.retriever import FAQRetriever
from chatbot.core.retriever.base_class import FAQRetrievalResult
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig

# ------------------- Global Model Classes -------------------

class FAQRetrievalOutput(BaseModel):
    """Class to store FAQ retrieval response data."""
    status: str = Field(..., description="The status of the retrieval operation.")
    results: Optional[FAQRetrievalResult] = Field(None, description="The retrieved FAQs.")
    message: Optional[str] = Field(None, description="An error message if the operation failed.")


class ServerStatus(BaseModel):
    """Class to store server status response data."""
    status: str = Field(..., description="The status of the server.")
    message: str = Field(..., description="A message providing additional information about the server status.")


# ------------------- Init Server API -------------------

# Create a lock to ensure thread safety when accessing shared resources
rw_lock = Lock()

mcp = FastMCP(
    name="FAQ Retrieval Server",
    instructions="This is a server for retrieving relevant FAQs about HCMUT.",
    include_tags=["faq"]
)

# Global variables - initialized as None
retriever = None
embedder = None
reranker = None
vector_db = None

try:
    # Load the FAQ retrieval model
    logger.info("🚀 Starting up FAQ Retrieval server...")
    
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        embedder_config = ModelsConfig.from_dict(models_config).embedding_config
        reranker_config = ModelsConfig.from_dict(models_config).reranker_config

    # Initialize the embedder
    embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        use_openai_client=(embedder_config.provider == "openai"),
        base_url= embedder_config.base_url,
        query_embedding_endpoint="v1/embeddings",
        doc_embedding_endpoint="v1/embeddings"
    ))

    # Initialize the reranker
    reranker = VLLMReranker(
        config=VLLMRerankerConfig(
            base_url=reranker_config.base_url,
            rerank_endpoint=reranker_config.rerank_endpoint
        )
    )

    # Initialize the vector database client
    vector_db = MilvusVectorDatabase(
        config=MilvusConfig(
            cloud_uri=SETTINGS.MILVUS_CLOUD_URI,
            token=SETTINGS.MILVUS_CLOUD_TOKEN,
            run_async=False
        )
    )
    
    logger.info("FAQ Retrieval Server base components initialized successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())

def check_retriever_ready() -> bool:
    """Check if retriever is ready for queries."""
    return retriever is not None

def initialize_retriever() -> bool:
    """Initialize retriever with BM25 weights. Returns True if successful."""
    global retriever
    
    try:
        # Check if BM25 state dict exists
        bm25_path = "./chatbot/data/bm25/faq/state_dict.json"
        if not os.path.exists(bm25_path):
            logger.warning(f"BM25 state dict not found at {bm25_path}")
            return False

        # Initialize BM25 client
        bm25_client = BM25Client(
            local_path=bm25_path,
            init_without_load=False
        )

        # Initialize retriever
        retriever = FAQRetriever(
            collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
            embedder=embedder,
            bm25_client=bm25_client,
            vector_db=vector_db,
            reranker=reranker
        )
        
        logger.info("FAQ retriever initialized successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        logger.error(traceback.format_exc())
        return False

@mcp.custom_route("/initialize", methods=["POST"])
async def manual_initialize(request: Request):
    """Manually initialize the retriever."""
    async with rw_lock:
        try:
            if initialize_retriever():
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "FAQ retriever initialized successfully."
                    },
                    status_code=200
                )
            else:
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": "Failed to initialize FAQ retriever. Check if BM25 weights are available."
                    },
                    status_code=500
                )
        except Exception as e:
            logger.error(f"Error during manual initialization: {e}")
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Manual initialization failed: {str(e)}"
                },
                status_code=500
            )
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
    async with rw_lock:
        try:
            # Check if retriever is ready
            if not check_retriever_ready():
                # Try to initialize retriever
                if not initialize_retriever():
                    return FAQRetrievalOutput(
                        status="error", 
                        message=(
                            "Retriever not ready. Please ensure indexing has "
                            "been completed and weights are available."
                        )
                    )
                
            # Retrieve relevant FAQs
            results = await asyncio.to_thread(
                retriever.retrieve_faqs,
                query=query,
                top_k=top_k
            )
            return FAQRetrievalOutput(status="success", results=results)
        except Exception as e:
            logger.error(f"Error retrieving relevant FAQs: {e}")
            logger.error(traceback.format_exc())
            return FAQRetrievalOutput(status="error", message=str(e))
    

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint to verify the server is running."""
    retriever_status = "ready" if check_retriever_ready() else "not_ready"

    return JSONResponse(
        content={
            "status": "healthy",
            "message": f"FAQ Retrieval Server is running. Retriever status: {retriever_status}"
        },
        status_code=200
    )
    
@mcp.custom_route("/reload-index", methods=["POST"])
async def reload_index(request: Request):
    """Reload the FAQ index."""
    async with rw_lock:
        try:
            # Load the BM25 state dict from local path
            logger.info("Reloading FAQ index from local path...")

            # Store reference to old client for cleanup
            global retriever
            old_bm25_client = retriever.bm25_client if retriever else None

            # Initialize new BM25 models
            new_bm25_client = BM25Client(
                local_path="./chatbot/data/bm25/faq/state_dict.json",
                init_without_load=False
            )

            # Reinitialize the retriever with the new BM25 client
            retriever = FAQRetriever(
                collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
                embedder=embedder,
                bm25_client=new_bm25_client,
                vector_db=vector_db,
                reranker=reranker
            )

            # Cleanup old client if exists
            if old_bm25_client:
                del old_bm25_client
                logger.info("Old BM25 client cleaned up")

            logger.info("FAQ index reloaded successfully")

            return JSONResponse(
                content={
                    "status": "success",
                    "message": "FAQ index reloaded successfully."
                },
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error reloading FAQ index: {e}")
            return JSONResponse(
                content={
                    "status": "error",
                    "message": str(e)
                },
                status_code=500
            )
    
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
            uvicorn_config={"workers": 6}
        )
    )
