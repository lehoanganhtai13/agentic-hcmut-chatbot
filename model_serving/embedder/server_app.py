import asyncio
import  traceback
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException

from vllm_client import VLLMClient
from models import EmbeddingRequest, EmbeddingResponse, EmbeddingItem


# ------- Initialize FastAPI app -------

# Global variable to store the client
embedding_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_client
    logger.info("🚀 Initializing embedding client...")
    embedding_client = VLLMClient()
    logger.info("Embedding client initialized successfully")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")
    del embedding_client
    logger.info("Embedding client shutdown complete")

app = FastAPI(
    title="vLLM Embedding Server",
    version="1.0.0",
    description="Host AITeamVN/Vietnamese_Embedding_v2 via vLLM with OpenAI-compatible schema",
    lifespan=lifespan,
)

# ------- Define API endpoints -------

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server status.
    """
    return {"status": "ok", "model": "AITeamVN/Vietnamese_Embedding_v2"}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for provided inputs.

    This endpoint follows OpenAI schema:
    {
      "data": [
        {"object": "embedding", "index": 0, "embedding": [...]},
        ...
      ]
    }
    """
    try:
        try:
            embeddings = await asyncio.to_thread(
                embedding_client.encode,
                texts=request.inputs
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

        items = [EmbeddingItem(index=i, embedding=vec) for i, vec in enumerate(embeddings)]
        return EmbeddingResponse(data=items)
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
