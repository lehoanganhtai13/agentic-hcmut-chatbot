import asyncio
import  traceback
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException

from vllm_client import VLLMClient
from models import RerankRequest, RerankResponse, RerankItem


# ------- Initialize FastAPI app -------

# Global variable to store the client
reranking_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global reranking_client
    logger.info("🚀 Initializing reranking client...")
    reranking_client = VLLMClient()
    logger.info("Reranking client initialized successfully")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")
    del reranking_client
    logger.info("Reranking client shutdown complete")

app = FastAPI(
    title="vLLM Reranking Server",
    version="1.0.0",
    description="Host AITeamVN/Vietnamese_Reranker via vLLM with OpenAI-compatible schema",
    lifespan=lifespan,
)

# ------- Define API endpoints -------

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server status.
    """
    return {"status": "ok", "model": "AITeamVN/Vietnamese_Reranker"}

@app.post("/v1/reranking", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on a query.

    This endpoint follows OpenAI schema:
    {
      "data": [
        {"object": "score", "index": 0, "score": 0.95},
        {"object": "score", "index": 1, "score": 0.90},
        ...
      ]
    }
    """
    try:
        try:
            scores = await asyncio.to_thread(
                reranking_client.score,
                query=request.query,
                documents=request.documents
            )
        except Exception as e:
            logger.error(f"Error reranking documents: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

        # Create items with original indices and scores
        items_with_scores = [
            RerankItem(index=i, score=score) 
            for i, score in enumerate(scores)
        ]
        
        # Sort by score in descending order (highest score first)
        sorted_items = sorted(items_with_scores, key=lambda x: x.score, reverse=True)

        return RerankResponse(data=sorted_items)
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
