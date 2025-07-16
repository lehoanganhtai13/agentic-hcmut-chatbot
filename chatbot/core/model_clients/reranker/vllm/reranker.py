import httpx
from typing import Any, List, Tuple
from loguru import logger

from chatbot.core.model_clients.reranker.base_reranker import BaseReranker
from chatbot.core.model_clients.reranker.vllm.config import VLLMRerankerConfig
from chatbot.core.model_clients.reranker.exceptions import (
    CallServerRerankerError,
    RerankerTimeoutError,
    RerankerConfigurationError
)


class VLLMReranker(BaseReranker):
    """
    A reranker client that interfaces with vLLM-based reranking servers.

    This class handles communication with vLLM reranking servers that expose
    reranking functionality through HTTP endpoints, supporting both synchronous
    and asynchronous operations with comprehensive error handling and retry logic.
    """

    def __init__(self, config: VLLMRerankerConfig, **kwargs):
        self.config: VLLMRerankerConfig
        super().__init__(config, **kwargs)

    def _initialize_reranker(self, **kwargs) -> None:
        """
        Initialize the HTTP clients for synchronous and asynchronous operations.
        
        Sets up httpx clients with appropriate headers, timeout, and retry
        configurations based on the provided configuration.
        """
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Initialize clients with retry transport
        self._sync_client = httpx.Client(
            headers=headers,
            timeout=self.config.timeout,
            transport=httpx.HTTPTransport(retries=self.config.max_retries)
        )
        
        self._async_client = httpx.AsyncClient(
            headers=headers,
            timeout=self.config.timeout,
            transport=httpx.AsyncHTTPTransport(retries=self.config.max_retries)
        )

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the reranker configuration.
        
        Raises:
            RerankerConfigurationError: If the configuration is invalid.
        """
        if not self.config.base_url:
            raise RerankerConfigurationError("base_url cannot be empty")
        
        if not self.config.rerank_endpoint:
            raise RerankerConfigurationError("rerank_endpoint cannot be empty")
        
        if self.config.top_k is not None and self.config.top_k <= 0:
            raise RerankerConfigurationError("top_k must be a positive integer")

    def _prepare_request_payload(self, query: str, documents: List[str]) -> dict:
        """
        Prepare the request payload for the reranking API.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            
        Returns:
            dict: The request payload formatted for the API.
        """
        payload = {
            "query": query,
            "documents": documents
        }
        
        # Add optional parameters if configured
        if self.config.return_documents:
            payload["return_documents"] = True
            
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
            
        return payload

    def _process_response(self, response_data: dict) -> List[Tuple[int, float]]:
        """
        Process the API response and extract reranking results.
        
        Args:
            response_data (dict): The JSON response from the API.
            
        Returns:
            List[Tuple[int, float]]: List of (original_index, score) tuples.
            
        Raises:
            CallServerRerankerError: If the response format is invalid.
        """
        try:
            if "data" not in response_data:
                raise CallServerRerankerError("Invalid response format: missing 'data' field")
            
            results = []
            for item in response_data["data"]:
                if "index" not in item or "score" not in item:
                    raise CallServerRerankerError("Invalid response format: missing 'index' or 'score' field")
                
                results.append((item["index"], item["score"]))
            
            return results
            
        except (KeyError, TypeError) as e:
            raise CallServerRerankerError(f"Failed to parse response: {e}") from e

    def _rerank_sync(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Internal synchronous reranking implementation.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            
        Returns:
            List[Tuple[int, float]]: List of (original_index, score) tuples.
            
        Raises:
            CallServerRerankerError: If the API call fails.
            RerankerTimeoutError: If the request times out.
        """
        try:
            payload = self._prepare_request_payload(query, documents)
            
            logger.debug(f"🔍 Reranking {len(documents)} documents for query: '{query[:100]}...'")
            
            response = self._sync_client.post(self.config.full_url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            results = self._process_response(response_data)
            
            logger.info(f"✅ Successfully reranked {len(documents)} documents, got {len(results)} results")
            return results
            
        except httpx.TimeoutException as e:
            raise RerankerTimeoutError(f"Request timed out after {self.config.timeout}s") from e
        except httpx.HTTPStatusError as e:
            raise CallServerRerankerError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise CallServerRerankerError(f"Unexpected error during reranking: {e}") from e

    async def _rerank_async(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Internal asynchronous reranking implementation.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            
        Returns:
            List[Tuple[int, float]]: List of (original_index, score) tuples.
            
        Raises:
            CallServerRerankerError: If the API call fails.
            RerankerTimeoutError: If the request times out.
        """
        try:
            payload = self._prepare_request_payload(query, documents)
            
            logger.debug(f"🔍 Async reranking {len(documents)} documents for query: '{query[:100]}...'")
            
            response = await self._async_client.post(self.config.full_url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            results = self._process_response(response_data)
            
            logger.info(f"✅ Successfully reranked {len(documents)} documents async, got {len(results)} results")
            return results
            
        except httpx.TimeoutException as e:
            raise RerankerTimeoutError(f"Async request timed out after {self.config.timeout}s") from e
        except httpx.HTTPStatusError as e:
            raise CallServerRerankerError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise CallServerRerankerError(f"Unexpected error during async reranking: {e}") from e

    # --- Synchronous Methods ---

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            **kwargs: Additional keyword arguments (unused for this implementation).
            
        Returns:
            List[Tuple[int, float]]: List of tuples containing (original_index, relevance_score)
                sorted by relevance score in descending order.
                
        Raises:
            CallServerRerankerError: If the reranking API call fails.
            RerankerTimeoutError: If the request times out.
            RerankerConfigurationError: If the configuration is invalid.
        """
        if not query or not query.strip():
            logger.warning("⚠️ Empty query provided for reranking")
            return [(i, 0.0) for i in range(len(documents))]
        
        if not documents:
            logger.warning("⚠️ No documents provided for reranking")
            return []
        
        return self._rerank_sync(query, documents)

    # --- Asynchronous Methods ---

    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """
        Asynchronously rerank documents based on query relevance.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            **kwargs: Additional keyword arguments (unused for this implementation).
            
        Returns:
            List[Tuple[int, float]]: List of tuples containing (original_index, relevance_score)
                sorted by relevance score in descending order.
                
        Raises:
            CallServerRerankerError: If the reranking API call fails.
            RerankerTimeoutError: If the request times out.
            RerankerConfigurationError: If the configuration is invalid.
        """
        if not query or not query.strip():
            logger.warning("⚠️ Empty query provided for async reranking")
            return [(i, 0.0) for i in range(len(documents))]
        
        if not documents:
            logger.warning("⚠️ No documents provided for async reranking")
            return []
        
        return await self._rerank_async(query, documents)

    def __del__(self):
        """Clean up HTTP clients when the object is destroyed."""
        try:
            if hasattr(self, "_sync_client"):
                self._sync_client.close()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Example usage
    config = VLLMRerankerConfig(base_url="http://localhost:8030")
    reranker = VLLMReranker(config)
    
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    
    results = reranker.rerank(query, documents)
    print("Reranked Results:", results)
        