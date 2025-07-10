import httpx
import tiktoken
from typing import Any, List, Literal
from loguru import logger
from openai import OpenAI, AsyncOpenAI, APIError

from chatbot.core.model_clients.embedder.base_embedder import BaseEmbedder
from chatbot.core.model_clients.embedder.openai.config import OpenAIClientConfig
from chatbot.core.model_clients.embedder.exceptions import CallServerEmbedderError
from chatbot.utils.embeddings import DenseEmbedding

# Define a type for clarity
InputType = Literal["query", "text"]


class OpenAIEmbedder(BaseEmbedder):
    """
    An embedder that uses either the official OpenAI API or an OpenAI-compatible API.

    This class handles the logic for generating embeddings for queries and documents,
    supporting both synchronous and asynchronous operations. It can be configured
    to point to different endpoints, making it suitable for production environments
    that might use self-hosted models (like vLLM) or other cloud services.
    """

    def __init__(self, config: OpenAIClientConfig, **kwargs):
        # The config object is now more specific
        self.config: OpenAIClientConfig
        super().__init__(config, **kwargs)

    def _initialize_embedder(self, **kwargs) -> None:
        """
        Initializes the appropriate synchronous and asynchronous clients
        based on the provided configuration.
        """
        self._sync_client: OpenAI | httpx.Client
        self._async_client: AsyncOpenAI | httpx.AsyncClient
        self._encoding = None

        if self.config.use_openai_client:
            # Use official OpenAI clients
            self._sync_client = OpenAI(api_key=self.config.api_key)
            self._async_client = AsyncOpenAI(api_key=self.config.api_key)
            # Tiktoken is specific to official OpenAI models
            self._encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Use generic httpx clients for compatible APIs
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._sync_client = httpx.Client(base_url=self.config.base_url, headers=headers, timeout=60.0)
            self._async_client = httpx.AsyncClient(base_url=self.config.base_url, headers=headers, timeout=60.0)

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the text using tiktoken.
        This is primarily useful for official OpenAI models to monitor usage.
        
        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        
        Raises:
            ValueError: If the tokenizer is not available (i.e., not using an official OpenAI client).
        """
        if not self._encoding:
            raise ValueError("Token counting is only available when `use_openai_client` is True.")
        return len(self._encoding.encode(text))

    def _embed_sync(self, texts: List[str], input_type: InputType) -> List[DenseEmbedding]:
        """Helper method for synchronous embedding generation."""
        try:
            if isinstance(self._sync_client, OpenAI):
                response = self._sync_client.embeddings.create(
                    input=texts,
                    model=self.config.model_id,
                    dimensions=self.config.model_dimensions
                )
                return [res.embedding for res in response.data]
            
            # Logic for compatible clients using httpx.Client
            elif isinstance(self._sync_client, httpx.Client):
                endpoint = (
                    self.config.query_embedding_endpoint 
                    if input_type == "query" 
                    else self.config.doc_embedding_endpoint
                )
                payload = {"input": texts, "model": self.config.model_id}
                
                response = self._sync_client.post(endpoint, json=payload)
                response.raise_for_status() # Raise an exception for 4xx/5xx responses
                
                # The response structure should match OpenAI's API
                return [item["embedding"] for item in response.json()["data"]]
            
            else:
                raise TypeError(f"Unsupported synchronous client type: {type(self._sync_client)}")

        except (APIError, httpx.HTTPStatusError) as e:
            raise CallServerEmbedderError(f"API call failed: {e!s}") from e
        except Exception as e:
            raise CallServerEmbedderError(f"An unexpected error occurred during embedding: {e!s}") from e

    async def _embed_async(self, texts: List[str], input_type: InputType) -> List[DenseEmbedding]:
        """Helper method for asynchronous embedding generation."""
        try:
            if isinstance(self._async_client, AsyncOpenAI):
                response = await self._async_client.embeddings.create(
                    input=texts,
                    model=self.config.model_id,
                    dimensions=self.config.model_dimensions
                )
                return [res.embedding for res in response.data]
            
            # Logic for compatible clients using httpx.AsyncClient
            elif isinstance(self._async_client, httpx.AsyncClient):
                endpoint = (
                    self.config.query_embedding_endpoint 
                    if input_type == "query" 
                    else self.config.doc_embedding_endpoint
                )
                payload = {"input": texts, "model": self.config.model_id}

                response = await self._async_client.post(endpoint, json=payload)
                response.raise_for_status() # Raise an exception for 4xx/5xx responses

                return [item["embedding"] for item in response.json()["data"]]

            else:
                raise TypeError(f"Unsupported asynchronous client type: {type(self._async_client)}")

        except (APIError, httpx.HTTPStatusError) as e:
            raise CallServerEmbedderError(f"Asynchronous API call failed: {e!s}") from e
        except Exception as e:
            raise CallServerEmbedderError(f"An unexpected error occurred during async embedding: {e!s}") from e

    # --- Synchronous Methods ---

    def get_query_embedding(self, query: str, **kwargs: Any) -> DenseEmbedding:
        """Get the embedding for a single query."""
        if self._encoding and self.config.count_tokens:
            logger.info(f"📊 Number of tokens in query: {self.count_tokens(query)}")
        embeddings = self._embed_sync([query], input_type="query")
        return embeddings[0]

    def get_text_embedding(self, text: str, **kwargs: Any) -> DenseEmbedding:
        """Get the embedding for a single text document."""
        if self._encoding and self.config.count_tokens:
            logger.info(f"📊 Number of tokens in text: {self.count_tokens(text)}")
        embeddings = self._embed_sync([text], input_type="text")
        return embeddings[0]

    def get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[DenseEmbedding]:
        """Get the embeddings for a list of text documents."""
        if self._encoding and self.config.count_tokens:
            total_tokens = sum(self.count_tokens(t) for t in texts)
            logger.info(f"📊 Total number of tokens in texts: {total_tokens}")
        return self._embed_sync(texts, input_type="text")

    # --- Asynchronous Methods ---

    async def aget_query_embedding(self, query: str, **kwargs: Any) -> DenseEmbedding:
        """Asynchronously get the embedding for a single query."""
        if self._encoding and self.config.count_tokens:
            logger.info(f"📊 Number of tokens in query: {self.count_tokens(query)}")
        embeddings = await self._embed_async([query], input_type="query")
        return embeddings[0]

    async def aget_text_embedding(self, text: str, **kwargs: Any) -> DenseEmbedding:
        """Asynchronously get the embedding for a single text document."""
        if self._encoding and self.config.count_tokens:
            logger.info(f"📊 Number of tokens in text: {self.count_tokens(text)}")
        embeddings = await self._embed_async([text], input_type="text")
        return embeddings[0]
        
    async def aget_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[DenseEmbedding]:
        """Asynchronously get the embeddings for a list of text documents."""
        if self._encoding and self.config.count_tokens:
            total_tokens = sum(self.count_tokens(t) for t in texts)
            logger.info(f"📊 Total number of tokens in texts: {total_tokens}")
        return await self._embed_async(texts, input_type="text")
    