from chatbot.core.model_clients.reranker.base_class import (
    RerankerBackend,
    RerankerConfig
)


class VLLMRerankerConfig(RerankerConfig):
    """
    Configuration for vLLM reranker client.

    This configuration handles connection to a vLLM-based reranking server
    that exposes reranking functionality through HTTP endpoints.

    Attributes:
        base_url (str): 
            The base URL of the vLLM reranker server.
            Defaults to "http://localhost:8000".
        rerank_endpoint (str):
            The specific endpoint for reranking requests.
            Defaults to "/v1/reranking".
        api_key (str, optional):
            API key for authentication if required by the server.
            Defaults to None.
        timeout (float):
            Request timeout in seconds. Defaults to 60.0.
        max_retries (int):
            Maximum number of retry attempts for failed requests.
            Defaults to 3.
        return_documents (bool):
            Whether to return the original documents along with scores.
            Defaults to False.
        top_k (int, optional):
            Maximum number of documents to return after reranking.
            If None, returns all documents. Defaults to None.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        rerank_endpoint: str = "/v1/reranking",
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        return_documents: bool = False,
        top_k: int = None,
        **kwargs
    ):
        super().__init__(RerankerBackend.VLLM, **kwargs)
        self.base_url = base_url.rstrip("/")  # Remove trailing slash
        self.rerank_endpoint = rerank_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.return_documents = return_documents
        self.top_k = top_k

    @property
    def full_url(self) -> str:
        """Get the complete URL for reranking requests."""
        return f"{self.base_url}{self.rerank_endpoint}"