from chatbot.core.model_clients.embedder.base_class import (
    EmbedderBackend,
    EmbedderConfig
)


class OpenAIClientConfig(EmbedderConfig):
    """
    Configuration for OpenAI and compatible embedding clients.

    This configuration supports both direct calls to the OpenAI API and calls
    to other OpenAI-compatible endpoints (e.g., self-hosted vLLM, HuggingFace TGI).

    Attributes:
        api_key (str, optional): 
            API key for the embedding service. Defaults to None.
        model_id (str): 
            The ID of the embedding model to use. 
            Defaults to "text-embedding-3-small".
        model_dimensions (int, optional): 
            The desired dimensions of the output embeddings.
            Only applicable for certain OpenAI models. Defaults to 1536.
        use_openai_client (bool): 
            If True, use the official OpenAI Python client. 
            If False, use a generic HTTP client to call a compatible API endpoint.
            Defaults to True.
        base_url (str, optional): 
            The base URL for the compatible API. Required if `use_openai_client` is False.
            Defaults to "http://localhost:8000/v1".
        query_embedding_endpoint (str):
            The specific endpoint for generating query embeddings.
            Appended to `base_url`. Defaults to "/embeddings".
        doc_embedding_endpoint (str):
            The specific endpoint for generating document embeddings.
            Appended to `base_url`. Defaults to "/embeddings".
        count_tokens (bool):
            If True, count the number of tokens in the input text.
            Defaults to False.
    """
    def __init__(
        self,
        api_key: str = None,
        model_id: str = "text-embedding-3-small",
        model_dimensions: int = 1536,
        use_openai_client: bool = True,
        base_url: str = "http://localhost:8000/v1",
        query_embedding_endpoint: str = "/embeddings",
        doc_embedding_endpoint: str = "/embeddings",
        count_tokens: bool = False,
        **kwargs
    ):
        super().__init__(EmbedderBackend.OPENAI, **kwargs)
        self.api_key = api_key
        self.model_id = model_id
        self.model_dimensions = model_dimensions
        self.use_openai_client = use_openai_client
        self.base_url = base_url
        self.query_embedding_endpoint = query_embedding_endpoint
        self.doc_embedding_endpoint = doc_embedding_endpoint
        self.count_tokens = count_tokens
