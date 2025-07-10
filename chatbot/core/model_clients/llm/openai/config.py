from typing import Optional
from chatbot.core.model_clients.llm.base_class import LLMBackend, LLMConfig

class OpenAIClientLLMConfig(LLMConfig):
    """
    Configuration for the OpenAI and compatible LLM clients.

    This class centralizes settings for connecting to either the official OpenAI API
    or a compatible endpoint (like a self-hosted vLLM instance).

    Attributes:
        model (str): 
            The ID of the model to use for completions (e.g., "gpt-4o", "meta-llama/Llama-3-8b-chat-hf").
        api_key (str, optional): 
            The API key for authentication. Defaults to None.
        temperature (float): 
            Controls randomness. Lower values make the model more deterministic. 
            Defaults to 0.1.
        max_tokens (int): 
            The maximum number of tokens to generate in the completion. 
            Defaults to 2048.
        use_openai_client (bool): 
            If True, connects directly to the official OpenAI API endpoint.
            If False, uses `base_url` to connect to a compatible API. 
            Defaults to True.
        base_url (str, optional): 
            The base URL of the compatible API. Only used if `use_openai_client` is False.
            Defaults to "https://api.openai.com/v1".
        system_prompt (str, optional): 
            A default system-level instruction for the model, applied to all requests.
            Defaults to None.
    """
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_openai_client: bool = True,
        base_url: Optional[str] = "https://api.openai.com/v1",
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        # We explicitly set the backend to OPENAI for this config type.
        super().__init__(backend=LLMBackend.OPENAI, max_tokens=max_tokens, **kwargs)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.use_openai_client = use_openai_client
        self.base_url = base_url if not use_openai_client else "https://api.openai.com/v1"
        self.system_prompt = system_prompt
