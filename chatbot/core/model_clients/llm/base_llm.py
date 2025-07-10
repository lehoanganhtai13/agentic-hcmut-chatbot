from abc import ABC, abstractmethod
from typing import Any

from chatbot.core.model_clients.llm.base_class import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMConfig
)


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model (LLM) implementations.
    This class provides a standardized interface for interacting with various LLM providers
    and models. It supports both synchronous and asynchronous operations, as well as 
    streaming and non-streaming completions.
    The LLM receives prompts (queries) and generates corresponding text responses based on
    the model's training and configuration.
    
    Attributes:
        config (LLMConfig): Configuration object containing LLM settings and parameters.
        
    Methods:
        **complete**: Generate a single completion response for a given prompt synchronously.
        **stream_complete**: Generate a streaming completion response synchronously.
        **acomplete**: Generate a single completion response for a given prompt asynchronously.
        **astream_complete**: Generate a streaming completion response asynchronously.

    Example:
        >>> llm = SomeLLMImplementation(config)
        >>> response = llm.complete("What is machine learning?")
        >>> print(response.text)
        >>>
        >>> async for chunk in llm.astream_complete("Explain AI"):
        ...     print(chunk.delta, end="")
    """

    def __init__(self, config: LLMConfig, **kwargs):
        self.config = config
        self._initialize_llm(**kwargs)
    
    @abstractmethod
    def _initialize_llm(self, **kwargs) -> None:
        """Initialize the LLM."""
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate a single completion response for a given prompt synchronously.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            CompletionResponse: The response containing the generated text.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Generate a streaming completion response for a given prompt synchronously.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            CompletionResponseGen: A generator yielding the response chunks.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate a single completion response for a given prompt asynchronously.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            CompletionResponse: The response containing the generated text.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        """
        Generate a streaming completion response for a given prompt asynchronously.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            CompletionResponseAsyncGen: A generator yielding the response chunks.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
