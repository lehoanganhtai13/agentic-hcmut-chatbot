from enum import Enum
from typing import AsyncGenerator, Generator, Optional
from pydantic import BaseModel


class LLMBackend(Enum):
    """Enum for different LLM backends."""
    OPENAI = "openai"
    LITELLM = "litellm"
    GOOGLE = "google"


class LLMConfig:
    """Base configuration for LLM."""
    def __init__(self, backend: LLMBackend, max_tokens: int, **kwargs):
        self.backend = backend
        self.max_tokens = max_tokens
        self.config = kwargs


class CompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]
