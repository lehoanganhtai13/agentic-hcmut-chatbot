"""
Google Gemini client that follows the same design pattern as BaseLLM.
This client provides a unified interface for synchronous and asynchronous
completions, including streaming capabilities. It supports Google Search
grounding, thinking budgets, and system instructions for persistent context.
This implementation uses the Google Generative AI unified SDK, which
provides a rich set of features and robust error handling.

Requires: pip install --upgrade google-genai>=1.18
"""

from typing import Optional, Any, List, Dict

from google import genai
from google.genai import types
from google.genai.errors import APIError

from chatbot.core.model_clients.llm.base_class import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from chatbot.core.model_clients.llm.base_llm import BaseLLM
from chatbot.core.model_clients.llm.google.config import GoogleAIClientLLMConfig, SchemaLike
from chatbot.core.model_clients.llm.exceptions import CallServerLLMError


class GoogleAIClientLLM(BaseLLM):
    """
    An LLM client that implements the BaseLLM interface for Google's Generative AI (Gemini) models.

    This class provides a standardized, simplified, and robust way to perform
    synchronous and asynchronous completions, including streaming.
    
    Features:
    - Sync/async completion with streaming support
    - Google Search grounding for up-to-date information
    - Thinking budgets for enhanced reasoning
    - System instructions for persistent context
    - Enterprise-grade error handling and client management

    Attributes:
        config (GoogleAIClientLLMConfig): The configuration object for the client.
        _client (genai.Client): The shared Google Generative AI client instance.
        _tools (List[types.Tool]): List of tools (e.g., Google Search) available to the model.
    """

    def __init__(self, config: GoogleAIClientLLMConfig, **kwargs):
        self.config: GoogleAIClientLLMConfig
        super().__init__(config, **kwargs)

    def _initialize_llm(self, **kwargs) -> None:
        """
        Initializes the Google Generative AI client and configures tools.
        
        This method is called once during instantiation to set up the client
        for reuse, which is more efficient than creating clients on each call.
        It also configures optional tools like Google Search grounding.
        
        Raises:
            CallServerLLMError: If client initialization fails.
        """
        try:
            # Create one shared client for both sync and async operations
            self._client = genai.Client(api_key=self.config.api_key)
        except Exception as e:
            raise CallServerLLMError(f"Failed to initialize Google Gen-AI client: {e}") from e

        # Configure optional Google Search tool for grounding
        self._tools: List[types.Tool] = []
        if self.config.use_grounding:
            self._tools.append(types.Tool(google_search=types.GoogleSearch()))

        if self.config.tools:
            # If tools are provided, ensure they are compatible with the Google Gen-AI client
            if isinstance(self.config.tools, list):
                self._tools.extend(self.config.tools)
            else:
                raise CallServerLLMError("Tools must be a list of functions.")

    def _make_generate_config(
        self,
        *,
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[SchemaLike] = None
    ) -> types.GenerateContentConfig:
        """
        Build GenerateContentConfig with all necessary parameters.
        
        This method centralizes configuration creation, similar to how
        OpenAI client prepares messages. It handles default values and
        optional features like thinking budgets.
        
        Args:
            system_instruction (str, optional): Override default system instruction.
            thinking_budget (int, optional): Override default thinking budget.
            temperature (float, optional): Override default temperature.
            top_p (float, optional): Override default top_p.
            max_tokens (int, optional): Override default max tokens.
            response_mime_type (str, optional): MIME type for the response.
            response_schema (SchemaLike, optional): Schema for response validation.
            
        Returns:
            types.GenerateContentConfig: Complete configuration for the API call.
        """
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "system_instruction": system_instruction or self.config.system_instruction,
            "tools": self._tools or None,
            "response_mime_type": response_mime_type or self.config.response_mime_type,
            "response_schema": response_schema or self.config.response_schema,
            "http_options": types.HttpOptions(timeout=60_000)
        }

        # Configure thinking budget if specified (0 disables thinking on Flash models)
        thinking_budget_to_use = (
            thinking_budget 
            if thinking_budget is not None 
            else self.config.thinking_budget
        )
        if thinking_budget_to_use is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget_to_use
            )

        return types.GenerateContentConfig(**config_kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generates a single, non-streaming completion response synchronously.
        
        Args:
            prompt (str): The user input prompt to process.
            **kwargs: Additional parameters to override config defaults.
            
        Returns:
            CompletionResponse: The model's response with generated text.
            
        Raises:
            CallServerLLMError: If the API call fails.
        """
        config = self._make_generate_config(**kwargs)
        try:
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise CallServerLLMError("Google Gen-AI API returned empty response text.")

            return CompletionResponse(text=response.text)
        except Exception as e:
            raise CallServerLLMError(f"Google Gen-AI API call failed: {e}")

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Generates a streaming completion response synchronously.
        
        This method yields incremental responses as they become available,
        allowing for real-time display of the model's output.
        
        Args:
            prompt (str): The user input prompt to process.
            **kwargs: Additional parameters to override config defaults.
            
        Yields:
            CompletionResponse: Incremental responses with full text and delta.
            
        Raises:
            CallServerLLMError: If the streaming call fails.
        """
        config = self._make_generate_config(**kwargs)
        try:
            stream = self._client.models.generate_content_stream(
                model=self.config.model,
                contents=prompt,
                config=config,
            )
            
            full_text = ""
            for chunk in stream:
                # Only process chunks that contain actual text content
                if delta := chunk.text:
                    full_text += delta
                    yield CompletionResponse(text=full_text, delta=delta)
        except APIError as e:
            raise CallServerLLMError(f"Google Gen-AI stream call failed: {e.message}") from e

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generates a single, non-streaming completion response asynchronously.
        
        Args:
            prompt (str): The user input prompt to process.
            **kwargs: Additional parameters to override config defaults.
            
        Returns:
            CompletionResponse: The model's response with generated text.
            
        Raises:
            CallServerLLMError: If the async API call fails.
        """
        config = self._make_generate_config(**kwargs)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise CallServerLLMError("Async Google Gen-AI API returned empty response text.")

            return CompletionResponse(text=response.text)
        except APIError as e:
            raise CallServerLLMError(f"Async Google Gen-AI API call failed: {e.message}") from e

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        """
        Generates a streaming completion response asynchronously.
        
        This method provides async streaming capabilities, yielding incremental
        responses as they become available in an async context.
        
        Args:
            prompt (str): The user input prompt to process.
            **kwargs: Additional parameters to override config defaults.
            
        Yields:
            CompletionResponse: Incremental responses with full text and delta.
            
        Raises:
            CallServerLLMError: If the async streaming call fails.
        """
        config = self._make_generate_config(**kwargs)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=prompt,
                config=config,
            )
            
            full_text = ""
            async for chunk in stream:
                # Only process chunks that contain actual text content
                if delta := chunk.text:
                    full_text += delta
                    yield CompletionResponse(text=full_text, delta=delta)
        except APIError as e:
            raise CallServerLLMError(f"Async Google Gen-AI stream call failed: {e.message}") from e
        