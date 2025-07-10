from typing import Optional, Any, Callable, Dict, Union, Type
from pydantic import BaseModel
from chatbot.core.model_clients.llm.base_class import LLMBackend, LLMConfig

JSONSchema = Dict[str, Any]
SchemaLike = Union[Type[BaseModel], JSONSchema, type, list, Dict[str, Any]]


class GoogleAIClientLLMConfig(LLMConfig):
    """
    Configuration for the Google AI (Gemini) client.

    This class centralizes settings for connecting to Google's Generative AI API
    with advanced features like grounding, thinking budgets, and system instructions.

    Important: Grounding and structured output are mutually exclusive. You must choose one:
    - Option A: Enable grounding for up-to-date web information (plain text responses only)
    - Option B: Use structured JSON/enum responses without grounding

    Attributes:
        model (str): 
            The ID of the Gemini model to use for completions 
            (e.g., "gemini-2.5-flash-preview", "gemini-2.5-pro-preview").
            Defaults to "gemini-2.5-flash-preview".
        api_key (str, optional): 
            The Google AI API key for authentication. It's recommended to set this via
            the `GOOGLE_API_KEY` environment variable. Defaults to None.
        temperature (float): 
            Controls randomness in the output. Lower values make the model more deterministic.
            Range: 0.0 to 2.0. Defaults to 0.1.
        top_p (float): 
            Controls diversity via nucleus sampling. Lower values restrict the model to
            more probable tokens. Range: 0.0 to 1.0. Defaults to 0.95.
        max_tokens (int): 
            The maximum number of tokens to generate in the completion. 
            Defaults to 8192.
        use_grounding (bool): 
            If True, enables Google Search grounding to base responses on
            up-to-date, verifiable information from the web. Defaults to False.
            
            NOTE: When grounding is enabled, you cannot use structured output
            (`response_mime_type` must be `"text/plain"` or None, and `response_schema`
            must be None). This is a Google API limitation.
        tools (Callable[..., Any], optional):
            Set of tools for the model can call to interact with external systems to
            perform an action outside of the knowledge and scope of the model.

            NOTE: When tools are provided, you cannot use structured output
            (`response_mime_type` must be `"text/plain"` or None, and `response_schema`
            must be None). This is a Google API limitation.
        system_instruction (str, optional): 
            A default system-level instruction for the model, applied to all requests.
            This serves as persistent context for the conversation. Defaults to None.
        thinking_budget (int, optional): 
            Controls the model's reasoning process. Set to 0 to disable thinking
            (Flash models only). Higher values allow more thorough reasoning.
            Defaults to None (uses model default).
        response_mime_type (str, optional):
            The MIME type for the response, such as `"application/json"` or `"text/plain"` or `"text/x.enum"`.
            This specifies the format of the model's output. Defaults to None.
            
            IMPORTANT: Cannot be used with grounding (use_grounding=True) or tools. 
            If grounding or tools are enabled, this must be None or `"text/plain"`.
        response_schema (SchemaLike, optional):
            A Pydantic model or JSON schema to validate the response structure.
            This ensures the model's output adheres to a specific format.
            Defaults to None (no validation).
            
            IMPORTANT: Cannot be used with grounding (use_grounding=True) or tools.
            If grounding or tools are enabled, this must be None.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        use_grounding: bool = False,
        tools: Optional[Callable[..., Any]] = None,
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[SchemaLike] = None,
        **kwargs
    ):
        # We explicitly set the backend to GOOGLE for this config type.
        super().__init__(backend=LLMBackend.GOOGLE, max_tokens=max_tokens, **kwargs)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.use_grounding = use_grounding
        self.system_instruction = system_instruction
        self.thinking_budget = thinking_budget
        self.response_mime_type = response_mime_type
        self.response_schema = response_schema
        self.tools = tools
        
        # Validate mutually exclusive options
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validates that grounding and structured output are not used together.
        
        Raises:
            ValueError: If both grounding and structured output are enabled.
        """
        if self.use_grounding or self.tools:
            # Check if structured output is being used
            has_structured_mime_type = (
                self.response_mime_type is not None and 
                self.response_mime_type != "text/plain"
            )
            has_response_schema = self.response_schema is not None
            
            if has_structured_mime_type or has_response_schema:
                raise ValueError(
                    "Cannot use grounding or tools with structured output. Choose one:\n"
                    "- Option A: use_grounding=True or provide tools with response_mime_type=None/text/plain and response_schema=None\n"
                    "- Option B: use_grounding=False and tools=None with response_mime_type/response_schema for structured output"
                )
        