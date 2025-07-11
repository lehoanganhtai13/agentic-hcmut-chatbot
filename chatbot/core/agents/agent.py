import os
from typing import Optional
from datetime import datetime

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .prompt import MANAGER_AGENT_INSTRUCTION_PROMPT
from .tools import search_information

MAX_RETRIES = 4
MAX_FUNCTION_CALLS = 3

# Disable OpenTelemetry to avoid context management 
# issues with incompatible GCP exporter
os.environ["OTEL_SDK_DISABLED"] = "true"

# Suppress OpenTelemetry warnings
import logging
logging.getLogger("opentelemetry").setLevel(logging.ERROR)


def setup_before_model_call(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # Update timestamp in the callback context
    callback_context.state["_timestamp"] = datetime.now().isoformat()

    if "current_attempt" not in callback_context.state:
        # Initialize the step counter if not present
        callback_context.state["current_attempt"] = 0

    step = callback_context.state["current_attempt"]
    if step >= MAX_RETRIES + 2: # 1 for result from the final function call and 1 for the first exceeded step
        # Reset the step counter
        callback_context.state["current_attempt"] = 0
        # Skip further model calls – return failure string
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=(
                        "Rất tiếc, tôi không tìm thấy thông tin phù hợp. "
                        "Nếu bạn có thêm chi tiết, tôi sẵn lòng thử tìm lại giúp bạn."
                    )
                )]
            )
        )
    return None

def after_model_call(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> Optional[LlmResponse]:
    step = callback_context.state.get("current_attempt", 0)
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            pass # Do nothing with the text part

            # original_text = llm_response.content.parts[0].text
            # print("[Callback] Original text:", original_text)
        if llm_response.content.parts[0].function_call:
            callback_context.state["current_attempt"] = step + 1
            # print("[Callback] Function call detected")

    return None

def process_after_agent_call(callback_context: CallbackContext) -> Optional[types.Content]:
    # Reset the step counter after the model call
    callback_context.state["current_attempt"] = 0
    return None

def init_session_state(callback_context: CallbackContext) -> None:
    # Initialize the session state for the agent
    if "current_attempt" not in callback_context.state:
        # Initialize the step counter if not present
        # This is necessary to ensure the agent can track attempts correctly
        # across multiple calls
        callback_context.state["current_attempt"] = 0

def create_agent() -> LlmAgent:
    """
    Create the Manager LLM agent for orchestrating information retrieval about HCMUT.
    This agent breaks down requests, uses the search_information tool, and synthesizes results.

    Returns:
        LlmAgent: The created Manager LLM agent.
    """
    agent = LlmAgent(
        model="gemini-2.5-flash",
        name="hcmut_information_manager",
        instruction=MANAGER_AGENT_INSTRUCTION_PROMPT.format(
            max_retries=MAX_RETRIES,
            current_attempt="{current_attempt}"
        ),
        tools=[search_information],
        before_model_callback=setup_before_model_call,
        before_agent_callback=init_session_state,
        after_model_callback=after_model_call,
        after_agent_callback=process_after_agent_call,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_FUNCTION_CALLS
            )
        ),
        planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1000  # Set a reasonable thinking budget for the agent
        ))
    )
    return agent

root_agent = create_agent()
