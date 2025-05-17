from typing import Optional
from datetime import datetime

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from .prompt import MANAGER_AGENT_INSTRUCTION_PROMPT
from .tools import search_information


def setup_before_model_call(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # Update timestamp in the callback context
    callback_context.state["_timestamp"] = datetime.now().isoformat()

    if "_tries" not in callback_context.state:
        # Initialize the step counter if not present
        callback_context.state["_tries"] = 0

    step = callback_context.state["_tries"]
    if step >= 6:
        # Reset the step counter
        callback_context.state["_tries"] = 0
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
    step = callback_context.state.get("_tries", 0)
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            # print("[Callback] Original text:", original_text)
        if llm_response.content.parts[0].function_call:
            callback_context.state["_tries"] = step + 1
            # print("[Callback] Function call detected")
    return None

def process_after_agent_call(callback_context: CallbackContext) -> Optional[types.Content]:
    # Reset the step counter after the model call
    callback_context.state["_tries"] = 0
    
    # print("[Callback] After agent call")

    return None

def create_agent() -> LlmAgent:
    """
    Create the Manager LLM agent for orchestrating information retrieval about HCMUT.
    This agent breaks down requests, uses the search_information tool, and synthesizes results.

    Returns:
        LlmAgent: The created Manager LLM agent.
    """
    agent = LlmAgent(
        model="gemini-2.0-flash",
        name="hcmut_information_manager",
        instruction=MANAGER_AGENT_INSTRUCTION_PROMPT,
        tools=[search_information],
        before_model_callback=setup_before_model_call,
        after_model_callback=after_model_call,
        after_agent_callback=process_after_agent_call
    )
    return agent

root_agent = create_agent()
