from typing import Optional

from datetime import datetime
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from .prompt import FAQ_SEARCH_INSTRUCTION_PROMPT


MAX_RETRIES = 3
MAX_FUNCTION_CALLS = 3


# ------------- Callbacks for the Document Search Agent -------------

def setup_before_model_call(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # Update timestamp in the callback context
    callback_context.state["_timestamp"] = datetime.now().isoformat()

    if "_tries" not in callback_context.state:
        # Initialize the step counter if not present
        callback_context.state["_tries"] = 0

    step = callback_context.state.get("_tries", 0)
    if step >= MAX_RETRIES + 2:  # 1 for result from the final function call and 1 for the first exceeded step
        # Reset the step counter
        callback_context.state["_tries"] = 0
        # Skip further model calls – return failure string
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Không tìm thấy tài liệu nào liên quan đến yêu cầu của bạn.")]
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
            pass # Do nothing with the text part

            # original_text = llm_response.content.parts[0].text
            # print("[Callback] Original text:", original_text)
        if llm_response.content.parts[0].function_call:
            callback_context.state["_tries"] = step + 1
            # print("[Callback] Function call detected")

    return None

def process_after_agent_call(callback_context: CallbackContext) -> Optional[types.Content]:
    # Reset the step counter after the model call
    callback_context.state["_tries"] = 0
    return None


# ------------- Document Search Agent Creation -------------

def create_agent(query_index: int, query: Optional[str] = None) -> LlmAgent:
    """
    Create the FAQ search agent.

    Args:
        query (Optional[str]): An optional user query to refine the search.
            If provided, the agent will focus on finding FAQs relevant to this query.
            If not provided, the agent will perform a general search.
        query_index (int): An index to specify the query's position in a list of queries.

    Returns:
        LlmAgent: The created FAQ search agent.
    """
    instruction = FAQ_SEARCH_INSTRUCTION_PROMPT
    if query:
        instruction += f"""
        ## Specific Query Mandate
        **IMPORTANT**: For this specific execution, your sole focus is to find FAQs relevant to the following user query: 
        ```
        {query}
        ```

        All your search actions and query reformulations must be aimed at answering this mandated query.
        If you receive generic instructions from a parent agent, prioritize this specific query mandate.
        """

    agent = LlmAgent(
        name="faq_search_agent",
        model="gemini-2.5-flash",
        description="FAQ search agent to find relevant FAQs about HCMUT.",
        instruction=instruction.format(max_retries=MAX_RETRIES),
        tools=[MCPToolset(connection_params=SseConnectionParams(url="http://faq-server:8000/sse"))],
        before_model_callback=setup_before_model_call,
        after_agent_callback=process_after_agent_call,
        output_key=f"searched_faqs_{query_index}",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_FUNCTION_CALLS
            )
        ),
        planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=False))
    )
    return agent

# root_agent = create_agent()
