from typing import Optional

from contextlib import AsyncExitStack
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.genai import types

from .prompt import FAQ_SEARCH_INSTRUCTION_PROMPT


def setup_before_model_call(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    step = callback_context.state.get("_tries", 0)
    if step >= 6:
        # Reset the step counter
        callback_context.state["_tries"] = 0
        # Skip further model calls – return failure string
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Cannot find relevant FAQ.")]
            )
        )
    # Otherwise, continue with the model call
    # and increment the step counter
    callback_context.state["_tries"] = step + 1
    return None

def process_after_agent_call(callback_context: CallbackContext) -> Optional[types.Content]:
    # Reset the step counter after the model call
    callback_context.state["_tries"] = 0
    return None

async def get_search_tools(exit_stack: Optional[AsyncExitStack] = None) -> tuple[MCPToolset, AsyncExitStack]:
    """
    Get the search tools from the MCP server.

    Args:
        exit_stack (Optional[AsyncExitStack]): An optional AsyncExitStack instance for managing resources.
            If not provided, a new AsyncExitStack will be created.

    Returns:
        tuple (MCPToolset, AsyncExitStack): A tuple containing the MCPToolset instance and an AsyncExitStack instance.
    """

    common_exit_stack = AsyncExitStack() if exit_stack is None else exit_stack

    search_tools, _ = await MCPToolset.from_server(
        connection_params=SseServerParams(
            url="http://localhost:8002/sse"
        ),
        async_exit_stack=common_exit_stack
    )

    return search_tools, common_exit_stack

async def create_agent(
    query_index: int,
    query: Optional[str] = None,
    exit_stack: Optional[AsyncExitStack] = None
) -> tuple[LlmAgent, AsyncExitStack]:
    """
    Create the FAQ search agent.

    Args:
        exit_stack (Optional[AsyncExitStack]): An optional AsyncExitStack instance for managing resources.
            If not provided, a new AsyncExitStack will be created.
        query (Optional[str]): An optional user query to refine the search.
            If provided, the agent will focus on finding FAQs relevant to this query.
            If not provided, the agent will perform a general search.
        query_index (int): An index to specify the query's position in a list of queries.

    Returns:
        tuple (tuple[LlmAgent, AsyncExitStack]:): A tuple containing the LlmAgent instance and an AsyncExitStack instance.
    """
    search_tools, common_exit_stack = await get_search_tools(exit_stack)

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
        model="gemini-2.0-flash",
        description="FAQ search agent to find relevant FAQs about HCMUT.",
        instruction=instruction,
        tools=search_tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
        before_model_callback=setup_before_model_call,
        after_agent_callback=process_after_agent_call,
        output_key=f"searched_faqs_{query_index}"
    )
    return agent, common_exit_stack

# root_agent = create_agent()
