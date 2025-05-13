import asyncio
from typing import Optional

from contextlib import AsyncExitStack
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.genai import types

from .prompt import DOCUMENT_SEARCH_INSTRUCTION_PROMPT


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

async def get_search_tools() -> tuple[MCPToolset, AsyncExitStack]:
    """
    Get the search tools from the MCP server.

    Returns:
        tuple (MCPToolset, AsyncExitStack): A tuple containing the MCPToolset instance and an AsyncExitStack instance.
    """
    common_exit_stack = AsyncExitStack()

    search_tools, _ = await MCPToolset.from_server(
        connection_params=SseServerParams(
            url="http://localhost:8001/sse"
        ),
        async_exit_stack=common_exit_stack
    )

    return search_tools, common_exit_stack

async def create_agent() -> tuple[LlmAgent, AsyncExitStack]:
    """
    Create the Document search agent.

    Returns:
        tuple (tuple[LlmAgent, AsyncExitStack]:): A tuple containing the LlmAgent instance and an AsyncExitStack instance.
    """
    search_tools, common_exit_stack = await get_search_tools()

    agent = LlmAgent(
        name="document_search_agent",
        model="gemini-2.0-flash",
        description="Document search agent to find relevant documents about HCMUT.",
        instruction=DOCUMENT_SEARCH_INSTRUCTION_PROMPT,
        tools=search_tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
        before_model_callback=setup_before_model_call,
        after_agent_callback=process_after_agent_call,
        output_key="searched_documents",
    )
    return agent, common_exit_stack

root_agent = create_agent()
