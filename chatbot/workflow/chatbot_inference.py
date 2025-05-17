import asyncio
from contextlib import AsyncExitStack
from google.adk.sessions import InMemorySessionService
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.adk.runners import Runner


async def create_agent():
    """Gets tools from MCP Server."""
    common_exit_stack = AsyncExitStack()

    remote_tools, _ = await MCPToolset.from_server(
        connection_params=SseServerParams(
            url="http://localhost:8001/mcp/sse"
        ),
        async_exit_stack=common_exit_stack
    )


    agent = LlmAgent(
        model='gemini-2.0-flash',
        name='enterprise_assistant',
        instruction=(
            'Help user accessing their file systems'
        ),
        tools=[],
    )
    return agent, common_exit_stack
