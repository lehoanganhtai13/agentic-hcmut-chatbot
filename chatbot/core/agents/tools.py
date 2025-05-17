from typing import Any, Dict, List

from contextlib import AsyncExitStack
from google.adk.agents import ParallelAgent
from google.adk.tools import agent_tool, ToolContext

from .sub_agents.faq_search_agent.agent import create_agent as create_faq_search_agent
from .sub_agents.document_search_agent.agent import create_agent as create_document_search_agent


async def create_search_agent(
    query: str,
    query_index: int,
    exit_stack: AsyncExitStack
) -> ParallelAgent:
    """
    Create a agent for searching information.
    This agent is used to search parallely for both relevant FAQs and documents.

    Args:
        query (str): The query string to search for.
        query_index (int): The index of the query in the list of queries.
        exit_stack (AsyncExitStack): The exit stack for managing resources.
            If not provided, a new AsyncExitStack will be created.

    Returns:
        ParallelAgent: The created agent for searching information.
    """
    faq_search_agent, _ = await create_faq_search_agent(
        query_index=query_index,
        query=query,
        exit_stack=exit_stack
    )
    document_search_agent, _ = await create_document_search_agent(
        query_index=query_index,
        query=query,
        exit_stack=exit_stack
    )

    agent = ParallelAgent(
        name=f"search_agent_{query_index}",
        description="Search for relevant FAQs and documents.",
        sub_agents=[
            faq_search_agent,
            document_search_agent
        ]
    )

    return agent

async def search_information(queries: List[str], tool_context: ToolContext) -> str:
    """
    Search for relevant information based on the provided queries.
    This function creates a parallel agent for each query, which in turn creates
    sub-agents for searching FAQs and documents.

    Args:
        queries (List[str]): A list of queries to search for.
        tool_context (ToolContext): The context in which the tool is being used.

    Returns:
        str: The result of the search.
    """
    all_query_results: List[Dict[str, Any]] = []
    async with AsyncExitStack() as common_exit_stack:
        per_query_handler_agents: List[ParallelAgent] = []

        for i, query_text in enumerate(queries):
            handler_agent = await create_search_agent(
                query=query_text,
                query_index=i,
                exit_stack=common_exit_stack
            )
            per_query_handler_agents.append(handler_agent)
        
        if not per_query_handler_agents:
            # Handle the case where no handler agents were created
            return "Failed to find relevant information."

        # Create a master agent to handle all the per-query handler agents
        master_search_agent = ParallelAgent(
            name="master_batch_query_processor",
            description="Processes multiple queries in parallel, each by a dedicated handler agent.",
            sub_agents=per_query_handler_agents
        )
        search_agent_tool = agent_tool.AgentTool(agent=master_search_agent)

        result = await search_agent_tool.run_async(
            args={
                "request": "Search the information based on the provided query.",
            },
            tool_context=tool_context
        )

        for i, query_text in enumerate(queries):
            all_query_results.append({
                "query": query_text,
                "results": {
                    "faq": tool_context.state[f"searched_faqs_{i}"],
                    "document": tool_context.state[f"searched_documents_{i}"]
                }
            })

        tool_context.state["searched_results"] = all_query_results
    
    return all_query_results

# root_agent = create_search_agent()
