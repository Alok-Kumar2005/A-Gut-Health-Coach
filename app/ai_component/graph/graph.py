import os
import sys
import asyncio

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from app.ai_component.graph.state import AICompanionState
from app.ai_component.graph.nodes import RouteNode, GutHealthNode, GeneralHealthNode, OffTopicNode
from app.ai_component.graph.edges import select_workflow
from typing import Optional
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv
load_dotenv()

os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME")

@lru_cache(maxsize=1)
def workflow_graph():
    graph_builder = StateGraph(AICompanionState)

    ## Adding node
    graph_builder.add_node("RouteNode", RouteNode)
    graph_builder.add_node("GutHealthNode", GutHealthNode)
    graph_builder.add_node("GeneralHealthNode", GeneralHealthNode)
    graph_builder.add_node("OffTopicNode", OffTopicNode)


    ## adding edgess
    graph_builder.add_edge(START, "RouteNode")
    graph_builder.add_conditional_edges("RouteNode",
                                        select_workflow,{"GutHealthNode": "GutHealthNode",
                                                         "GeneralHealthNode": "GeneralHealthNode",
                                                         "OffTopicNode": "OffTopicNode"})
    graph_builder.add_edge("GutHealthNode", END)
    graph_builder.add_edge("GeneralHealthNode", END)
    graph_builder.add_edge("OffTopicNode", END)

    return graph_builder.compile()

graph = workflow_graph()
tracer = OpikTracer(graph=graph.get_graph(xray=True))

# try:
#     img_data = graph.get_graph().draw_mermaid_png()
#     with open("workflow.png", "wb") as f:
#         f.write(img_data)
#     print("Graph saved as workflow.png")
# except Exception as e:
#     print(f"Error: {e}")


async def process_query_async(query: str, route: str = "GeneralHealthNode"):
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "workflow": route
    }
    
    result = await graph.ainvoke(initial_state)
    return result

if __name__ == "__main__":
    async def test_async_execution():
        query = "What is gut microbiome?"
        result = await process_query_async(query)
        print(result)
        
    asyncio.run(test_async_execution())