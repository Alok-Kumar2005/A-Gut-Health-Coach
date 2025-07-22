from app.ai_component.graph.state import AICompanionState
from app.ai_component.llm import LLMChainFactory
from app.ai_component.graph.utils.chains import router_chain
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException

import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate

async def RouteNode(state: AICompanionState)-> dict:
    """
    Return the best suited node as per query
    """
    try:
        logging.info("Calling Route Node")
        query = state["messages"][-1]
        # default route
        workflow = "GeneralNode"
        if query:
            chain = await router_chain()
            response = await chain.ainvoke({"query": query})
            workflow = response.route_node
            logging.info(f"Route Node selected: {workflow}")
        return {
            "route": workflow
        }
    except CustomException as e:
        logging.error(f"Error in route_node: {e}")
        raise CustomException(e, sys) from e