from app.ai_component.graph.state import AICompanionState
from app.ai_component.llm import LLMChainFactory
from app.ai_component.graph.utils.chains import router_chain
from app.ai_component.modules.hybrid_retriever import memory
from app.ai_component.core.prompts import guthealthNode_template
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException

import os
import sys
from langchain.chains import RetrievalQA
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
    

async def GutHealthNode(state: AICompanionState)->dict:
    """
    Retrieve the data from the hybrid search
    """
    try:
        logging.info("GutHealthNode calling.... ")
        prompt = PromptTemplate(
            input_variables=[],
            template= guthealthNode_template.prompt
        )
        factory = LLMChainFactory(model_type= "gemini")
        llm = await factory.get_llm_chain_async(prompt= prompt)
        results = memory.hybrid_search(query, collection_name, k=5)