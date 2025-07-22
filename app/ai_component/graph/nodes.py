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
        if isinstance(query, dict):
            query_content = query.get("content", str(query))
        elif hasattr(query, 'content'):
            query_content = query.content
        else:
            query_content = str(query)
            
        workflow = "GeneralHealthNode"
        if query_content:
            chain = await router_chain()
            response = await chain.ainvoke({"query": query_content})
            workflow = response.route_node
            logging.info(f"Route Node selected: {workflow}")
        return {
            "route": workflow
        }
    except CustomException as e:
        logging.error(f"Error in route_node: {e}")
        raise CustomException(e, sys) from e
    

async def GutHealthNode(state: AICompanionState) -> dict:
    """
    Retrieve the data from the hybrid search and answer via RetrievalQA
    """
    try:
        logging.info("GutHealthNode calling...")
        query = state["messages"][-1]
        if isinstance(query, dict):
            query_content = query.get("content", str(query))
        elif hasattr(query, 'content'):
            query_content = query.content
        else:
            query_content = str(query)
            
        logging.info(f"Processing query: {query_content}")
        collection_name = "health_articles_collection"
        docs = memory.hybrid_search(query=query_content, collection_name=collection_name, k=5)
        
        if not docs:
            logging.warning("No documents retrieved for query")
            return {"messages": "I couldn't find any relevant information."}
            
        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=guthealthNode_template.prompt
        )
        
        factory = LLMChainFactory(model_type="gemini")
        llm = factory._get_llm() 
        llm_chain = await factory.get_llm_chain_async(prompt=prompt)
        answer = await llm_chain.ainvoke({
            "context": context_text,
            "query": query_content
        })
        if hasattr(answer, 'content'):
            final_answer = answer.content
        else:
            final_answer = str(answer)

        logging.info("GutHealthNode response generated")
        return {"messages": final_answer}

    except Exception as e:
        logging.error(f"Error in GutHealthNode: {e}")
        raise CustomException(e, sys) from e