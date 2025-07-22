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
    

async def GutHealthNode(state: AICompanionState) -> dict:
    """
    Retrieve the data from the hybrid search and answer via RetrievalQA
    """
    try:
        logging.info("GutHealthNode calling...")
        query = state["messages"][-1]

        # Define or fetch your collection name
        collection_name = "health_articles_collection"
        docs = memory.hybrid_search(query=query, collection_name=collection_name, k=5)
        if not docs:
            logging.warning("No documents retrieved for query")
            return {"messages": "I couldn't find any relevant information."}
        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=guthealthNode_template.prompt
        )
        factory = LLMChainFactory(model_type="gemini")
        llm_chain = await factory.get_llm_chain_async(prompt=prompt)
        qa = RetrievalQA.from_chain_type(
            llm=llm_chain.llm,
            chain_type="stuff",
            retriever=None,
            return_source_documents=False
        )
        answer = await qa.arun({
            "context": context_text,
            "query": query
        })

        logging.info("GutHealthNode response generated")
        return {"messages": answer}

    except Exception as e:
        logging.error(f"Error in GutHealthNode: {e}")
        raise CustomException(e, sys) from e
