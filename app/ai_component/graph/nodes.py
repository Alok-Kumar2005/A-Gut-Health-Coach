from app.ai_component.graph.state import AICompanionState
from app.ai_component.llm import LLMChainFactory
from app.ai_component.graph.utils.chains import router_chain
from app.ai_component.modules.hybrid_retriever import memory
from app.ai_component.core.prompts import guthealthNode_template, generalHealthNode_template, offtopic_template
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException

import os
import sys
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

def format_conversation_history(messages: list, max_turns: int = 3) -> str:
    """Format recent conversation history for context"""
    if not messages or len(messages) <= 1:
        return "This is the start of our conversation."
    
    recent_messages = messages[-(max_turns * 2):-1] 
    
    history_parts = []
    for i in range(0, len(recent_messages), 2):
        if i + 1 < len(recent_messages):
            user_msg = recent_messages[i]
            ai_msg = recent_messages[i + 1]
            
            user_content = user_msg.content if hasattr(user_msg, 'content') else str(user_msg)
            ai_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
            
            history_parts.append(f"You asked: {user_content}")
            history_parts.append(f"I responded: {ai_content}")
    
    return "\n".join(history_parts) if history_parts else "This is the start of our conversation."

async def RouteNode(state: AICompanionState) -> dict:
    """Route user query to appropriate node"""
    try:
        logging.info("Calling Route Node")
        current_message = state["messages"][-1]
        
        if hasattr(current_message, 'content'):
            query_content = current_message.content
        else:
            query_content = str(current_message)
            
        workflow = "GeneralHealthNode"
        if query_content:
            chain = await router_chain()
            response = await chain.ainvoke({"query": query_content})
            workflow = response.route_node
            logging.info(f"Route Node selected: {workflow}")
            
        return {
            "route": workflow,
            "conversation_history": format_conversation_history(state["messages"])
        }
    except Exception as e:
        logging.error(f"Error in route_node: {e}")
        raise CustomException(e, sys) from e

async def GutHealthNode(state: AICompanionState) -> dict:
    """Handle gut health queries with August AI personality"""
    try:
        logging.info("GutHealthNode calling...")
        current_message = state["messages"][-1]
        
        if hasattr(current_message, 'content'):
            query_content = current_message.content
        else:
            query_content = str(current_message)
            
        logging.info(f"Processing gut health query: {query_content}")
        
        collection_name = "health_articles_collection"
        docs = memory.hybrid_search(query=query_content, collection_name=collection_name, k=5)
        
        context_text = ""
        if docs:
            context_text = "\n\n".join([doc.page_content for doc in docs])
        else:
            logging.warning("No documents retrieved for query")
            context_text = "I'll use my knowledge to help you with this gut health question."
        
        conversation_history = format_conversation_history(state["messages"])
        
        prompt = PromptTemplate(
            input_variables=["context", "query", "conversation_history"],
            template=guthealthNode_template.prompt
        )
        
        factory = LLMChainFactory(model_type="gemini")
        llm_chain = await factory.get_llm_chain_async(prompt=prompt)
        
        answer = await llm_chain.ainvoke({
            "context": context_text,
            "query": query_content,
            "conversation_history": conversation_history
        })
        
        if hasattr(answer, 'content'):
            final_answer = answer.content
        else:
            final_answer = str(answer)

        logging.info("GutHealthNode response generated with August AI tone")
        
        ai_message = AIMessage(content=final_answer)
        updated_messages = state["messages"] + [ai_message]
        
        return {
            "messages": updated_messages,
            "conversation_history": format_conversation_history(updated_messages)
        }

    except Exception as e:
        logging.error(f"Error in GutHealthNode: {e}")
        fallback_response = "I understand you're looking for help with your gut health. While I'm having trouble accessing my knowledge base right now, I'm here to support you. Could you tell me a bit more about what you're experiencing?"
        ai_message = AIMessage(content=fallback_response)
        return {"messages": state["messages"] + [ai_message]}

async def GeneralHealthNode(state: AICompanionState) -> dict:
    """Handle general health queries with August's warm tone"""
    try:
        logging.info("GeneralHealthNode calling...")
        current_message = state["messages"][-1]
        
        if hasattr(current_message, 'content'):
            query_content = current_message.content
        else:
            query_content = str(current_message)
        
        conversation_history = format_conversation_history(state["messages"])
        
        prompt = PromptTemplate(
            input_variables=["query", "conversation_history"],
            template=generalHealthNode_template.prompt
        )
        
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt=prompt)
        
        response = await chain.ainvoke({
            "query": query_content,
            "conversation_history": conversation_history
        })
        
        final_answer = response.content if hasattr(response, 'content') else str(response)
        
        ai_message = AIMessage(content=final_answer)
        updated_messages = state["messages"] + [ai_message]
        
        return {
            "messages": updated_messages,
            "conversation_history": format_conversation_history(updated_messages)
        }
        
    except Exception as e:
        logging.error(f"Error in GeneralHealthNode: {str(e)}")
        fallback_response = "I'd be happy to help with your health question! While my specialty is gut health, I can provide some general guidance. Could you share more details about what you're looking for?"
        ai_message = AIMessage(content=fallback_response)
        return {"messages": state["messages"] + [ai_message]}

async def OffTopicNode(state: AICompanionState) -> dict:
    """Handle off-topic queries with friendly redirection"""
    try:
        logging.info("Calling OffTopicNode")
        current_message = state["messages"][-1]
        
        if hasattr(current_message, 'content'):
            query_content = current_message.content
        else:
            query_content = str(current_message)
        
        prompt = PromptTemplate(
            input_variables=["query"],
            template=offtopic_template.prompt
        )
        
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt=prompt)
        
        response = await chain.ainvoke({"query": query_content})
        final_answer = response.content if hasattr(response, 'content') else str(response)
        
        ai_message = AIMessage(content=final_answer)
        updated_messages = state["messages"] + [ai_message]
        
        return {
            "messages": updated_messages,
            "conversation_history": format_conversation_history(updated_messages)
        }
        
    except Exception as e:
        logging.error(f"Error in OffTopicNode: {str(e)}")
        fallback_response = "I'd love to help, but my expertise is in gut health and digestive wellness. I'm here if you have any questions about your digestive health!"
        ai_message = AIMessage(content=fallback_response)
        return {"messages": state["messages"] + [ai_message]}