import os
import sys
import asyncio
from pydantic import BaseModel , Field
from typing import Optional, Literal , Union
from app.ai_component.llm import LLMChainFactory
from app.ai_component.core.prompts import router_template
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException



class Route(BaseModel):
    route_node: Literal["GutHealthNode", "GeneralHealthNode", "OffTopicNode"] = Field(..., description= "Choose as per user query and template given")


async def router_chain():
    """
    Return the node according to user query and the prompt
    """
    try:
        logging.info("Calling router chain")
        prompt = PromptTemplate(
            input_variables=["query"],
            template=router_template.prompt
        )
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_structured_llm_chain_async(prompt, Route)
        return chain
    except CustomException as e:
        logging.error(f"Error in calling router chain {str(e)}")
        raise CustomException(e, sys) from e
    