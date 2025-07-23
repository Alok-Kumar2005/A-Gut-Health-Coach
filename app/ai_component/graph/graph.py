import os
import sys
import asyncio
import uuid
from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from app.ai_component.graph.state import AICompanionState
from app.ai_component.graph.nodes import RouteNode, GutHealthNode, GeneralHealthNode, OffTopicNode
from app.ai_component.graph.edges import select_workflow
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv

load_dotenv()

os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME")

@lru_cache(maxsize=1)
def workflow_graph():
    """Create the workflow graph with memory saver"""
    graph_builder = StateGraph(AICompanionState)

    # Add nodes
    graph_builder.add_node("RouteNode", RouteNode)
    graph_builder.add_node("GutHealthNode", GutHealthNode)
    graph_builder.add_node("GeneralHealthNode", GeneralHealthNode)
    graph_builder.add_node("OffTopicNode", OffTopicNode)

    # Add edges
    graph_builder.add_edge(START, "RouteNode")
    graph_builder.add_conditional_edges(
        "RouteNode",
        select_workflow,
        {
            "GutHealthNode": "GutHealthNode",
            "GeneralHealthNode": "GeneralHealthNode",
            "OffTopicNode": "OffTopicNode"
        }
    )
    graph_builder.add_edge("GutHealthNode", END)
    graph_builder.add_edge("GeneralHealthNode", END)
    graph_builder.add_edge("OffTopicNode", END)

    memory_saver = MemorySaver()
    return graph_builder.compile(checkpointer=memory_saver)

graph = workflow_graph()
tracer = OpikTracer(graph=graph.get_graph(xray=True))

class GutHealthCoach:
    """Main interface for the Gut Health Coach"""
    
    def __init__(self):
        self.graph = graph
        
    async def process_message(self, message: str, session_id: str = None) -> str:
        """Process a single message with conversation memory"""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        human_message = HumanMessage(content=message)
        
        initial_state = {
            "messages": [human_message],
            "route": "",
            "conversation_history": "",
            "user_context": {},
            "session_id": session_id
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "I'm here to help with your gut health questions!"
                
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I'm having some technical difficulties, but I'm still here to help with your gut health questions. Could you try asking again?"
    
    async def start_conversation(self, session_id: str = None) -> str:
        """Start a new conversation"""
        welcome_message = """Hi there! I'm August, your gut health coach. 

I'm here to help you understand your digestive health in a warm, supportive way. Whether you're dealing with bloating, food sensitivities, or just want to optimize your gut health, I'm here to listen and guide you.

What's on your mind today? """
        
        if session_id:
            config = {"configurable": {"thread_id": session_id}}
            initial_state = {
                "messages": [AIMessage(content=welcome_message)],
                "route": "",
                "conversation_history": "This is the start of our conversation.",
                "user_context": {},
                "session_id": session_id
            }
            await self.graph.ainvoke(initial_state, config=config)
        
        return welcome_message

coach = GutHealthCoach()

async def process_query_async(query: str, session_id: str = None):
    """Legacy function for backward compatibility"""
    return await coach.process_message(query, session_id)

async def test_critical_questions():
    """Test the 10 critical gut health questions"""
    critical_questions = [
        "I've been bloated for three days — what should I do?",
        "How does gut health affect sleep?",
        "What are the best probiotics for lactose intolerance?",
        "What does mucus in stool indicate?",
        "I feel nauseous after eating fermented foods. Is that normal?",
        "Should I fast if my gut is inflamed?",
        "Can antibiotics damage gut flora permanently?",
        "How do I know if I have SIBO?",
        "What are signs that my gut is healing?",
        "Why do I feel brain fog after eating sugar?"
    ]
    
    print("Testing Critical Gut Health Questions with August AI Coach\n")
    
    session_id = str(uuid.uuid4())
    
    for i, question in enumerate(critical_questions, 1):
        print(f"{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}")
        
        response = await coach.process_message(question, session_id)
        print(f"August's Response:\n{response}\n")
        
        # Small delay to prevent overwhelming the system
        await asyncio.sleep(1)

if __name__ == "__main__":
    async def main():
        session_id = str(uuid.uuid4())
        
        welcome = await coach.start_conversation(session_id)
        print("August:", welcome)
        
        query = "I've been feeling bloated after every meal. What could be causing this?"
        response = await coach.process_message(query, session_id)
        print(f"\nUser: {query}")
        print(f"August: {response}")
        
        follow_up = "What foods should I avoid?"
        response2 = await coach.process_message(follow_up, session_id)
        print(f"\nUser: {follow_up}")
        print(f"August: {response2}")
        
        ### uncomment to run full critical question
        # await test_critical_questions()
        
    asyncio.run(main())