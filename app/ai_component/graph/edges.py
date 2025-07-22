from app.ai_component.graph.state import AICompanionState
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException
import sys


def select_workflow(state: AICompanionState) ->str:
    """
    Selects the workflow based on the current state.
    This function is used to determine which workflow to execute next.
    """
    try:
        logging.info("Selecting workflow")
        workflow = state["route"]
        if workflow == "GutHealthNode":
            return "GutHealthNode"
        elif workflow == "GeneralHealthNode":
            return "GeneralHealthNode"
        else:
            return "OffTopicNode"
        logging.info("selected workflow is: ".format(workflow))
    except CustomException as e:
        logging.error(f"Error in selecting workflow : {str(e)}")
        raise CustomException(e, sys) from e
    