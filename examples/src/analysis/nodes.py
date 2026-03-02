from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from .models import AnalysisState
from .. import lightrag_retrieve, llm, Model

def create_analysis_node(name : str, model : type[Model]):
    def analysis_node(state : AnalysisState):
        agent = create_agent(
            model=llm,
            tools=[lightrag_retrieve],
            response_format=model,
        )
        response = agent.invoke({'messages': HumanMessage(f"{state.query}")})
        return {
            name: response['structured_response']
        }
    return analysis_node

__all__ = [
    'create_analysis_node'
]