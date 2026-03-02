from langgraph.graph import StateGraph, START, END
from .models import AnalysisState, SocioEconomicModel, SpatialModel, TransportModel, EngineeringModel
from .nodes import create_analysis_node

analysis_graph = StateGraph(AnalysisState)

analysis_nodes = {name: create_analysis_node(name,model) for name,model in {
    'socio_economic': SocioEconomicModel,
    'spatial': SpatialModel,
    'transport': TransportModel,
    'engineering': EngineeringModel
}.items()}

for name,node in analysis_nodes.items():
    analysis_graph.add_node(name, node)
    analysis_graph.add_edge(START, name)
    analysis_graph.add_edge(name, END)

analysis_app = analysis_graph.compile()

__all__ = [
    'analysis_app'
]