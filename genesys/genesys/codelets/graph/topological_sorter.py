from .node import Node
from .graph import Graph
from .sorter import Sorter
from .cycle_analyzer import CycleAnalyzer

class TopologicalSorter(Sorter):
    """
    Base class for topological sorter
    """

    def __init__(self):
        super().__init__()

    def __preprocess(self, graph:Graph):

        # no cycle is imposed for topological sorting
        analyzer = CycleAnalyzer()
        out = analyzer.run(graph)
        assert out == False, 'no cycle is imposed for topological sorting'

        graph.duplicate_attr_of_nodes_by_key('in_degree', '__ts_in_degree')
        graph.duplicate_attr_of_nodes_by_key('out_degree', '__ts_out_degree')

    def _custom_preprocess(self, graph:Graph):
        pass
        
    def __postprocess(self, graph:Graph):
        graph.clear_attr_from_nodes_by_key('__ts_in_degree')
        graph.clear_attr_from_nodes_by_key('__ts_out_degree')

    def _custom_postprocess(self, graph:Graph):
        pass
        
    def run(self, graph:Graph, debug=False):
        self.__preprocess(graph)
        self._custom_preprocess(graph)
        out = self._run(graph, debug)
        self._custom_postprocess(graph)
        self.__postprocess(graph)

        return out

    # NOTE private functions cannot be overridden, so use protected here
    def _run(self, graph:Graph, debug=False):
        raise NotImplementedError
