from .node import Node
from .graph import Graph
from .analyzer import Analyzer

def get_in_degree(node):
    return node.get_attr('__ca_in_degree')

def set_in_degree(node, value):
    node.set_attr('__ca_in_degree', value)

def decrement_in_degree(node):
    in_degree = get_in_degree(node)
    set_in_degree(node, in_degree-1)

class CycleAnalyzer(Analyzer):
    """
    Derived class for cycle analyzer
    """

    def __init__(self):
        super().__init__()
    
    def __preprocess(self, graph):
        graph.duplicate_attr_of_nodes_by_key('in_degree', '__ca_in_degree')
        graph.duplicate_attr_of_nodes_by_key('out_degree', '__ca_out_degree')
        
    def __postprocess(self, graph):
        graph.clear_attr_from_nodes_by_key('__ca_in_degree')
        graph.clear_attr_from_nodes_by_key('__ca_out_degree')

    def run(self, graph:Graph):
        self.__preprocess(graph)
        out = self._run(graph)
        self.__postprocess(graph)

        return out

    def _run(self, graph:Graph):

        # create queue and add vertices if in_degree is zero
        # NOTE queue means this is BFS
        queue = []
        for node in graph.get_nodes():
            if get_in_degree(node) == 0:
                queue.append(node)

        count = 0

        # while queue is not empty
        while queue:
            # if inside queue, pop
            node = queue.pop(0)

            # decrement in_degree of succs
            for succ in node.get_succs():
                decrement_in_degree(succ)

                # if in_degree is zero, append to queue
                if get_in_degree(succ) == 0:
                    queue.append(succ)

            count += 1

        # check if there was cycle
        if count == graph.get_number_of_nodes():
            return False
        else:
            return True
