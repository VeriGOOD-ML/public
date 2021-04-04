import copy

from .node import Node
from .graph import Graph
from .topological_sorter import TopologicalSorter

def get_in_degree(node):
    return node.get_attr('__ts_in_degree')

def set_in_degree(node, value):
    node.set_attr('__ts_in_degree', value)

def decrement_in_degree(node):
    in_degree = get_in_degree(node)
    set_in_degree(node, in_degree-1)

class BFSTopologicalSorter(TopologicalSorter):
    """
    Derived class for BFS topological sorter (Kahn's Algorithm)
    """

    def __init__(self):
        super().__init__()

    def _run(self, graph:Graph, debug=False):

        # create output list
        out = []

        # create queue
        # NOTE queue means this is BFS
        queue = []
        for node in graph.get_nodes():
            if get_in_degree(node) == 0:
                queue.append(node)
               
        # while queue is not empty
        while queue:
            if debug:
                print([node.index for node in queue])

            # if inside queue
            node = queue.pop(0)
            out.append(node)

            # decrement in_degree of succs
            for succ in node.get_succs():
                decrement_in_degree(succ)

                # if in_degree is zero, append to queue
                if get_in_degree(succ) == 0:
                    queue.append(succ)

        return out
