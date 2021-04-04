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

class DFSTopologicalSorter(TopologicalSorter):
    """
    Derived class for DFS topological sorter
    """

    def __init__(self):
        super().__init__()

    def _run(self, graph:Graph, debug=False):

        # create output list
        out = []

        # create stack
        # NOTE stack means this is DFS
        stack = []
        for node in graph.get_nodes():
            if get_in_degree(node) == 0:
                stack.append(node)
               
        # while stack is not empty
        while stack:
            if debug:
                print([node.index for node in stack])

            # if inside stack
            node = stack.pop()
            out.append(node)

            # decrement in_degree of succs
            for succ in node.get_succs():
                decrement_in_degree(succ)

                # if in_degree is zero, append to stack
                if get_in_degree(succ) == 0:
                    stack.append(succ)

        return out
