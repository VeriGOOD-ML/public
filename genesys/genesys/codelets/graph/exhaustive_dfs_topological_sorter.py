import copy

from .node import Node
from .graph import Graph
from .topological_sorter import TopologicalSorter

def get_in_degree(node):
    return node.get_attr('__ts_in_degree')

def get_in_degree_from_dict(in_degree_dict, node):
    return in_degree_dict[node.index]

def set_in_degree_from_dict(in_degree_dict, node, value):
    in_degree_dict[node.index] = value

def decrement_in_degree_from_dict(in_degree_dict, node):
    in_degree = get_in_degree_from_dict(in_degree_dict, node)
    set_in_degree_from_dict(in_degree_dict, node, in_degree-1)

def increment_in_degree_from_dict(in_degree_dict, node):
    in_degree = get_in_degree_from_dict(in_degree_dict, node)
    set_in_degree_from_dict(in_degree_dict, node, in_degree+1)


class ExhaustiveDFSTopologicalSorter(TopologicalSorter):
    """
    Derived class for exhaustive DFS topological sorter
    """

    def __init__(self):
        super().__init__()

    def _run(self, graph:Graph, debug=False):

        # create output_components list
        outs = []

        # create initial dictionary for in_degree
        in_degree_dict = {}
        for node in graph.get_nodes():
            in_degree_dict[node.index] = get_in_degree(node)

        # create stack
        # NOTE stack means this is DFS
        stack = []
        for node in graph.get_nodes():
            if get_in_degree(node) == 0:
                # append a pair of (curr_out, curr_in_degree_dict)
                curr_out = [node]
                curr_in_degree_dict = copy.deepcopy(in_degree_dict)
                stack.append((curr_out, curr_in_degree_dict))

        # while stack is not empty
        while stack:
            if debug:
                print([[node.index for node in curr_pair[0]] for curr_pair in stack])
            
            # if inside stack
            curr_pair = stack.pop()
            curr_out = curr_pair[0]
            curr_in_degree_dict = curr_pair[1]
            
            # if all nodes have been visited, add to outs
            if len(curr_out) == graph.get_number_of_nodes():
                outs.append(curr_out)
                continue
            
            # decrement in_degree of succs
            for succ in curr_out[-1].get_succs():
                decrement_in_degree_from_dict(curr_in_degree_dict, succ)

            # if in_degree is zero and not in curr_out, append to stack
            for node in graph.get_nodes():
                if get_in_degree_from_dict(curr_in_degree_dict, node) == 0 and node not in curr_out:

                    # append a next pair of (next_out, next_in_degree_dict)
                    next_out = copy.copy(curr_out + [node]) # shallow copy
                    next_in_degree_dict = copy.deepcopy(curr_in_degree_dict) # deep copy
                    stack.append((next_out, next_in_degree_dict))
                    
        return outs
