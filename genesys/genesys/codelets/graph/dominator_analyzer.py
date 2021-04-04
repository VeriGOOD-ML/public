from functools import reduce

from .node import Node
from .graph import Graph
from .analyzer import Analyzer

def get_in_degree(node):
    return node.get_attr('__da_in_degree')

def set_in_degree(node, value):
    node.set_attr('__da_in_degree', value)

def decrement_in_degree(node):
    in_degree = get_in_degree(node)
    set_in_degree(node, in_degree-1)

def intersection(lists):

    return list(reduce(set.intersection, [set(l) for l in lists]))

class DominatorAnalyzer(Analyzer):
    """
    Derived class for dominator analyzer
    """

    def __init__(self):
        super().__init__()
    
    def __preprocess(self, graph):
        graph.duplicate_attr_of_nodes_by_key('in_degree', '__da_in_degree')
        graph.duplicate_attr_of_nodes_by_key('out_degree', '__da_out_degree')
        
    def __postprocess(self, graph):
        graph.clear_attr_from_nodes_by_key('__da_in_degree')
        graph.clear_attr_from_nodes_by_key('__da_out_degree')

    def run(self, graph:Graph):
        self.__preprocess(graph)
        out = self._run(graph)
        self.__postprocess(graph)

        return out

    def _run(self, graph:Graph):

        # find input node
        zero_in_degree_set = []
        for node in graph.get_nodes():
            if get_in_degree(node) == 0:
                zero_in_degree_set.append(node)

        # TODO extend to when there are multiple start nodes
        assert len(zero_in_degree_set) == 1, 'dominator analysis currently support only one start node graph'
        start_node = zero_in_degree_set[0]
        
        # set dom(start) = start
        dominators = {}
        dominators[start_node] = [start_node]

        # set dom(others) = all
        for node in graph.get_nodes():
            if node == start_node:
                continue
            dominators[node] = graph.get_nodes()

        # eliminate ones that are not dominators
        # flag represents change in dom
        flag = True

        while flag:
            flag = False

            # iterate all except input node
            for node in graph.get_nodes():
                if node == start_node:
                    continue

                # get preds
                pred_dominators = []
                for pred in node.get_preds():
                    pred_dominators.append(dominators[pred])
                pred_dominators = intersection(pred_dominators)
                new_dominators = [node] + pred_dominators

                # detect change
                if set(new_dominators) != set(dominators[node]):
                    flag = True

                # apply
                dominators[node] = new_dominators
        
        return dominators
