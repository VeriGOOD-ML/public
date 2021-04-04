from .node import Node
from .graph import Graph

class Sorter(object):
    """
    Base class for sorting
    """

    def __init__(self):
        pass

    def run(self, graph:Graph, debug=False):
        raise NotImplementedError
