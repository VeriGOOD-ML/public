from .node import Node
from .graph import Graph

class Analyzer(object):
    """
    Base class for analyzing
    """

    def __init__(self):
        pass

    def run(self, graph:Graph):
        raise NotImplementedError
