from binarytree import tree, bst, heap, Node


class DTNode(Node):
    def __init__ (self, value, left=None, right=None, operation=None,threshold=None, feature=None):
        Node.__init__(self, value, left=None, right=None)
        self.operation = operation
        self.threshold = threshold
        self.feature = feature



