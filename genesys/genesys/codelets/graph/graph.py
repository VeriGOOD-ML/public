import os
import graphviz
from graphviz import Digraph
import copy
from .node import Node

class Graph(object):
    """
    Base class for graph
    """

    def __init__(self, parent_graph=None):
        self._nodes = {}
        # attributes
        self._attrs = {}
        self._parent_graph = parent_graph
        self._inputs = []
        self._outputs = []

    def _add_node(self, node: Node):
        node.set_graph(self)
        self._nodes[node.index] = node

    def add_input(self, src):
        self._inputs.append(src.index)

    def add_output(self, dst):
        self._outputs.append(dst.index)

    def get_node_by_index(self, node_index):
        return self._nodes[node_index]

    def get_nodes(self):
        return list(self._nodes.values())
    
    def get_number_of_nodes(self):
        return len(self._nodes.keys())

    def get_node_indices(self):
        return self._nodes.keys()

    def _add_edge(self, src_node, dst_node):

        # check if src and dest are both in the graph
        assert src_node.index in self._nodes.keys() or src_node.index in self._inputs, 'src_node is not in graph'
        assert dst_node.index in self._nodes.keys() or dst_node.index in self._outputs, 'dst_node is not in graph'

        # NOTE add_succ and add_pred deals with both src and dest nodes.
        # therefore, only one invocation is required
        src_node.add_succ(dst_node)
        # dst_node.add_pred(src_node)
    
    def remove_edge(self, src_node, dst_node):
        
        # check if src and dest are both in the graph
        assert src_node.index in self._nodes.keys(), 'src_node is not in graph'
        assert dst_node.index in self._nodes.keys(), 'dst_node is not in graph'

        # NOTE remove_succ and remove_pred deals with both src and dest nodes.
        # therefore, only one invocation is required
        src_node.remove_succ(dst_node)
        #dst_node.remove_pred(src_node)
    
    def add_edge_by_index(self, src_node_index, dst_node_index):
        src_node = self.get_node_by_index(src_node_index)
        dst_node = self.get_node_by_index(dst_node_index)
        self._add_edge(src_node, dst_node)


    def delete_node(self, node:Node):
        node.delete()

    def delete_node_by_index(self, node_index:int):
        self._nodes[node_index].delete()

    def dissolve_node(self, node:Node):
        node.dissolve()

    def dissolve_node_by_index(self, node_index:int):
        self._nodes[node_index].dissolve()

    def remove_node(self, node:Node):
        del self._nodes[node.index]

    def remove_node_by_index(self, node_index:int):
        del self._nodes[node_index]


    # attributes
    def set_attr(self, attr_key:str, attr_value):
        self._attrs[attr_key] = attr_value

    def get_attr(self, attr_key:str):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        return self._attrs[attr_key]

    def append_attr(self, attr_key:str, attr_value):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        assert type(self._attrs[attr_key]) == list, 'attr_key is not list'
        self._attrs[attr_key].append(attr_value)

    def duplicate_attr_by_key(self, old_attr_key:str, new_attr_key:str):
        assert old_attr_key in self._attrs, 'old_attr_key is not in attrs'
        
        # TODO check whether this should be deep copy or maybe shallow copy?
        self._attrs[new_attr_key] = copy.deepcopy(self._attrs[old_attr_key])        

    def print_attrs(self):
        print('- Print Attributes')
        for attr_key in self._attrs:
            print(f' . {attr_key}')

    def clear_attrs(self):
        self._attrs = {}

    def clear_attr_by_key(self, attr_key:str):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        del self._attrs[attr_key]


    def duplicate_attr_of_nodes_by_key(self, old_attr_key:str, new_attr_key:str):
        for node in self.get_nodes():
            node.duplicate_attr_by_key(old_attr_key, new_attr_key)
    
    def clear_attrs_from_nodes(self):
        for node in self.get_nodes():
            node.clear_attrs()
    
    def clear_attr_from_nodes_by_key(self, attr_key:str):
        for node in self.get_nodes():
            node.clear_attr_by_key(attr_key)


    def visualize(self, filename='output'):
        
        digraph = Digraph(format='pdf')

        # draw nodes
        for node in self.get_nodes():
            self._draw_node(digraph, node)
        
        # draw edges
        for src_node in self.get_nodes():
            for dst_node in src_node.get_succs():
                self._draw_edge(digraph, src_node, dst_node)

        # save as pdf
        digraph.render(filename)
    
    def _draw_node(self, digraph, node:Node, attrs=None, shape='circle', style='filled', color='white'):
        
        name = f'node_{node.index}'
        label = f'{node.index}'
        
        digraph.node(name, label, shape=shape, style=style, fillcolor=color)

    def _draw_edge(self, digraph, src_node:Node, dst_node:Node):
        
        src_name = f'node_{src_node.index}'
        dst_name = f'node_{dst_node.index}'

        digraph.edge(src_name, dst_name)

