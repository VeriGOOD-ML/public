from codelets.graph import Graph
from graphviz import Digraph
import networkx as nx
from collections import defaultdict, namedtuple


class ArchitectureGraph(Graph):
    """
    Class for ArchitectureGraph
    """

    def __init__(self, old_name=None):
        super().__init__()
        self._edges = []
        # store the field_name of the graph from the .pb or .onnx
        self._old_name = old_name
        self._is_top_level = True

    def _networkx_visualize(self, filename='output'):
        dgraph = nx.MultiDiGraph(compound=True, concentrate=True)
        # draw nodes
        subgraphs = defaultdict(list)
        for node in self.get_nodes():
            attrs = None
            color = 'white' if not node.is_attr_key("node_color") else node.get_attr("node_color")
            self._add_nx_node(dgraph, subgraphs, node, attrs=attrs, color=color)

        # draw edges
        for src_node in self.get_nodes():
            for dst_node in src_node.get_succs():
                self._add_nx_edge(dgraph, src_node, dst_node)

        # save as pdf
        gviz = nx.nx_agraph.to_agraph(dgraph)
        for k, v in subgraphs.items():
            _ = gviz.subgraph(v, name=f"cluster_{k}", label=f"{k}")
        gviz.layout(prog='fdp')
        gviz.draw(f"{filename}.pdf", format="pdf")

    def unset_toplevel(self):
        self._is_top_level = False

    def _add_nx_subgraph(self, gviz_graph, node, attrs=None, shape='record', style='rounded,filled', color='white'):
        label = f'{type(node).__name__}' if not node.is_attr_key("field_name") else node.get_attr("field_name")

        if len(node.subgraph.get_nodes()) > 0:
            nids = []
            cluster_name = f"cluster_{node.index}"
            for n in node.subgraph.get_nodes():
                nid = self._add_nx_subgraph(gviz_graph, n, attrs, shape, style, color)
                nids.append(nid)
            _ = gviz_graph.subgraph(nids, name=cluster_name, label=label)
            return cluster_name
        else:
            name = f"node_{node.index}"
            node_attrs = {}
            node_attrs['style'] = style
            node_attrs['shape_symbols'] = shape
            node_attrs['fillcolor'] = 'white' if not node.is_attr_key("node_color") else node.get_attr("node_color")
            label = f"{label}\\n{node.get_viz_attr()}"
            if node.get_type() == "StorageNode":
                node_attrs['bitwidth'] = str(0.5)
                node_attrs['height'] = str(0.5)
                node_attrs['margin'] = str(0)
                node_attrs['fontsize'] = str(10)
            if attrs is not None:
                for key, value in attrs.items():
                    assert key not in node_attrs
                    node_attrs[key] = value
            _ = gviz_graph.add_node(name, label=label, **node_attrs)
            return name


    def _add_nx_node(self, nx_graph, subgraphs, node, attrs=None, shape='record', style='rounded,filled', color='white'):
        name = f'node_{node.index}'

        if node.is_attr_key("has_subgraph") and node.get_attr("has_subgraph"):
            return

        if node.is_attr_key("outer_graph"):
            parent_name = f"{node.get_attr('outer_graph')}"
            subgraphs[parent_name].append(name)

        node_attrs = {}
        node_attrs['label'] = f'{type(node).__name__}' if not node.is_attr_key("field_name") else node.get_attr("field_name")
        node_attrs['style'] = style
        node_attrs['shape_symbols'] = shape
        node_attrs['fillcolor'] = color

        if node.get_type() == "StorageNode":
            node_attrs['bitwidth'] = 0.5
            node_attrs['height'] = 0.5
            node_attrs['margin'] = 0
            node_attrs['fontsize'] = 10

        if attrs is not None:
            for key, value in attrs.items():
                assert key not in node_attrs
                node_attrs[key] = value

        nx_graph.add_node(name, **node_attrs)

    def _add_nx_edge(self, nx_graph, src_node, dst_node):
        src_name = f'node_{src_node.index}'
        dst_name = f'node_{dst_node.index}'
        nx_graph.add_edge(src_name, dst_name)

    def visualize(self, filename='output'):
        
        digraph = Digraph(format='pdf')

        # draw nodes
        for node in self.get_nodes():
            attrs = None
            color = 'white' if not node.is_attr_key("node_color") else node.get_attr("node_color")
            self._draw_node(digraph, node, attrs=attrs, color=color)
        
        # draw edges
        for src_node in self.get_nodes():
            for dst_node in src_node.get_succs():
                self._draw_edge(digraph, src_node, dst_node)

        # save as pdf
        digraph.render(filename)

    def _draw_node(self, digraph, node, attrs=None, shape='record', style='rounded,filled', color='white'):
        
        name = f'node_{node.index}'
        label = f'{type(node).__name__}' if not node.is_attr_key("field_name") else node.get_attr("field_name")
        if attrs is not None:
            label += '|'
            for key, value in attrs.items():
                label += f'{key}: {value}\l'
            label = '{{' + label + '}}'
        digraph.node(name, label, shape=shape, style=style, fillcolor=color)
    
    def _draw_edge(self, digraph, src_node, dst_node):
        
        src_name = f'node_{src_node.index}'
        dst_name = f'node_{dst_node.index}'

        digraph.edge(src_name, dst_name)

    def get_viz_edge_list(self):
        edges = []
        for n in self.get_nodes():
            n_name = self.get_viz_name(n)
            preds = n.get_preds()
            edges += [(self.get_viz_name(p), n_name) for p in preds]
        return edges

    def get_viz_name(self, node):
        if len(node.subgraph.get_nodes()) > 0:
            return f"cluster_{node.index}"
        else:
            return f"node_{node.index}"

    def add_node(self, node):
        self._add_node(node)

    def add_edge(self, src, dst, attributes=None):
        self._add_edge(src, dst)

