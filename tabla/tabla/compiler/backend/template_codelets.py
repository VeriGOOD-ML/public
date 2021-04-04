from collections import defaultdict
import numpy as np
import json
import math
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class VectorMul(object):

    def __init__(self, vmul_node, num_pes, num_pus, size, schedule="spatial"):
        assert vmul_node["operation"] == "vmul"
        assert len(vmul_node["parents"]) == 2
        assert all([isinstance(i, (list, np.ndarray)) for i in vmul_node["parents"]])
        self.vmul_node = vmul_node
        self.pes_per_pu = num_pes // num_pus
        self.schedule = schedule
        self.num_pus = num_pus
        self.num_pes = num_pes
        self.size = size
        self.total_ops = len(vmul_node["parents"][0])*2
        self.input_pairs = list(zip(*(vmul_node["parents"])))

        self.name_map = {i: f"a{i+1}" for i in vmul_node["parents"][0]}


        for i in vmul_node["parents"][1]:
            self.name_map[i] = f"b{i - len(vmul_node['parents'][0])}"

        self.mul_pairings = defaultdict(list)
        self.operations = deque()
        self.graph = nx.DiGraph()

        if self.schedule == "spatial":
            self.spatial_assignment()

    def __str__(self):
        pr_str = f"Size: {self.size}\n" \
            f"A nodes: {self.vmul_node['parents'][0]}\n" \
            f"B nodes: {self.vmul_node['parents'][1]}\n" \
            f"PE assignments:\n\t{json.dumps(dict(self.mul_pairings), indent=2)}\n"
        return pr_str





    def draw_graph(self):
        # pos = graphviz_layout(self.graph, prog='dot')
        # pos = nx.nx_pydot.pydot_layout(self.graph, prog='dot')
        labels = nx.get_node_attributes(self.graph, 'label')
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph,pos=pos, labels=labels, with_labels=True, node_size=100, font_size=5)
        plt.show()

    def spatial_assignment(self):
        scale = 2
        depth = np.ceil(np.log2(self.size))
        pe_counter = self.num_pes - 1

        while len(self.input_pairs) > 0:
            pair = self.input_pairs.pop(0)
            x1_pos = pair[0]
            x2_pos = pair[1]
            x3_pos = pair[1] - 0.5

            self.graph.add_node(pair[0], label=f"{self.name_map[pair[0]]}:\nPE{pe_counter}", pe=pe_counter, pos=(x1_pos, (depth + 1)/scale))
            self.graph.add_node(pair[1], label=f"{self.name_map[pair[1]]}:\nPE{pe_counter}", pe=pe_counter, pos=(x2_pos, (depth + 1)/scale))
            self.graph.add_node(self.total_ops, label=f"PE{pe_counter}", pe=pe_counter, pos=(x3_pos, depth/scale))
            self.graph.add_edge(pair[1], self.total_ops)
            self.graph.add_edge(pair[0], self.total_ops)
            self.operations.append(self.total_ops)
            self.total_ops += 1

            self.mul_pairings[pe_counter].append(pair)
            pe_counter -= 1
            if pe_counter == 0:
                pe_counter = self.num_pes - 1
            elif pe_counter % self.pes_per_pu == 0:
                pe_counter -= 1

        while len(self.operations) > 1:
            op1 = self.operations.popleft()
            op2 = self.operations.popleft()
            depth = np.ceil(np.log2(len(self.operations))) if len(self.operations) > 0 else -1/scale

            op1_node = self.graph.nodes[op1]
            op2_node = self.graph.nodes[op2]
            target_pe = max(op1_node['pe'], op2_node['pe'])
            x_pos = op1_node['pos'][0] + (op2_node['pos'][0] - op1_node['pos'][0])/2
            self.graph.add_node(self.total_ops, label=f"PE{target_pe}", pe=target_pe, pos=(x_pos, depth/scale))
            self.graph.add_edge(op1, self.total_ops)
            self.graph.add_edge(op2, self.total_ops)
            self.operations.append(self.total_ops)
            self.total_ops += 1


    def spatio_temporal_assignment(self):
        scale = 2
        depth = np.ceil(np.log2(self.size))
        pe_counter = self.num_pes - 1

        while len(self.input_pairs) > 0:
            pair = self.input_pairs.pop(0)
            x1_pos = pair[0]
            x2_pos = pair[1]
            x3_pos = pair[1] - 0.5

            self.graph.add_node(pair[0], label=f"{self.name_map[pair[0]]}:\nPE{pe_counter}", pe=pe_counter, pos=(x1_pos, (depth + 1)/scale))
            self.graph.add_node(pair[1], label=f"{self.name_map[pair[1]]}:\nPE{pe_counter}", pe=pe_counter, pos=(x2_pos, (depth + 1)/scale))
            self.graph.add_node(self.total_ops, label=f"PE{pe_counter}", pe=pe_counter, pos=(x3_pos, depth/scale))
            self.graph.add_edge(pair[1], self.total_ops)
            self.graph.add_edge(pair[0], self.total_ops)
            self.operations.append(self.total_ops)
            self.total_ops += 1

            self.mul_pairings[pe_counter].append(pair)
            pe_counter -= 1
            if pe_counter == 0:
                pe_counter = self.num_pes - 1
            elif pe_counter % self.pes_per_pu == 0:
                pe_counter -= 1

        while len(self.operations) > 1:
            op1 = self.operations.popleft()
            op2 = self.operations.popleft()
            depth = np.ceil(np.log2(len(self.operations))) if len(self.operations) > 0 else -1/scale

            op1_node = self.graph.nodes[op1]
            op2_node = self.graph.nodes[op2]
            target_pe = max(op1_node['pe'], op2_node['pe'])
            x_pos = op1_node['pos'][0] + (op2_node['pos'][0] - op1_node['pos'][0])/2
            self.graph.add_node(self.total_ops, label=f"PE{target_pe}", pe=target_pe, pos=(x_pos, depth/scale))
            self.graph.add_edge(op1, self.total_ops)
            self.graph.add_edge(op2, self.total_ops)
            self.operations.append(self.total_ops)
            self.total_ops += 1
