from axiline.compiler.templates import Templates
from axiline.compiler.impl_node import ImplNode
from axiline.compiler.compiler_template import PrePipeline
from axiline.compiler.compiler_pipeline import PrintNodes, AddSuccessors
from pathlib import Path
import json
import polymath.polymath as pm


class Optimization():
    def __init__(self, impl_graph, templates_path):
        self.impl_graph = impl_graph
        self.init_templates(templates_path)
        self.syn_pre_suc()
        self.traversal()
        self.syn_pre_suc()

    def traversal(self):
        for node in self.impl_graph:
            if not isinstance(node, ImplNode):
                exit("Error, not a impl node")
            if node.level == 0:
                node.stage = 2
            elif node.op_name == 'osip':
                node.stage = 1
            elif node.op_name == 'sgd':
                # self.remove_node_in_impl_graph(node)
                impl_mul = ImplNode()
                impl_mul.update_template(self.templates.data['mul'])
                impl_mul.stage = 2

                impl_pipe = ImplNode()
                impl_pipe.update_template(self.templates.data['pipe'])
                impl_pipe.stage = 2

                # update connections

                for id in node.predecessors:
                    predecessor = self.impl_graph[id]
                    if predecessor.level < 3:
                        # print(f"{predecessor.name}")
                        self.remove_node_predecessor(node.id,predecessor.id)
                        # if isinstance(predecessor, ImplNode):
                        self.remove_node_successor(predecessor.id, node.id)
                        self.add_node_successor(predecessor.id, impl_mul.id)
                        impl_mul.add_predecessor(predecessor.id)
                    elif 'Const' in predecessor.name:
                        # print(f"{predecessor.name}")
                        self.remove_node_predecessor(node.id,predecessor.id)
                        self.add_node_successor(predecessor.id,impl_mul.id)
                        impl_mul.add_predecessor(predecessor.id)

                node.update_template(self.op_templates.data['op_sgd'])
                node.stage = 3
                node.add_predecessor(impl_pipe)

                self.add_node_predecessor(node.id,impl_pipe.id)
                self.add_impl_node(impl_mul)
                self.add_impl_node(impl_pipe)
                self.add_node_predecessor(impl_pipe.id, impl_mul.id)

    def init_templates(self, templates_path):
        template_json_path =f"{templates_path}/template.json"
        if not Path(template_json_path).exists():
            exit(f"{template_json_path} does not exist!")
        with open(template_json_path, 'r') as init_template:
            init = json.load(init_template)
            self.templates = Templates(init['templates'])
            self.op_templates = Templates(init['op_templates'])

    def syn_pre_suc(self):
        for node in self.impl_graph:
            for id in node.predecessors:
                pre = self.impl_graph[id]
                if isinstance(pre,ImplNode) and (node.id not in pre.successors):
                    self.impl_graph[id].add_successor(node.id)
            for id in node.successors:
                suc = self.impl_graph[id]
                if isinstance(suc, ImplNode) and (node.id not in suc.predecessors):
                    self.impl_graph[id].add_predecessor(node.id)


    def remove_impl_node(self, node):
        if isinstance(node, ImplNode) and node in self.impl_graph:
            self.impl_graph.remove(node)

    def add_impl_node(self, node):
        if isinstance(node, ImplNode) and node not in self.impl_graph:
            self.impl_graph.append(node)

    def remove_node_predecessor(self, node_id, predecessor_id):
        #print(f"remove_node_predecessor- {self.impl_graph[node_id].name} -{self.impl_graph[predecessor_id].name}")
        if isinstance(self.impl_graph[node_id],ImplNode):
            self.impl_graph[node_id].remove_predecessor(predecessor_id)
            self.impl_graph[predecessor_id].remove_successor(node_id)

    def add_node_predecessor(self, node_id, predecessor_id):
        #print(f"add_node_predecessor-{self.impl_graph[node_id].name} -{self.impl_graph[predecessor_id].name}")
        if isinstance(self.impl_graph[node_id],ImplNode):
            self.impl_graph[node_id].add_predecessor(predecessor_id)

    def remove_node_successor(self, node_id, successor_id):
        #print(f"remove_node_successor-{self.impl_graph[node_id].name} -{self.impl_graph[successor_id].name}")
        if isinstance(self.impl_graph[node_id],ImplNode):
            self.impl_graph[node_id].remove_successor(successor_id)
            self.impl_graph[successor_id].remove_predecessor(node_id)

    def add_node_successor(self, node_id, successor_id):
        #print(f"add_node_successor-{self.impl_graph[node_id].name} -{self.impl_graph[successor_id].name}")
        if isinstance(self.impl_graph[node_id],ImplNode):
            self.impl_graph[node_id].add_successor(successor_id)