import itertools
import polymath as pm

class ImplNode():
    id_iter = itertools.count()

    def __init__(self, name='', predecessor=[], successor=[], template={}, dim=0):
        self.id = next(ImplNode.id_iter)
        self.name = f"{name}_{self.id}"
        self.op_name = name,
        self.predecessors = []
        self.successors = []
        self.operation = []
        self.verilog_name = ''
        # add predecessor and successor
        self.add_successor(successor)
        self.add_predecessor(predecessor)
        self.template = dict.copy(template)
        self.dim = dim
        self.stage = 0
        self.level = 0
        self.value = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = f"{value}_{self.id}"

    def update_name(self, name):
        self.name = name

    def update_from_dfg_node(self, dfg_node):
        if not isinstance(dfg_node, pm.Node):
            exit("not a PM Node!")
        self.name = dfg_node.name
        self.op_name = dfg_node.name
        if dfg_node.name == 'sum':
            print(sum)
        # if len(dfg_node.args) and dfg_node.args[0].op_name == 'index':
        if dfg_node.op_name =='sum':
            index = dfg_node.args[0]
            if isinstance(index.args[0], int) and isinstance(index.args[1],int):
                dim = index.args[1] - index.args[0] + 1
                self.dim = dim

    def update_template(self, template, c_nodes=0):
        if not isinstance(template, dict):
            exit("not a template dict!")
        self.template = template
        self.level = template['level']
        self.op_name = template['name']
        self.name = template['name']
        self.operation = template['operation']
        if isinstance(c_nodes, list):
            for node in c_nodes:
                if 'slice_mul' in node.op_name or 'mat' in node.op_name:
                    self.dim = self.get_shape(node)
                if 'slice_sub' in node.op_name:
                    index = node.shape
                    self.dim = self.get_shape(index)
                    break
        elif isinstance(c_nodes, pm.Node):
            self.dim = self.get_shape(c_nodes)

    def get_shape(self,node):
        dim = None
        if isinstance(node, tuple):
            if node[0].op_name == 'index':
                if isinstance(node[0].args[0],int) and isinstance(node[0].args[1], int):
                    dim = node[0].args[1] - node[0].args[0] + 1
                    #print(f"DIM of node{node[0].name} is {dim}")
        if isinstance(node, pm.Node) and isinstance(node.shape, tuple):
            if len(node.shape) ==1:
                dim = node.shape[0]
            elif len(node.shape) == 2:
                dim = node.shape[1] - node.shape[0] + 1
            #print(f"DIM of node{node.name} is {dim}")
        return dim
        # need to update for SGD dimension! Oct 25
        # if isinstance(node, pm.Node):
        #     if node.op_name =='index':
        #         shape = node.args[1] - node.args[0] + 1
        #         return shape
        #     else:
        #         for arg in node.args:
        #             if isinstance(node, pm.Node) and node.op_name == 'index':
        #                 shape = self.get_shape(arg)
        #                 break
        #         return shape

    def add_predecessor(self, predecessor):
        if isinstance(predecessor, int) and predecessor not in self.predecessors:
            self.predecessors.append(predecessor)
        elif isinstance(predecessor, list):
            for node in predecessor:
                self.add_predecessor(node)

    def remove_predecessor(self, predecessor):
        if isinstance(predecessor, int) and predecessor in self.predecessors:
            self.predecessors.remove(predecessor)
        elif isinstance(predecessor, list):
            for node in predecessor:
                self.remove_predecessor(node)

    def add_successor(self, successor):
        if isinstance(successor, int) and successor not in self.successors:
            # print(f"add suc {successor} to {self.name}")
            self.successors.append(successor)
        elif isinstance(successor, list):
            for node in successor:
                self.add_successor(node)

    def remove_successor(self, successor):
        if isinstance(successor, int) and successor in self.successors:
            self.successors.remove(successor)
        elif isinstance(successor, list):
            for node in successor:
                self.remove_successor(node)
