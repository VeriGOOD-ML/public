from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower
# from polymath.mgdfg.util import _flatten_iterable, is_iterable, extend_indices, squeeze_indices, get_indices
# from polymath import func_op, DEFAULT_SHAPES, UNSET_SHAPE, SCALAR_IDX
import polymath.polymath as pm
import json
from pathlib import Path
from numbers import Integral
from collections import defaultdict
from itertools import product
from axiline.compiler.impl_node import ImplNode
from axiline.compiler.templates import Templates, Pattern

class PrePipeline(Pass):
    def __init__(self, template_path, debug=False):
        self.debug = debug
        self.ctx = []
        self.impl_graph = []
        self.impl_mem = []
        self.impl_param = []
        self.impl_output = []
        # definition of mem and other nodes
        self.mem_op_name = ['state', 'input','parameter']
        self.output_op_name =['write']
        self.other_op_names = ['tf2onnx', 'index', 'var_index', 'cast','write']
        self.init_parameters(template_path)
        # if isinstance(parameter,dict):
        #     self.parameter =parameter
        # else:
        #     exit("Parameter need to be a dict!")

    def init_parameters(self,template_path):
        parameter_json_path =f"{template_path}/template.json"
        if not Path(parameter_json_path).exists():
            exit(f"Error, path ({parameter_json_path}) not exist")
        with open(parameter_json_path, 'r') as init_template:
            init = json.load(init_template)
        self.parameter = init['parameters']
        self.templates = Templates(init['templates'])
        self.op_templates = Templates(init['op_templates'])

    # map traversal at high level
    def apply_pass(self, node, counts):
        # if is not mapped
        if self.debug:
            print(f"Current Node: {node.op_name}- {node.name}")
        if not self.check_mapped(node):
            self.map(node)
        return node

    # map traversal at low level
    def finalize_pass(self, node, counts):
        if not('mapped' in node.kwargs.keys() and node.kwargs['mapped']==True):
            # print(f"Name: {node.name}, OP: {node.op_name}")
            self.map_low_level(node)
        return node

    def check_mapped(self, node):

        if node.op_name in self.other_op_names:
            node.kwargs['mapped'] = True
            return True
        elif 'mapped' in node.kwargs.keys():
            if node.kwargs['mapped']:
                return True
            else:
                return False
        else:
            return False

    def map(self, node):
        if self.match_memory(node):
            self.map_memory(node)
        elif self.match_output(node):
            self.map_output(node)
        else:
            status = self.map_high_level(node)
        # elif self.match_low_level(node):
        #     self.map_low_level(node)
        return node

    def match_memory(self, node):
        if node.op_name in self.mem_op_name:
            return True
        else:
            return False

    def match_output(self, node):
        if node.op_name in self.output_op_name:
            return True
        else:
            return False

    def map_memory(self, node):
        node.kwargs['mapped']=True
        # print(f"memory :{node.name} -- {node.op_name}")
        impl = ImplNode()
        impl.level = 3
        impl.update_from_dfg_node(node)
        if node.op_name=='parameter' and node.name in self.parameter.keys():
            impl.value = self.parameter[node.name]
        # else:
        #     value = input(f"Error: need to define a int value for node {node.name}")
        #     if isinstance(value,int):
        #         impl.value=value
        node.kwargs['impl'] = impl
        impl.dim=impl.get_shape(node)
        self.impl_mem.append(impl)
        self.impl_graph.append(impl)

    def map_output(self, node):
        node.kwargs['mapped']=True
        # print(f"memory :{node.name} -- {node.op_name}")
        impl = ImplNode()
        impl.level = 3
        impl.update_from_dfg_node(node)
        node.kwargs['impl'] = impl
        impl.dim = impl.get_shape(node)
        self.impl_output.append(impl)
        self.impl_graph.append(impl)

    def map_high_level(self, node):
        high_level = self.templates.high_level
        for pattern in high_level:
            if isinstance(pattern,Pattern):
                if node.op_name==pattern.end:
                    # print(f"\n Current Node{node}, {pattern.end}")
                    status=self.match_high_level(node,pattern)

    def match_high_level(self, node, pattern):
        # template nodes, current nodes, pattern nodes
        t_nodes = [node]
        c_nodes = [node]
        p_nodes = pattern.nodes
        c_node = node
        for i in range(len(p_nodes)-1):
            p_node = p_nodes[i]
            c_node = self.args(c_node)
            c_nodes.extend(c_node)
            c_node = self.match_node(c_node, p_node)
            t_nodes.append(c_node)
            # if not match , return false, exit loop
            # if matched, return matched node, (c_node in next loop)
            if c_node == False:
                return False
        # matched
        c_node = self.args(c_node)
        c_nodes.extend(c_node)
        # set to mapped

        # generate Impl Node
        impl = ImplNode()
        name = pattern.name
        impl.update_template( self.templates.data[name],c_nodes)

        # add impl mapped information
        for t_node in t_nodes:
            t_node.kwargs['impl'] =impl
            t_node.kwargs['mapped'] = True
            # if predecessor is mapped
        # add predecessor
        for c_node in c_nodes:
            if c_node not in t_nodes:
                if 'impl' in c_node.kwargs.keys():
                    pred = c_node.kwargs['impl']
                    if isinstance(pred, ImplNode):
                        if pred.id not in impl.predecessors:
                            impl.add_predecessor(pred.id)
                    else:
                        c_node.kwargs["suc"] = impl

                else:
                    c_node.kwargs["suc"] = impl
        # if isinstance(node.shape[0],pm.Node):
        #     #impl.dim=(node.shape[0].args[0],node.shape[0].args[1])
        #     impl.dim =  node.shape[0].args[1]-node.shape[0].args[0]+1
        # else:
        #     impl.dim=node.shape
        self.impl_graph.append(impl)

        return True

    def match_node(self, c_nodes, p_node):
        if isinstance(c_nodes, list):
            #names = [c_node.op_name for c_node in c_nodes]
            for c_node in c_nodes:
                if p_node == c_node.op_name:
                    return c_node
            else:
                return False
        elif isinstance(c_nodes, pm.Node):
            if p_node == c_nodes.op_name:
                return c_nodes
            else:
                return False

    def args(self, node):
        re_args = []
        if not isinstance(node, pm.Node):
            exit("not a PM node")
        for arg in node.args:
            if isinstance(arg, list) and len(arg) == 1:
                arg = arg[0]
            if isinstance(arg, tuple) and len(arg) == 1:
                arg = arg[0]
            if arg.op_name != 'index':
                if arg.op_name == 'var_index' or arg.op_name == 'cast':
                    re = self.args(arg)
                    re_args.extend(re)
                else:
                    re_args.append(arg)
        return re_args

    # def match_low_level(self, node):
    #
    def map_low_level(self, node):
        low_level = self.templates.low_level
        for pattern in low_level:
            if node.op_name == pattern.name:
                temp=self.templates.data[pattern.name]
                self.match_lower_level(node,temp)
                return True
        print(f"Warning: node: {node.name} is not matched after high-level and low-level mapping")
        return False

    def match_lower_level(self, node, temp):
        impl = ImplNode()
        name = temp['name']

        impl.update_template(self.templates.data[name], node)
        if 'suc' in node.kwargs.keys():
            suc=node.kwargs['suc']
            if isinstance(suc,ImplNode):
                impl.add_successor(suc.id)
                if impl.id not in suc.predecessors:
                    id=suc.id
                    self.impl_graph[id].add_predecessor(impl.id)
                    # suc.add_predecessor(impl)
                    # print(f"adding predecessor {impl.name} to {suc.name}")
            else:
                exit("Error not a Impl Node")
        args=self.args(node)

        # need to fix the casting node
        for arg in args:
            if not 'impl' in arg.kwargs.keys():
                print(f"Error: Predecessor of Node{node.name} -- {arg.name} is not implemented!")
            else:
                pred =arg.kwargs['impl']
                if isinstance(pred,ImplNode):
                    pred.add_successor(impl.id)
                    if pred.id not in impl.predecessors:
                        impl.add_predecessor(pred.id)
                else:
                    exit(f"Error: Not a Impl Node: {pred}")
        node.kwargs['impl']=impl
        node.kwargs['mapped']=True
        self.impl_graph.append(impl)

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


    '''
        # # user defined trace function
        # def ip_trace(self, node):
        #     if node.op_name == "sum":
        #         check = self.args(node)
        #         if isinstance(check, list) and len(check) == 1:
        #             check = check[0]
        #         if check.op_name == 'slice_mul':
        #             check.kwargs['mapped'] = True
        #             node.kwargs['mapped'] = True
        #             impl = ImplNode()
        #             impl.update_template(self.templates.data['osip'])
        #             # only apply to one layer small ML
        #             impl.add_predecessor(self.impl_mem)
        #             check.kwargs['impl'] = impl
        #             node.kwargs['impl'] = impl
        #             self.impl_graph.append(impl)
        #             return True
        #         else:
        #             return False
        #
        # def sgd_trace(self, node):
        #     if node.op_name == "slice_sub":
        #         sub = node
        #         args1 = self.args(node)
        #         for arg1 in args1:
        #             if arg1.op_name == 'state':
        #                 w = arg1
        #             elif arg1.op_name == 'slice_mul':
        #                 mul2 = arg1
        #                 args2 = self.args(mul2)
        #                 for arg2 in args2:
        #                     if arg2.op_name == 'parameter':
        #                         mu=arg2
        #                     elif arg2.op_name == 'slice_mul':
        #                         mul1 = arg2
        #                         # print(arg2)
        #                         # mapping
        #                         sub.kwargs['mapped'] = True
        #                         mul2.kwargs['mapped'] = True
        #                         mul1.kwargs['mapped'] = True
        #                         #
        #                         impl = ImplNode()
        #                         impl.update_template(self.templates.data['sgd'])
        #                         sub.kwargs['impl'] = impl
        #                         mul2.kwargs['impl'] = impl
        #                         mul1.kwargs['impl'] = impl
        #                         # only apply to one layer small ML
        #                         impl.add_predecessor(self.impl_mem, )
        #                         self.impl_graph.append(impl)
    '''
