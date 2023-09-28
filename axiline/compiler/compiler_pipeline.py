
from polymath.polymath.srdfg.passes import register_pass, Pass
from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower

# from polymath.mgdfg.passes import register_pass, Pass
# from polymath.mgdfg.util import _flatten_iterable, is_iterable, extend_indices, squeeze_indices, get_indices
# from polymath import func_op, DEFAULT_SHAPES, UNSET_SHAPE, SCALAR_IDX
import polymath.polymath as pm
from numbers import Integral
from collections import defaultdict
from itertools import product

import numpy as np
from math import ceil

# class AddSuccessor(Pass):
#     def __init__(self, debug=False):
#         self.debug = debug
#         self.ctx = []
#
#     def apply_pass(self, node, ctx):
#         # print(f"node_name:{node.name},class:{node.__class__},op_name:{node.op_name}")
#         node.add_attribute('successors',[])
#         if len(node.args):
#             for arg in node.args:
#                 if isinstance(arg,pm.Node):
#                     # arg.kwargs['successors'].append(node)
#                     arg.successors.append(node)
#         print(f"    kwargs:{node.kwargs.keys()}")
#         return node
#
#     def finalize_pass(self, node, ctx):
#         # print(f"    kwargs:{node.kwargs.keys()}")
#         return node
#
#
class PrintNodes(Pass):
    def __init__(self, debug=False):
        self.debug = debug
        self.ctx = []

    def apply_pass(self, node, ctx):
        # if (not (isinstance(node, pm.index) or isinstance(node, pm.var_index))):
        print(f"node_name:{node.name},class:{node.__class__},op_name:{node.op_name}")
        print(f"node_shape{node.shape}")
        # if node.index:
        #     print("    index",node.index.name)
        for i in range(len(node.args)):
            print(f"    args{i}:{node.args[i]}")
            if (hasattr(node.args[i], 'name')):
                print(f"    name:{node.args[i].name}")
        if isinstance(node.successors,list):
            for i in range(len(node.successors)):
                print(f"    suc{i}:{node.successors[i]}")
                if (hasattr(node.successors[i], 'name')):
                    print(f"    name:{node.node.successors[i].name}")
        # print(f"    kargs:{node.kwargs.keys()}")
        # print(f"    kargs:{node.graph}")
        return node


@register_pass(analysis=True)
class CountOpTypes(Pass):
    def __init__(self, skip=None):
        self.op_types = defaultdict(int)
        if skip:
            self.skip = skip
        else:
            self.skip = []
        super(CountOpTypes, self).__init__({})

    def apply_pass(self, node, counts):
        if node.op_name not in self.skip:
            self.op_types[node.op_name] += 1
        return node


@register_pass(analysis=True)
class AddSuccessors(Pass):
    def __init__(self, debug=False):
        self.debug = debug
        self.ctx = []

    def apply_pass(self, node, ctx):
        node.successors=[]
        for arg in node.args:
            if (isinstance(arg, pm.Node)):
                arg.successors.append(node)
        return node


class Pipeline(Pass):
    def __init__(self, debug=False):
        self.debug = debug
        self.status = False
        self.stage1 = False
        self.stage2 = False
        self.stage3 = False
        self.matmul = []
        self.ctx = []
        self.counts = {"dim": 0,'matmul':0 }
        self.stage1_hold = []
        self.stage2_hold = []
        self.stage3_hold = []
        self.stage1_sink = 0
        self.stage2_sink = 0
        self.stage3_sink = 0

    def apply_pass(self, node, counts):
        if node.op_name == 'Matmul' or node.op_name == 'slice_mul':
            self.counts['matmul'] += 1
            self.matmul.append(node)
        if not self.stage1:
            self.check_stage_1(node)
        elif not self.stage2:
            self.check_stage_2(node)
        elif not  self.stage3:
            self.check_stage_3(node)
        return node

    # check stage 1
    # check if there is only one mul_slick/matmul connected to W,X
    # mul followed by a sum
    # check dimension (svm,logi,linear=1, reco=n)
    def check_stage_1(self, node):
        args = node.args
        for arg in args:
            if isinstance(arg, pm.Node):
                if arg.op_name == 'slice_mul' or arg.name in self.stage1_hold:  # or arg.op_name=='matmul'):
                    if node.op_name == 'var_index':
                        self.stage1_hold.append(node.name)
                    elif (node.op_name == 'sum'):
                        self.stage1_hold = []
                        self.stage1 = True
                        self.stage1_sink = node
                        dim=self.get_dim(node)
                        self.counts['dim'] =dim
                        print(f"\nPassed, dimension initiated with {dim}")
                        try:
                            self.stage2_hold = node.successor
                        except AttributeError:
                            print(f"\nNode {node} do NOT have a successor attribute!")

    # check stage 2
    # only one connection to sum
    # check dimension before sgd (svm,logi,linear=1, reco=n)
    def check_stage_2(self, node):
        if self.update_stage2_sink(node):
            self.stage2_sink = node
            # print(node.name)

    # check stage 3 b after stage 2
    # SGD computation connected to output
    def check_stage_3(self, node):
        self.sgd_check(node)

    def update_stage2_sink(self, node):
        state = False
        # successor = self.track_node_forward(node)
        if node.op_name != 'index' and node.op_name!='var_index' and len(node.successor)==1:
            successor=node.successor[0]
            # training benchmarks
            if successor.op_name == 'slice_mul' and len(successor.args) == 2:
                state = self.sgd_start_check(successor)
                self.stage2_sink=node
                self.stage2 = True
                dim=self.get_dim(self.track_node_backward(node))
                if dim==1:
                    print(f"\nPassed, dimension of stage 2 matched")
                else:
                    print(f"\nError, dimension of stage 2 not match")
            # inference benchmarks
            elif successor.op_name == 'write':
                self.stage2_sink = node
                self.stage2 = True
                self.stage3 = True
                self.status = 'inference'
                print(f"\nPassed, dimension of stage 2 matched")

        return state

    def sgd_start_check(self, node):
        state = False
        if node.args[0].op_name == 'input':
            state = True
        elif isinstance(node.args[1], pm.Node):
            if node.args[1].op_name == 'var_index':
                state = self.sgd_start_check(node.args[1])
        return state

    def sgd_check(self,node):
        state=False
        if node.op_name!='index' and node.op_name!='var_index':
            if node.op_name=='slice_mul':
                if node.args[0]==self.stage2_sink or node.args[0].op_name=='parameter':
                    state=True
            elif node.op_name == 'slice_sub':
                arg=self.track_node_backward(node.args[0])
                if arg.op_name== 'state':
                    self.stage3_sink=node
                    state=True
                    dim = self.get_dim(node)
                    if dim == self.counts['dim']:
                        print(f"\nPassed, dimension of stage 3 matched")
                    else:
                        print(f"\nError, dimension of stage 3 not match")
                else:
                    print(f'SGD check failed! Other computation node detected!')
            elif node.op_name=='write':
                if self.track_node_backward(node)==self.stage3_sink:
                    self.stage3=True
                    self.status = 'training'
            else:
                print(f'\nSGD check failed! Other computation node detected!')
        return state

    def track_node_backward(self,node):
        if node.op_name=='var_index' or node.op_name=='index':
            if len(node.args)>0 and isinstance(node.args[0],pm.Node):
                state=self.track_node_backward(node.args[0])
            else:
                print("\nError occurs when tracking node backward!")
        elif isinstance(node,pm.Node):
            state=node
        else:
            print("\nError occurs when tracking node backward!")
        return state

    def track_node_forward(self,node):
        if node.op_name == 'var_index' or node.op_name=='index':
            try:
                successor= node.successor[0]
            except AttributeError:
                print(f"\nError occurs when tracking node forward, node {node} do NOT have a successor attribute!")
            state = self.track_node_forward(successor)
        elif isinstance(node, pm.Node):
            state = node
        else:
            print("\nError occurs when tracking node backward!")
        return state

    def get_dim(self,node):
        if len(node.args)==2:
            arg = node.args[1]
            if isinstance(arg,list):
                arg=arg[0]
            if arg.op_name=='sub' or arg.op_name=='mul' or arg.op_name=='add':
                dim=1
            elif arg.op_name=='index' and len(arg.args)==2:
                if isinstance(arg.args[0],int) and isinstance(arg.args[1],int):
                    dim=arg.args[1]-arg.args[0]+1
            elif arg.op_name=='var_index' and len(arg.args)==2:
                dim=self.get_dim(arg)
            else:
                print(f"\nError with get dimension")
                if self.debug:
                    print(f"node name: {node.name}")
                    print(f"node op name: {node.op_name}")
                    print(f"arg name: {arg.name}")
                    print(f"arg op name: {arg.op_name}")
                return False
        return dim

class VerilogGeneratePipeline(Pass):
    def __init__(self, param):
        self.param = param
