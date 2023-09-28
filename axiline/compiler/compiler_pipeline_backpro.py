# old version
# from polymath.mgdfg.passes import register_pass, Pass
# from polymath.mgdfg.util import _flatten_iterable, is_iterable, extend_indices, squeeze_indices, get_indices
# from polymath import func_op, DEFAULT_SHAPES, UNSET_SHAPE, SCALAR_IDX

from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower
import polymath.polymath as pm
import pprint
from numbers import Integral
from collections import defaultdict
from itertools import product

import numpy as np
from math import ceil

class PipelineBackpro(Pass):
    def __init__(self, debug=False):
        self.debug = debug
        self.status = False
        self.stage1 = False
        self.stage2 = False
        self.stage3 = False
        self.matmul = []
        self.ctx = []
        self.counts = {"dim1": 0,"dim2": 0,"dim3": 0,'matmul':0 }
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
        # elif not  self.stage3:
        #     self.check_stage_3(node)
        return node

    # check stage 1
    # check if there is only one mul_slick/matmul connected to W,X
    # mul followed by a sum
    # check dimension (svm,logi,linear=1, reco=n)
    def check_stage_1(self, node):
        args = node.args
        # polymath old version
        # if isinstance(node,pm.mgdfg.from_onnx.node_definitions.matmul):
        if isinstance(node, pm.matmul):
            print(f"\n {node.name},{len(node.args)}")
            for i in range (len(args)-1):
                if args[i].op_name == 'state':
                    self.counts['dim_w1']=self.get_dim(args[i])
                elif args[i].op_name =='input':
                    self.counts['dim_x1'] = self.get_dim(args[i])
            self.stage1=True


    # check stage 2
    # only one connection to sum
    # check dimension before sgd (svm,logi,linear=1, reco=n)
    def check_stage_2(self, node):
        args = node.args
        # polymath old version
        # if isinstance(node, pm.mgdfg.from_onnx.node_definitions.matmul):
        if isinstance(node, pm.matmul):
            print(f"\n {node.name},{len(node.args)}")
            for i in range(len(args) - 1):
                if args[i].op_name == 'state':
                    self.counts['dim_w2'] = self.get_dim(args[i])
                elif args[i].op_name == 'input':
                    self.counts['dim_x2'] = self.get_dim(args[i])
            self.stage1 = True

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
            if node.op_name=='index':
                dim = node.args[1] - node.args[0] + 1
            elif node.op_name=='sub' or node.op_name=='mul' or node.op_name=='add':
                dim=1
            elif node.op_name=='var_index' and len(node.args)==2:
                if(isinstance(node.args[1],pm.Node)):
                    dim=self.get_dim(node.args[1])
                else:
                    dim=False
            else:
                dim=False
        elif len(node.args)==3:
            arg_list = node.args[1]
            dim = []
            if isinstance(arg_list, list):
                for arg in arg_list:
                    dim_arg = self.get_dim(arg)
                    dim.append(dim_arg)
        elif isinstance(node.shape,tuple) and all(isinstance(x, int) for x in node.shape):
            dim=node.shape
        else:
            print(node.shape.__class__)
            dim=False

        if dim:
            return dim
        else:
            print(f"\nError with get dimension")
            if self.debug:
                print(f"node name: {node.name}")
                print(f"node op name: {node.op_name}")
            return False