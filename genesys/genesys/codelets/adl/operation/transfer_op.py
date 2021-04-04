from .base_op import Operation
from typing import List, Union, Dict, Tuple, Callable
from codelets.adl.operation.operand import OperandTemplate, Offset, IndexedOperandTemplate
from . import Loop
import numpy as np
from itertools import chain
from numbers import Integral
from codelets.adl import util
from dataclasses import dataclass, replace, field
from copy import copy, deepcopy
from sympy import Basic, symbols, Idx, IndexedBase, Integer
from . import size_from_offsets, get_transfer_dim_sizes

OffsetType = Union[str, Integral, Basic]

# TODO: Check to make sure there are edges for the entire path
class Transfer(Operation):

    def __init__(self, operand: Union[OperandTemplate, IndexedOperandTemplate], path: List[str],
                 sizes=None,
                 add_codelet=True,
                 **kwargs):

        # TODO: Add type checking for offset lists
        # Set path, make sure there is at least 2 locations to transfer to/from
        assert len(path) >= 2
        self._path = path


        sizes = sizes or [[]]*(len(path)-1)

        # For each pair of nodes in the path, there needs to be a size
        assert len(path) == (len(sizes) + 1)
        self._access_indices = []
        req_params = []
        dependencies = []

        for s in sizes:
            if isinstance(s, str):
                req_params.append(s)

        super(Transfer, self).__init__('transfer', req_params,
                                       add_codelet=add_codelet,
                                       dependencies=dependencies,
                                       **kwargs)
        start_idx = len(operand.data_moves)
        operand = operand.add_transfer_access(path, self.op_str, sizes)
        end_idx = len(operand.data_moves)
        self._access_indices += list(range(start_idx, end_idx))

        self._operand = operand
        self._dependencies += [d for d in self._operand.dependencies if d not in self.dependencies]
        self._required_params += [r for r in self.operand.required_params if r not in self.required_params]

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def domain_offsets(self):
        accesses = self.operand.get_op_accesses(self.op_str)
        offsets = [a.domain_offsets for a in accesses]
        return offsets

    @property
    def offsets(self):
        accesses = self.operand.get_op_accesses(self.op_str)
        return list(accesses[0].offset_map.values())

    @property
    def sizes(self) -> List[List[Union[str, Integral]]]:
        # accesses = self.operand.get_op_accesses(self.op_str)
        # return [list(a.evaluated_offsets.values()) for a in accesses]
        # TODO: Fix the data movement shapes, they are incorrect
        sizes = [list(v.values()) for k, v in self.operand.tiling.items() if k in self.path]
        return sizes

    @property
    def data_transfer_sizes(self) -> List[Union[str, Integral]]:
        return [np.prod(s) for s in self.sizes]

    @property
    def operand(self) -> OperandTemplate:
        return self._operand

    @property
    def access_indices(self):
        return self._access_indices

    def get_src_movement(self, src, dst):
        accesses = self.operand.get_op_accesses(self.op_str)
        for i, a in enumerate(accesses):
            if a.src_node == src and a.dst_node == dst:
                return a
        raise KeyError

    def get_dst_movement(self, src, dst):
        accesses = self.operand.get_op_accesses(self.op_str)
        for a in accesses:
            if a.src_node == src and a.dst_node == dst:
                return a
        raise KeyError

    def get_outgoing_dst_movement(self, src, dst):
        accesses = self.operand.get_op_accesses(self.op_str)
        for i, a in enumerate(accesses):
            if a.src_node == src and a.dst_node == dst:
                return a
        raise KeyError

    def get_src_offset(self, src, dst):
        return self.get_src_movement(src, dst).domain_offsets()

    def get_dst_offset(self, src, dst):
        return self.get_dst_movement(src, dst).domain_offsets()

    def initialize_offsets(self, offset):
        if isinstance(offset, (list, tuple)):
            assert len(offset) == len(self.operand.shape_list)
            off = offset
        else:
            # TODO: Add check for numpy type as well
            assert isinstance(offset, int)
            off = [0] * len(self.operand.shape_list)
            off[-1] = offset
        arr_idx_symbols = []
        for dim, idx in enumerate(off):
            if isinstance(idx, Loop):
                arr_idx_symbol = idx.param_symbols[idx.op_str]
                self.dependencies.append(idx.op_str)
            elif isinstance(idx, Basic):
                arr_idx_symbol = idx
                loop_idx = idx.atoms(Idx)
                self.dependencies += [str(l) for l in list(loop_idx) if str(l) not in self.dependencies]
            elif isinstance(idx, str):
                self.required_params.append(idx)
                arr_idx_symbol = symbols(idx, integer=True)
            elif isinstance(idx, int):
                arr_idx_symbol = idx
            else:
                raise TypeError(f"Invalid type for loop index: {idx}, type: {type(idx)}")

            arr_idx_symbols.append(arr_idx_symbol)
        return arr_idx_symbols

    # TODO: FIx this
    def op_type_params(self):
        op_params = []
        for i, off in enumerate(self.offsets):
            if isinstance(off, List):
                offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in off])
            else:
                assert isinstance(off, Basic)
                offset_str = f"{off}"
            op_params.append(f"{self.path[i]}[{offset_str}]")

        return op_params

    def evaluate_parameters(self, node, hag, cdlt):
        pass

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: OPERAND: {self.operand.name}[{'->'.join(self.path)}], SIZES: {self.sizes}," \
                     f"OFFSETS: {self.offsets}"
        elif output_type == "json":
            transfer_info = {}
            for move in self.operand.data_moves:
                text_key = f"{move.src_node}->{move.dst_node}"
                transfer_info[text_key] = {}
                transfer_info[text_key]['size'] = move.size()
                transfer_info[text_key]['offset'] = [str(o) for o in move.domain_offsets()]

            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "operand": self.operand.name,
                      "transfer_path": self.path,
                      "transfers": transfer_info}
        else:
            op_str = []
            for ft in self.instructions:
                ft_out = ft.emit(output_type)
                if len(ft_out) == 0:
                    continue
                op_str += ft_out
        return op_str


    def copy(self, cdlt, operand=None, path=None, access_indices=None, transfers=None, **kwargs):
        obj = super(Transfer, self).copy(cdlt, **kwargs)
        obj._operand = operand or cdlt.get_operand(self.operand.name)
        obj._path = path or self.path.copy()
        obj._access_indices = access_indices or self._access_indices.copy()
        return obj



