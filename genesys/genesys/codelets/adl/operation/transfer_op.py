from .base_op import Operation
from typing import List, Union, Dict, Tuple, Callable
from codelets.adl.operation.operand import Operand, Offset, IndexedOperandTemplate
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

    def __init__(self, operand: Union[Operand, IndexedOperandTemplate], path: List[str],
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
        req_params = {}
        dependencies = []

        for s in sizes:
            if isinstance(s, str):
                req_params[s] = None

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
        for r, v in self.operand.required_params.items():
            if r not in self.resolved_params:
                self._required_params[r] = None

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
        # TODO: Fix the data movement shapes, they are incorrect
        sizes = []
        for p in self.path:
            if p in self.operand.tiling:
                sizes.append(list(self.operand.tiling[p].values()))
        return sizes

    @property
    def data_transfer_sizes(self) -> List[Union[str, Integral]]:
        return [np.prod(s) for s in self.sizes]

    @property
    def operand(self) -> Operand:
        return self._operand

    @property
    def access_indices(self):
        return self._access_indices

    def sizes_for_node(self, node: str):
        assert node in self.path
        return self.sizes[self.path.index(node)]

    def get_contiguous_xfer_size(self):
        if np.prod(self.sizes[0]) < np.prod(self.sizes[1]):
            xfer_sizes = self.sizes[0]
            ref_sizes = self.sizes[1]
        else:
            xfer_sizes = self.sizes[1]
            ref_sizes = self.sizes[0]

        # rsize = 1
        # for i in range(len(xfer_sizes)):
        #     idx = len(xfer_sizes) - i - 1
        #     rsize *= xfer_sizes[idx]
        #     if xfer_sizes[idx] != ref_sizes[idx]:
        #         break
        rsize = 1
        for i in range(len(xfer_sizes)):
            idx = len(xfer_sizes) - i - 1
            rsize *= xfer_sizes[idx]
            if xfer_sizes[idx] != ref_sizes[idx]:
                break
        return rsize

    def strides_iters(self, data_width, divisor = 1, max_bits = 64, contiguous=False):
        assert len(self.sizes) == 2
        if np.prod(self.sizes[0]) < np.prod(self.sizes[1]):
            xfer_sizes = self.sizes[0]
            ref_sizes = self.sizes[1]
        else:
            xfer_sizes = self.sizes[1]
            ref_sizes = self.sizes[0]
        if contiguous:
        # TODO: include array size for req size divisibility
            idx = [0]
        else:
            idx = [i for i in range(len(xfer_sizes)) if xfer_sizes[i] != ref_sizes[i]]
        if 0 not in idx:
            idx.insert(0, 0)

        strides = []
        iters = []

        for p, c in zip(idx, idx[1:]):
            iters.append(np.prod(xfer_sizes[p:c], dtype=np.int32))
            strides.append(np.prod(ref_sizes[c:], dtype=np.int32))
        iters.append(1)

        strides.append(np.prod(xfer_sizes[idx[-1]:], dtype=np.int32))

        # dtype_strides = [self.operand.dtype.bytes()*s for s in strides]
        dtype_strides = [(self.operand.dtype.bits()//data_width)*s for s in strides]
        assert all([v != 0 and (self.operand.dtype.bits()*strides[i]) % data_width == 0 for i, v
                    in enumerate(dtype_strides)]), f"Invalid strides: " \
                                                   f"{dtype_strides} \n" \
                                                   f"{strides}\n" \
                                                   f"Datatype: {self.operand.dtype.bits()}\n" \
                                                   f"Width: {data_width}"
        total_req_size = dtype_strides[-1]
        if np.ceil(np.log2(total_req_size)) > max_bits:

            total_iters = (1+np.ceil(np.ceil(np.log2(total_req_size))/max_bits))

            while total_req_size % total_iters != 0 or total_req_size//total_iters % divisor == 0:
                total_iters += 1
            dtype_strides[-1] /= total_iters
            iters[-1] = total_iters
        final_strides = [np.int32(s) for s in dtype_strides]
        iters = [np.int32(i) for i in iters]

        return final_strides, iters

    def test_contig_strides(self):
        assert len(self.sizes) == 2
        if self.sizes[0] == self.sizes[1]:
            return [np.prod(self.sizes[0])], [np.prod(self.sizes[0])]
        elif np.prod(self.sizes[0]) < np.prod(self.sizes[1]):
            xfer_sizes = self.sizes[0]
            ref_sizes = self.sizes[1]
        else:
            xfer_sizes = self.sizes[1]
            ref_sizes = self.sizes[0]

        it1 = [1]
        st1 = [1]
        mult = 0
        for i, s in enumerate(xfer_sizes):
            if s == ref_sizes[i]:
                st1[-1] *= s
                it1[-1] *= s
                mult += 1
            else:
                if mult == 0:
                    st1[-1] = np.prod(ref_sizes[i:], dtype=np.int)
                else:
                    st1[-1] = st1[-1] * np.prod(ref_sizes[i + 1:], dtype=np.int)
                st1.append(s)
                it1.append(s)
                mult = 0
        # if self.operand.name in ["data", "out"]:

        it2 = [1]
        st2 = [np.prod(ref_sizes)]
        for i, s in enumerate(xfer_sizes):
            if s != ref_sizes[i]:
                st2.append(np.prod(ref_sizes[i + 1:]))
                it2.append(s)
            else:
                it2[-1] *= s
                st2[-1] = st2[-1] // ref_sizes[i]
        if st2[-1] == 1:
            st2[-1] = xfer_sizes[-1]

        # if self.operand.name == "out":
        # idx = [i for i in range(len(xfer_sizes)) if xfer_sizes[i] != ref_sizes[i]]
        idx = [0] + [i for i in range(len(xfer_sizes)) if xfer_sizes[i] != ref_sizes[i]]
        st3 = []
        it3 = []
        prev_idx = 0
        for p, c in zip(idx, idx[1:]):
            it3.append(np.prod(xfer_sizes[p:c], dtype=np.int32))
            st3.append(np.prod(ref_sizes[c:], dtype=np.int32))

        # for i, idx in enumerate(idx):
        #     it3.insert(0, np.prod(xfer_sizes[prev_idx:idx]))
        #     st3.insert(0, np.prod(ref_sizes[idx:]))
        #     prev_idx = idx
        it3.append(1)
        st3.append(np.prod(xfer_sizes[idx[-1]:], dtype=np.int32))
        done_req = False
        req_size = xfer_sizes[-1]

        print(f'Operand: {self.operand.name}, {self.path}')
        st4, it4 = self.strides_iters()
        print(f"Xfer sizes: {xfer_sizes}, Ref sizes: {ref_sizes}, Strides: {st1}, iterations: {it1}\n"
              f"Xfer sizes: {xfer_sizes}, Ref sizes: {ref_sizes}, Strides: {st2}, iterations: {it2}\n"
              f"Xfer sizes: {xfer_sizes}, Ref sizes: {ref_sizes}, Strides: {st3}, iterations: {it3}, {self.get_contiguous_xfer_size()}\n"
              f"Xfer sizes: {xfer_sizes}, Ref sizes: {ref_sizes}, Strides: {st4}, iterations: {it4}\n"
              f""
              f"")
        assert len(st2) == len(it2)
        return st2, it2

    def get_contiguous_strides(self):
        assert len(self.sizes) == 2
        if self.sizes[0] == self.sizes[1]:
            return [np.prod(self.sizes[0])], [np.prod(self.sizes[0])]
        elif np.prod(self.sizes[0]) < np.prod(self.sizes[1]):
            xfer_sizes = self.sizes[0]
            ref_sizes = self.sizes[1]
        else:
            xfer_sizes = self.sizes[1]
            ref_sizes = self.sizes[0]

        iterations = [1]
        strides = [1]
        mult = 0
        for i, s in enumerate(xfer_sizes):
            if s == ref_sizes[i]:
                strides[-1] *= s
                iterations[-1] *= s
                mult += 1
            else:
                if mult == 0:
                    strides[-1] = np.prod(ref_sizes[i:], dtype=np.int)
                else:
                    strides[-1] = strides[-1]*np.prod(ref_sizes[i+1:], dtype=np.int)
                strides.append(s)
                iterations.append(s)
                mult = 0
        # if self.operand.name in ["data", "out"]:

        it = [1]
        sss = [np.prod(ref_sizes)]
        for i, s in enumerate(xfer_sizes):
            if s != ref_sizes[i]:
                sss.append(np.prod(ref_sizes[i+1:]))
                it.append(s)
            else:
                it[-1] *= s
                sss[-1] = sss[-1]//ref_sizes[i]
        if sss[-1] == 1:
            sss[-1] = xfer_sizes[-1]

        assert len(sss) == len(iterations)
        return sss, iterations

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
                # self.required_params.append(idx)
                self._required_params[idx] = None
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
        path_str = " -> ".join(self.path)
        path_str += "["
        for i, off in enumerate(self.offsets):
            if isinstance(off, List):
                offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in off])
            else:
                assert isinstance(off, Basic)
                offset_str = f"{off}"
            path_str += f"{offset_str}, "
            # op_params.append(f"{self.path[i]}[{offset_str}]")
        path_str += "]"
        return [path_str]

    def evaluate_parameters(self, node, hag, cdlt):
        pass

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: OPERAND: {self.operand.name}[{'->'.join(self.path)}], SIZES: {self.sizes}"
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



