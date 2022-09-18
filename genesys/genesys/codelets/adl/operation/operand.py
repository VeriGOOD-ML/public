from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union, ClassVar
from collections import namedtuple
from functools import partial
from codelets import Datatype
from codelets.adl.flex_param import FlexParam
import numpy as np

from pytools import memoize, memoize_method
from collections import defaultdict
from . import pairwise
import polymath as pm
from numbers import Integral
from copy import deepcopy
from sympy import Basic, Idx, symbols, Integer, lambdify
from codelets.adl import util
from dataclasses import dataclass, field

@memoize
def sympy_as_str(o):
    return str(o)

@dataclass(frozen=True)
class Offset:
    dim: int
    loop_id: int
    stride: int
    dim_size: int
    offset: int

    @property
    def loop_name(self):
        return f"loop{self.loop_id}"

    def __str__(self):
        return f"DIM:{self.dim},LOOPID:{self.loop_id},OFFSET:{self.offset}"

@dataclass
class DataMovement:
    src_node: str
    dst_node: str
    operand_name: str
    shape_symbols: List[str]
    op_name: Union[str, None]
    shape_map: Dict[str, Integral]
    offset_map: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)
    evaluated_offsets: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)
    evaluated_domain_offsets: Dict[str, List[Union[int, str, Basic, Offset]]] = field(default_factory=dict)
    lambdified_expr: Dict[str, Any] = field(default_factory=dict)
    symbol_str_map: Dict[Basic, str] = field(default_factory=dict)
    symbol_atoms_map: Dict[Basic, str] = field(default_factory=dict)
    derived_sizes: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.set_symbol_maps()

    def set_symbol_maps(self):
        for o in self.offset_map.values():
            if isinstance(o, Basic):
                self.symbol_str_map[o] = str(o)
                self.symbol_atoms_map[o] = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in self.symbol_atoms_map[o]]

                for idx in self.symbol_atoms_map[o]:
                    self.symbol_str_map[idx] = str(idx)

                for oth in others:
                    self.symbol_str_map[oth] = str(oth)

    @memoize_method
    def get_symbol_str(self, obj):
        return self.symbol_str_map[obj]

    def get_symbol_atoms(self, obj):
        return self.symbol_atoms_map[obj]

    def __str__(self):
        path = f"PATH: {self.src_node}->{self.dst_node}"
        op = f"OP: {self.op_name}"
        offsets = f"OFFSETS: {self.offset_map}"
        return ", ".join([path, op, offsets])

    @property
    def shape_list(self):
        return list(self.shape_map.values())

    @property
    def unset_offsets(self):
        return all([v == 0 for v in self.offset_map.values()])

    def replace_atom(self, prev_name, new_name):
        pass

    def size(self):
        return np.prod(self.dim_sizes())

    def dim_sizes(self):
        return [self.shape_map[s] for s in self.shape_symbols]

    def domain_offsets(self):
        all_offsets = []
        for k, v in self.evaluated_domain_offsets.items():
            for i in v:
                all_offsets.append(i)
        return all_offsets

    def copy(self):
        return DataMovement(self.src_node,
                          self.dst_node,
                          self.operand_name,
                          deepcopy(self.shape_symbols),
                          self.op_name,
                          deepcopy(self.shape_map),
                          deepcopy(self.offset_map),
                          deepcopy(self.evaluated_offsets),
                          deepcopy(self.evaluated_domain_offsets),
                          )

    def substitute_offset_symbols(self, cdlt, dep_map, replacements):
        new_offset_map = {}

        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = self.get_symbol_atoms(o)
                for idx, i in enumerate(indices):
                    i_str = self.get_symbol_str(i)
                    if i_str in replacements and i_str in dep_map:
                        replacement_op = cdlt.op_map[dep_map[str(i)]]
                        assert replacement_op.op_type == "loop"
                        o.subs(i, replacement_op.get_symbol())
                new_offset_map[name] = o
            else:
                new_offset_map[name] = 0
        self.reinit_offset_map(new_offset_map)
        self.set_size_from_splits(cdlt, cdlt.domain_tiling)
        self.set_offset_map(cdlt, cdlt.domain_loop_map, dep_map)

    def update_offset_map(self, key, update_val):
        self.offset_map[key] = update_val
        self.set_symbol_maps()

    def reinit_offset_map(self, new_offset_map):
        self.offset_map = new_offset_map
        self.set_symbol_maps()

    def get_size_from_splits(self, cdlt, splits):
        sizes = {}

        src_level = cdlt.get_tile_level(self.src_node)
        dst_level = cdlt.get_tile_level(self.dst_node)
        if src_level > dst_level:
            level = dst_level
        else:
            level = src_level

        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = self.get_symbol_atoms(o)
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                rel_splits = []
                for idx, i in enumerate(indices):
                    i_as_str = self.get_symbol_str(i)
                    rel_splits.append(splits[i_as_str])

                    if i_as_str in self.derived_sizes:
                        max_vals[i] = self.derived_sizes[i_as_str]
                    else:
                        assert cdlt.op_map[i_as_str].end % splits[i_as_str] == 0, f"Invalid: {i_as_str}, {self.derived_sizes.keys()}, {self.operand_name}"
                        max_vals[i] = cdlt.op_map[i_as_str].end // splits[i_as_str] - 1

                max_vals.update({i: cdlt.required_params[self.get_symbol_str(i)].value for i in others})

                size = self.resolve_offset(o, max_vals) + 1

                if np.prod(rel_splits) == 1 and \
                        size < cdlt.get_operand(self.operand_name).shape_symbols[name]\
                        and level == 0:
                    size = cdlt.get_operand(self.operand_name).shape_symbols[name]

                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size

        return sizes

    def get_size_from_splits_derived(self, cdlt, splits, derived_sizes):
        sizes = {}

        src_level = cdlt.get_tile_level(self.src_node)
        dst_level = cdlt.get_tile_level(self.dst_node)
        if src_level > dst_level:
            level = dst_level
        else:
            level = src_level

        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = self.get_symbol_atoms(o)
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                rel_splits = []
                for idx, i in enumerate(indices):
                    i_as_str = self.get_symbol_str(i)
                    # assert cdlt.op_map[i_as_str].end % splits[i_as_str] == 0
                    rel_splits.append(splits[i_as_str])
                    if i_as_str in derived_sizes:
                        max_vals[i] = derived_sizes[i_as_str] - 1
                    else:
                        max_vals[i] = cdlt.op_map[i_as_str].end // splits[i_as_str] - 1

                max_vals.update({i: cdlt.required_params[self.get_symbol_str(i)].value for i in others})

                size = self.resolve_offset(o, max_vals) + 1

                if np.prod(rel_splits) == 1 and \
                        size < cdlt.get_operand(self.operand_name).shape_symbols[name]\
                        and level == 0:
                    size = cdlt.get_operand(self.operand_name).shape_symbols[name]

                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size

        return sizes

    def get_size_from_loops(self, cdlt, loops):
        sizes = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = self.get_symbol_atoms(o)
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                for idx, i in enumerate(indices):
                    i_as_str = self.get_symbol_str(i)
                    assert i_as_str in loops, f"Index is not in loops: {i}: {loops}, Codelet: {cdlt.op_name}"
                    max_vals[i] = loops[i_as_str] - 1

                max_vals.update({i: cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size
        return sizes

    def set_size_from_splits(self, cdlt, split_levels):
        src_level = cdlt.get_tile_level(self.src_node)
        dst_level = cdlt.get_tile_level(self.dst_node)
        if src_level > dst_level:
            level = dst_level
        else:
            level = src_level
        splits = defaultdict(lambda: 1)
        for lev in range(level):
            for key, loop in split_levels[lev+1].items():
                splits[key] *= loop
        self.shape_map = self.get_size_from_splits(cdlt, splits)

    def set_offset_map(self, cdlt, loop_shapes, dep_map=None):
        dep_map = dep_map or {}
        self.resolve_domain_offsets(cdlt, loop_shapes, dep_map)

    def resolve_shape_map(self, sizes):
        self.shape_map = {self.shape_symbols[i]: sizes[i] for i in range(len(sizes))}

    def resolve_domain_offsets(self, cdlt, loop_shapes, dep_map):

        for idx, (name, o) in enumerate(self.offset_map.items()):
            idx_offset = int(np.prod(self.shape_list[idx + 1:]))
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                dom_offsets = self.resolve_domain_offset(cdlt, o, indices, idx, loop_shapes, dep_map)
            else:
                dom_offsets = [Offset(idx, -1, idx_offset, int(self.shape_list[idx]), o)]
            self.evaluated_domain_offsets[name] = dom_offsets

    def resolve_domain_offset(self, cdlt, expr, indices, dim, loop_shapes, dep_map):
        offsets = []
        for f_sym in list(expr.free_symbols):
            if str(f_sym) in cdlt.required_params:
                expr = expr.subs(f_sym, cdlt.required_params[str(f_sym)].value)
        offset, coeffs, nonlin = util.split(expr, indices)
        dim_size = self.shape_list[dim]
        base_offset = int(np.prod(self.shape_list[dim + 1:]))

        for i, idx in enumerate(indices):
            coeff = coeffs[i]
            coeff *= base_offset
            if not isinstance(coeff, (Integer, Integral)) or not isinstance(dim_size, (Integer, Integral)):
                raise TypeError(f"Unable to compute domain offsets because coefficient is not an intenger:"
                                f"Coeff: {coeff}\tType: {type(coeff)}")
            if str(idx) in dep_map:
                str_idx = dep_map[str(idx)]
            else:
                str_idx = str(idx)
            o_info = Offset(dim, cdlt.op_map[str_idx].loop_id, int(coeff), int(dim_size), offset)
            offsets.append(o_info)
        return offsets


    def resolve_offsets(self, cdlt):
        for idx, (name, o) in enumerate(self.offset_map.items()):
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {i: cdlt.op_map[str(i)].end - 1 for i in indices}
                max_vals.update({i: cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
            else:
                size = o
            self.evaluated_offsets[name] = size

    def resolve_offset(self, expr: Basic, values: Dict[str, int]):
        if expr in self.lambdified_expr:
            f = self.lambdified_expr[expr]
        else:
            free_symbs = list(expr.free_symbols)
            f = lambdify(free_symbs, expr, "numpy")
            self.lambdified_expr[expr] = f

        args = tuple([values[f] for f in list(expr.free_symbols) if f in values])
        res = f(*args)

        if not isinstance(res, (Integer, Integral)):
            raise TypeError(f"Unable to compute domain domain_offsets because offset is not an integer:"
                            f"Offset: {expr}\tType: {type(expr)}")

        return int(res)



@dataclass
class Operand:
    name: str
    supported_dtypes: Union[List[Datatype]]
    shape_list: List[str]
    shape_symbols: Dict = field(default_factory=dict)
    tiling: Dict[str, Dict[str,int]] = field(default_factory=dict)
    data_path: List[str] = field(default_factory=list)
    dtype: Datatype = field(default=None)
    node_name: str = field(default=None)
    mem_locations: Dict[str, Dict[str, int]] = field(default_factory=dict)
    write_destination: str = field(default=None)
    evaluated_tiling: List[Tuple[str, List]] = field(default_factory=list, init=False)
    dependencies: List[str] = field(default_factory=list)
    data_moves: List[DataMovement] = field(default_factory=list)
    required_params: Dict[str, Union[FlexParam, int, None]] = field(default_factory=dict)
    current_codelet: ClassVar = field(default=None)
    compute_pad_dim: int = field(default=-1)
    permutation: tuple = field(default=tuple([]))
    static_padding: Dict[str, int] = field(default_factory=dict)
    dynamic_padding: Dict[str, int] = field(default_factory=dict)
    dim_order: List[int] = field(default=None)
    offset_memo: Dict = field(default_factory=dict)

    @property
    def data_size(self):
        assert self.dtype is not None
        return np.prod(self.shape)*self.dtype.bits()

    @property
    def mem_locations_set(self) -> bool:
        return all([loc in self.mem_locations for loc in set(self.data_path)])

    @property
    def unset_mem_locations(self) -> List[str]:
        return [loc for loc in set(self.data_path) if loc not in self.mem_locations]

    @property
    def unique_dependencies(self) -> List[str]:
        return list(set(self.dependencies))

    def add_padding(self, dimension, pad_size, symmetric=False, dynamic=False):
        if isinstance(dimension, str):
            assert dimension in self.shape_list
            key = dimension
        else:
            assert isinstance(dimension, int) and dimension < len(self.shape_list)
            key = self.shape_list[dimension]

        if symmetric:
            pad_val = 2*pad_size
        else:
            pad_val = pad_size

        if dynamic:
            self.dynamic_padding[key] = pad_val
        else:
            self.static_padding[key] = pad_val
            if len(self.shape_symbols) > 0:
                self.shape_symbols[key] += pad_size

    def set_dim_order(self, dims):

        if len(self.shape_symbols) > 0:
            for d in dims:
                assert d in self.shape_symbols
                val = self.shape_symbols.pop(d)
                self.shape_symbols[d] = val
        self.shape_list = dims

        for dm in self.data_moves:
            dm.shape_symbols = self.shape_list
            for dim in dm.shape_symbols:
                val = dm.offset_map.pop(dim)
                dm.update_offset_map(dim, val)
    @property
    def used(self):
        return len(self.data_path) != 0

    @property
    def current_location(self):
        if len(self.data_path) == 0:
            return None
        else:
            return self.data_path[-1]

    def is_tiled(self):
        return len(self.tiling.keys()) == len(self.unique_data_locations())

    def transfer_tile(self, transfer_op, outgoing):

        key = transfer_op.path[0]

        if key not in self.tiling:
            movement = transfer_op.get_src_movement(transfer_op.path[0], transfer_op.path[1])
            self.tiling[transfer_op.path[0]] = movement.shape_map

        else:
            # TODO: Check if already set
            pass

    def find_compute_rd_movement(self, op_name, compute_unit):
        tgt = None
        for dm in self.data_moves:
            if dm.op_name == op_name and dm.dst_node == compute_unit:
                return dm

        if tgt is None:
            raise RuntimeError(f"Unable to find {op_name} in movements for {self.name}"
                               f" with destination {compute_unit}.")

    def find_compute_wrt_movement(self, op_name, compute_unit):
        tgt = None
        for dm in self.data_moves:
            if dm.op_name == op_name and dm.src_node == compute_unit:
                return dm

        if tgt is None:
            raise RuntimeError(f"Unable to find {op_name} in movements for {self.name}"
                               f" with destination {compute_unit}.")

    def find_transfer_movement(self, op_name):
        tgt = None
        for dm in self.data_moves:
            if dm.op_name == op_name:
                tgt = dm
                break

        if tgt is None:
            raise RuntimeError(f"Unable to find {op_name} in data movements for {self.name}.")

    def get_tile_size(self, src_loc, dst_loc):
        move = None
        for dm in self.data_moves:
            if dm.dst_node == dst_loc and dm.src_node == src_loc:
                move = dm

        if move is None:
            raise RuntimeError(f"Could not find tile size for {src_loc}, {dst_loc}")
        stride_val = np.prod(move.shape_list)
        return stride_val.astype(np.int64)


    def get_offset(self,  cdlt, loop_id, hag, op_name, node_name, write=False, outer_loop=False):
        mem_key = (loop_id, op_name, node_name, write, outer_loop)
        if mem_key in self.offset_memo:
            return self.offset_memo[mem_key]

        if len(self.data_moves) > 0 and (
                self.data_moves[-1].src_node == "IMM" or self.data_moves[-1].dst_node == "IMM"):
            self.offset_memo[mem_key] = 0
            return 0

        if not outer_loop:
            assert cdlt.op_map[op_name].target == node_name
            if write:
                target_movement = self.find_compute_wrt_movement(op_name, node_name)
            else:
                target_movement = self.find_compute_rd_movement(op_name, node_name)
        else:
            if write:
                target_movement = self.get_store_transfer(cdlt.op_map[op_name].target, node_name, op_name)
            else:
                target_movement = self.get_load_transfer(node_name, cdlt.op_map[op_name].target, op_name)


        if cdlt.get_tile_level(target_movement.src_node) > cdlt.get_tile_level(target_movement.dst_node):
            node_key = target_movement.src_node
            other_key = target_movement.dst_node
        else:
            node_key = target_movement.dst_node
            other_key = target_movement.src_node

        offset_val = None

        other_offsets = []

        if all([o.loop_id == -1 for o in target_movement.domain_offsets()]):
            dm_info = "\n".join([str({"src": dm.src_node,
                                      "dst": dm.dst_node,
                                      "dst_lvl": cdlt.get_tile_level(dm.dst_node),
                                      "src_lvl": cdlt.get_tile_level(dm.src_node)
                                      }) for dm in self.data_moves])
            raise RuntimeError(f"All loop ids unset for target movement."
                               f"Could not find data movement for op {op_name} in operand "
                               f"{self.name}: {self.data_path}\n"
                               f"Loop id: {loop_id}\n"
                               f"Data movements: {dm_info}\n"
                               f"Target movement info:\n"
                               f"src: {target_movement.src_node} -> {target_movement.dst_node}\n"
                               f"{[o.loop_id for o in target_movement.domain_offsets()]}")
        else:
            dom_offsets = target_movement.domain_offsets()

        for o in dom_offsets:
            if o.loop_id == loop_id:
                offset_val = o
            # TODO: Check to make sure other loops are nested
            elif o.loop_id > loop_id:
                other_offsets.append(o)

        if offset_val is None:
            self.offset_memo[mem_key] = 0
            return 0

        other_sizes = [1]
        acc_dims = [offset_val.dim]

        for o in dom_offsets:

            if o.loop_id > loop_id and o.dim not in acc_dims:
                if node_key not in self.tiling:
                    raise RuntimeError(f"Unable to find tiling for node key:\n"
                                       f"{self.name}:\n"
                                       f"node key: {node_key}\n"
                                       f"SRC->DST: {target_movement.src_node} -> {target_movement.dst_node}\n"
                                       f"Tiling: {self.tiling}"
                                       )
                other_sizes.append(self.tiling[node_key][self.shape_list[o.dim]])
                acc_dims.append(o.dim)

        src_node = hag.get_subgraph_node(target_movement.src_node)
        dst_node = hag.get_subgraph_node(target_movement.dst_node)

        if src_node.name == "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            width = src_node.banks
            if (("conv" in cdlt.op_name and loop_name in ["IC", "OC"]) \
                or ("gemm" in cdlt.op_name)) and cdlt.is_direct_loop_dep(cdlt.op_map[loop_str], "pe_array"):
                width = np.sqrt(width)
        elif src_node.node_type == "compute":
            width = dst_node.banks
        elif dst_node.node_type == "compute":
            width = src_node.banks
        else:
            width = 1

        if outer_loop and dst_node.name == "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            loop = cdlt.op_map[loop_str]
            tile_sizes = []
            num_tiles = []
            for i, o in enumerate(dom_offsets):
                tile_sizes.append(self.tiling[node_key][self.shape_list[o.dim]])

                if o.dim in acc_dims and o.dim != offset_val.dim:
                    num_tiles.append(self.tiling[other_key][self.shape_list[i]] // tile_sizes[i])
            tile_size = np.prod([1] + tile_sizes)
            tot_tiles = np.prod([1] + num_tiles)
            stride_val = tile_size * tot_tiles
        else:
            stride_val = offset_val.stride

        if outer_loop and dst_node.name != "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            loop = cdlt.op_map[loop_str]
            stride_val *= loop.stride
        offset = np.ceil(stride_val / width).astype(np.int64)
        self.offset_memo[mem_key] = offset
        return offset

    # 'up' -> dram -> compute unit
    # 'down' -> compute unit -> dram
    def get_offset_(self, cdlt, level, loop_id, hag, movement_type='up', zero_not_found=True, outer_loop=False):

        if movement_type == 'up':
            prev_level = level + 1
        else:
            prev_level = level - 1

        assert prev_level >= 0
        target_movement = None

        for dm in self.data_moves:

            if movement_type == 'up' and cdlt.get_tile_level(dm.dst_node) == level:
                target_movement = dm
                break
            elif movement_type == 'down' and cdlt.get_tile_level(dm.src_node) == level and cdlt.get_tile_level(dm.dst_node) == prev_level:
                target_movement = dm
                break
            elif movement_type == 'up' and self in cdlt.outputs and cdlt.get_tile_level(dm.src_node) == level:
                target_movement = dm
                break

        if len(self.data_moves) > 0 and (self.data_moves[-1].src_node == "IMM" or self.data_moves[-1].dst_node == "IMM"):

            return 0

        if target_movement is None:
            dm_info = "\n".join([str({"src": dm.src_node,
                                      "dst": dm.dst_node,
                                      "dst_lvl": cdlt.get_tile_level(dm.dst_node),
                                      "src_lvl": cdlt.get_tile_level(dm.src_node)
                                      }) for dm in self.data_moves])
            raise RuntimeError(f"Could not find data movement for level {level} in operand "
                               f"{self.name}: {self.data_path}\n"
                               f"Movement type: {movement_type}\n"
                               f"Loop id: {loop_id}\n"
                               f"Data movements: {dm_info}")


        if cdlt.get_tile_level(target_movement.src_node) > cdlt.get_tile_level(target_movement.dst_node):
            node_key = target_movement.src_node
            other_key = target_movement.dst_node
        else:
            node_key = target_movement.dst_node
            other_key = target_movement.src_node

        offset_val = None

        other_offsets = []

        if all([o.loop_id == -1 for o in target_movement.domain_offsets()]):
            dm_info = "\n".join([str({"src": dm.src_node,
                                      "dst": dm.dst_node,
                                      "dst_lvl": cdlt.get_tile_level(dm.dst_node),
                                      "src_lvl": cdlt.get_tile_level(dm.src_node)
                                      }) for dm in self.data_moves])
            raise RuntimeError(f"All loop ids unset for target movement."
                               f"Could not find data movement for level {level} in operand "
                               f"{self.name}: {self.data_path}\n"
                               f"Movement type: {movement_type}\n"
                               f"Loop id: {loop_id}\n"
                               f"Data movements: {dm_info}\n"
                               f"Target movement info:\n"
                               f"src: {target_movement.src_node} -> {target_movement.dst_node}\n"
                               f"{[o.loop_id for o in target_movement.domain_offsets()]}")
        else:
            dom_offsets = target_movement.domain_offsets()

        for o in dom_offsets:
            if o.loop_id == loop_id:
                offset_val = o
            # TODO: Check to make sure other loops are nested
            elif o.loop_id > loop_id:
                other_offsets.append(o)

        if offset_val is None:

            if zero_not_found:
                return 0
            else:
                raise RuntimeError(f"Could not find offset movement {level} from "
                                   f"{target_movement.src_node}->{target_movement.dst_node} "
                                   f"in operand {self.name}")
        other_sizes = [1]
        acc_dims = [offset_val.dim]

        for o in dom_offsets:


            if o.loop_id > loop_id and o.dim not in acc_dims:
                if node_key not in self.tiling:
                    raise RuntimeError(f"Unable to find tiling for node key:\n"
                                       f"{self.name}:\n"
                                       f"node key: {node_key}\n"
                                       f"SRC->DST: {target_movement.src_node} -> {target_movement.dst_node}\n"
                                       f"Tiling: {self.tiling}"
                                       )
                other_sizes.append(self.tiling[node_key][self.shape_list[o.dim]])
                acc_dims.append(o.dim)

        src_node = hag.get_subgraph_node(target_movement.src_node)
        dst_node = hag.get_subgraph_node(target_movement.dst_node)
        if src_node.name == "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            width = src_node.banks
            if (("conv" in cdlt.op_name and loop_name in ["IC", "OC"]) \
                    or ("gemm" in cdlt.op_name)) and cdlt.is_direct_loop_dep(cdlt.op_map[loop_str], "pe_array"):
                width = np.sqrt(width)
        elif src_node.node_type == "compute":
            width = dst_node.banks
        elif dst_node.node_type == "compute":
            width = src_node.banks
        else:
            width = 1

        if outer_loop and dst_node.name == "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            loop = cdlt.op_map[loop_str]
            tile_sizes = []
            num_tiles = []
            for i, o in enumerate(dom_offsets):
                tile_sizes.append(self.tiling[node_key][self.shape_list[o.dim]])

                if o.dim in acc_dims and o.dim != offset_val.dim:
                    num_tiles.append(self.tiling[other_key][self.shape_list[i]]//tile_sizes[i])
            tile_size = np.prod([1] + tile_sizes)
            tot_tiles = np.prod([1] + num_tiles)
            stride_val = tile_size*tot_tiles
        else:
            stride_val = offset_val.stride

        if outer_loop and dst_node.name != "WBUF":
            loop_str = f"loop{loop_id}"
            loop_name = cdlt.loop_param_map[loop_str]
            loop = cdlt.op_map[loop_str]
            stride_val *= loop.stride

        # if not outer_loop and dst_node.name != "WBUF" and level == 2:
        #     loop_str = f"loop{loop_id}"
        #     loop_name = cdlt.loop_param_map[loop_str]
        #     width = cdlt.param_tiling[level][loop_name]

        # if self.name == "data":
        #     varname = cdlt.loop_param_map['loop' + str(loop_id)]
        #     print(f"{self.name}, Loop {loop_id}, {varname}\n"
        #           f"Width: {width}\n"
        #           f"Stride val: {stride_val}")
        #     print(f"Level: {level}\n"
        #           f"{cdlt.param_tiling[2][varname]}\n")


        return np.ceil(stride_val/width).astype(np.int64)


    def compute_tile(self, compute_op, operand_type):
        if operand_type == "source":
            movement = compute_op.get_src_movement(self.name)
            if movement.src_node not in self.tiling:
                self.tiling[movement.src_node] = movement.shape_map
        elif operand_type == "dest":
            movement = compute_op.get_dest_movement(self.name)
            if movement.dst_node not in self.tiling:
                self.tiling[movement.dst_node] = movement.shape_map
        else:
            raise RuntimeError(f"Invalid operand type {operand_type} for tiling computation.\n"
                               f"Possible values: 'source', 'dest'")

        return movement

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            assert not isinstance(item, list)
            item = (item,)
        offsets = []

        for idx in item:
            if isinstance(idx, Basic):
                off = idx
                loop_idx = idx.atoms(Idx)
                self.dependencies += [str(l) for l in list(loop_idx) if str(l) not in self.dependencies]
            elif isinstance(idx, str):
                off = symbols(idx, integer=True)
                self.required_params[idx] = None
                # self.required_params.append(idx)
            elif isinstance(idx, int):
                off = idx
            else:
                try:
                    off = idx.param_symbols[idx.op_str]
                except TypeError:
                    off = idx
                self.dependencies.append(idx.op_str)
            offsets.append(off)
        if len(offsets) != len(self.shape_list):
            raise RuntimeError("Too many address offsets for operand shape:\n"
                               f"Operand: {self.name}\n"
                               f"Shape: {self.shape_list}\n"
                               f"Offsets: {offsets}")
        return IndexedOperandTemplate(self, offsets)

    def set_write_destination(self, location):
        self.write_destination = location

    def get_access_offsets(self, offsets):

        if len(offsets) == 0 and len(self.data_moves) > 0:

            a_offsets = {}
            for i in range(len(self.shape_list)):
                offset_values = list(self.data_moves[-1].offset_map.values())
                omap_val = offset_values[i]
                if isinstance(omap_val, int):
                    a_offsets[self.shape_list[i]] = omap_val
                else:
                    a_offsets[self.shape_list[i]] = omap_val.copy()
        elif len(offsets) == 0:
            a_offsets = {self.shape_list[i]: 0 for i in range(len(self.shape_list))}
        else:
            a_offsets = {}
            for i in range(len(self.shape_list)):
                if isinstance(offsets[i], int):
                    a_offsets[self.shape_list[i]] = 0
                else:
                    a_offsets[self.shape_list[i]] = offsets[i].copy()
        return a_offsets

    def add_dependency(self, op_name):
        if op_name not in self.dependencies:
            self.dependencies.append(op_name)

    def movement_keys(self):
        return [(dm.src_node, dm.dst_node) for dm in self.data_moves]

    def add_transfer_access(self, path, op_name, sizes, offsets=None):
        offsets = offsets or []
        self.add_dependency(op_name)
        pairs = pairwise(path)
        all_pairs = []

        for i, (src, dst) in enumerate(pairs):
            if self.current_location != src:
                self.data_path.append(src)

            if self.current_location != dst:
                self.data_path.append(dst)

            if src == path[0]:
                if len(self.data_moves) >= 1 and self.data_moves[-1].dst_node is None:
                    if self.data_moves[-1].src_node == src:
                        self.data_moves[-1].dst_node = dst
                    else:
                        self.data_moves[-1].dst_node = src


            if len(self.data_moves) > 0 and self.data_moves[-1].unset_offsets:
                self.data_moves[-1].reinit_offset_map(self.get_access_offsets(offsets))
                all_pairs.append((self.data_moves[-1].src_node, self.data_moves[-1].dst_node))

            dm_offsets = self.get_access_offsets(offsets)
            shape = self.get_shape_map(sizes[i])

            movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, dm_offsets)
            self.data_moves.append(movement)

        if len(all_pairs) == 1 and len(self.data_moves) > 2:
            last_idx = len(self.data_moves) - 2
            last_move = self.data_moves[last_idx]
            src, dst = all_pairs[0]

            while last_idx >= 0 and (src, dst) == (last_move.src_node, last_move.dst_node) or \
                    (dst, src) == (last_move.src_node, last_move.dst_node):

                if last_move.unset_offsets:
                    last_move.reinit_offset_map(self.get_access_offsets(offsets))
                last_idx -= 1
                last_move = self.data_moves[last_idx]

        return self

    def get_mem_index(self, loc) -> int:
        assert loc in self.mem_locations, f"Not a valid location for this operand"
        return self.mem_locations[loc]['index']

    def get_mem_offset(self, loc) -> int:
        assert loc in self.mem_locations, f"Not a valid location for this operand"
        return self.mem_locations[loc]['offset']

    def add_compute_access(self, target, op_name, operand_type, offsets=None):

        assert operand_type in ["source", "dest"]
        offsets = offsets or []

        offsets = self.get_access_offsets(offsets)
        self.add_dependency(op_name)

        shape = self.get_shape_map([0] * len(self.shape_list))
        if operand_type == "source":
            src = self.current_location
            dst = target

            if src != target:
                self.data_path.append(target)

            if src == dst:
                src = self.data_path[-2]
        else:
            src = target
            dst = self.write_destination

            if self.current_location != target:
                self.data_path.append(target)

            if self.current_location != self.write_destination and self.write_destination is not None:
                self.data_path.append(self.write_destination)

        movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, offsets)
        self.data_moves.append(movement)

        return self

    def update_transfer_access(self, new_op, outgoing=False):
        pairs = list(pairwise(new_op.path))
        for a in self.data_moves:
            key = (a.src_node, a.dst_node)
            if key in pairs:
                a.op_name = new_op.op_str
        self.transfer_tile(new_op, outgoing)

    def update_offset_maps(self, op, dep_map):
        if op.op_type == "compute":
            if self in op.sources:
                movement = op.get_src_movement(self.name)
            else:
                assert self in op.dests
                movement = op.get_dest_movement(self.name)
        elif op.op_type == "transfer":
            movement = op.get_src_movement(op.path[0], op.path[1])

    def update_op_accesses(self, cdlt, op, dep_map: Dict[str, str]):

        accesses = self.get_op_accesses(op.op_str)
        for a in accesses:
            a.substitute_offset_symbols(cdlt, dep_map, op.dependencies)

    def is_dtype_supported(self, dtype_name) -> bool:
        return dtype_name in [str(dt) for dt in self.supported_dtypes]

    def supported_dtype_str(self):
        return [str(dt) for dt in self.supported_dtypes]

    def get_dtype_from_str(self, dtype_str: str):
        for dt in self.supported_dtypes:
            if str(dt) == dtype_str:
                return dt
        raise RuntimeError(f"{dtype_str} is not a supported datatype for operand {self.name}")

    def update_shape_symbols(self, shape_key, value):
        if shape_key in self.shape_symbols and self.shape_symbols[shape_key] != value:
            raise KeyError(f"Value for shape_symbols {shape_key} has already been set:\n"
                           f"Previous value: {self.shape_symbols[shape_key]}\n"
                           f"New value: {value}")
        assert isinstance(value, Integral)
        self.shape_symbols[shape_key] = value

    def is_instantiated(self):
        return len(list(self.shape_symbols.keys())) == len(self.shape_list)

    def set_dtype(self, dtype: Union[Datatype, str]):
        if isinstance(dtype, Datatype):
            dtype_str = str(dtype)
        else:
            dtype_str = dtype

        if not self.is_dtype_supported(dtype_str):
            raise RuntimeError(f"{dtype_str} is not a supported datatype for operand {self.name}.\n"
                               f"Possible datatype strings: {self.supported_dtype_str()}")

        dtype_var = self.get_dtype_from_str(dtype_str)
        self.dtype = dtype_var

    def set_node_name(self, name):
        self.node_name = name

    def get_op_accesses(self, op_name: str):
        all_accesses = []
        for dm in self.data_moves:
            if dm.op_name == op_name:
                all_accesses.append(dm)
        if len(all_accesses) == 0:
            raise KeyError(f"Unablet to find movement for operation {op_name}\n"
                           f"Names: {[dm.op_name for dm in self.data_moves]}")
        return all_accesses

    # TODO: This needs to support multiple computations
    def get_compute_access_location(self, compute_node):
        for dm in self.data_moves:
            if dm.src_node == compute_node:
                return dm.dst_node
            elif dm.dst_node == compute_node:
                return dm.src_node

        raise KeyError(f"Unable to find Data Movement for operand {self.name}, "
                       f"Compute node: {compute_node}")

    def get_storage_access_location(self, storage_node):
        for dm in self.data_moves:
            if dm.src_node == storage_node:
                return dm.dst_node
            elif dm.dst_node == storage_node:
                return dm.src_node

        raise KeyError(f"Unable to find Data Movement for operand {self.name}, "
                       f"Compute node: {storage_node}")

    def has_transfer(self, path):
        # def x_in_y(query, base):

        try:
            l = len(path)
        except TypeError:
            l = 1
            query = type(self.data_path)((path,))

        for i in range(len(self.data_path)):
            if self.data_path[i:i + l] == path:
                return True
        return False


    def has_transfer_from_src(self, src):
        if src not in self.data_path:
            return False
        edges = [s for s,d in zip(self.data_path, self.data_path[1:])]
        return src in edges

    @property
    def shape(self):
        s = []
        for sname in self.shape_list:
            s.append(self.shape_symbols[sname])
        return tuple(s)

    @property
    def unset_tiling(self):
        return [k for k, v in self.tiling.items() if len(v) == 0]

    def unique_data_locations(self):
        return list(set(self.data_path))

    def is_tiling_set(self, path_key):
        if path_key in self.tiling:
            return len(self.tiling[path_key]) == len(self.shape_symbols)
        else:
            return False

    def get_shape_map(self,  size):
        if len(size) == len(self.shape_list):
            shape_dict = {s: size[idx] for idx, s in enumerate(self.shape_list)}
        elif len(size) == 0:
            shape_dict = {}
        else:
            assert len(size) == 1
            shape_dict = {s: 1 for idx, s in enumerate(self.shape_list)}
            shape_dict[self.shape_list[-1]] = size[0]
        return shape_dict


    def add_path_tiling(self, path_key: Tuple[str, str], dim_tiling: Dict[str, Union[str, int, None]]):
        if self.is_tiling_set(path_key):
            raise RuntimeError(f"Tiling for operand {self.name} has already been set!")

        if len(self.data_path) == 0:
            self.data_path.append(path_key[0])

        self.tiling[path_key] = []
        for k, v in dim_tiling.items():
            self.tiling[path_key].append(v)

        # TODO: Check logic here to make sure this is a valid data transfer
        # assert path_key[0] == self.data_path[-1]
        if self.current_location != path_key[1]:
            self.data_path.append(path_key[1])

    def get_ld_storage_location(self, cdlt, level, return_null=False):
        prev_level = level - 1
        assert prev_level >= 0
        for dm in self.data_moves:
            if cdlt.get_tile_level(dm.src_node) == prev_level:
                assert cdlt.get_tile_level(dm.dst_node) == level
                return dm.dst_node
            elif cdlt.get_tile_level(dm.dst_node) == prev_level:
                assert cdlt.get_tile_level(dm.src_node) == level
                return dm.src_node
        dm_info = "\n".join([str({"src": dm.src_node,
                                  "dst": dm.dst_node,
                                  "dst_lvl": cdlt.get_tile_level(dm.dst_node),
                                  "src_lvl": cdlt.get_tile_level(dm.src_node)
                                  }) for dm in self.data_moves])
        if return_null:
            return None
        raise RuntimeError(f"Unable to find storage node for level {level} in operand"
                           f" {self.name}:\n"
                           f"{dm_info}")

    def get_transfer_dest(self, src_node):
        if src_node not in self.data_path:
            raise RuntimeError(f"Not a valid source.")

        src_idx = None
        for i, d in enumerate(self.data_path[:-1]):
            if d == src_node:
                src_idx = i
                break
        if src_idx is None:
            raise RuntimeError(f"No such transfer exists for {self.name}"
                               f" with destination {src_node}:\n"
                               f"Data path: {self.data_path}")
        return self.data_path[src_idx + 1]

    def get_transfer_source(self, dst_node):
        if dst_node not in self.data_path:
            raise RuntimeError(f"Not a valid source.")

        dst_idx = None
        for i, d in enumerate(self.data_path):
            if d == dst_node and i > 0:
                dst_idx = i
                break
        if dst_idx is None:
            raise RuntimeError(f"No such transfer exists for {self.name}"
                               f" with destination {dst_node}:\n"
                               f"Data path: {self.data_path}\n")
        return self.data_path[dst_idx - 1]

    def set_start_location(self, location: str):
        if len(self.data_path) > 0:
            raise RuntimeError(f"Unable to set default location for operand {self.name} because "
                               f"path has already been initialized: {self.data_path}")
        self.data_path.append(location)

    def get_load_transfer(self, src_node, dst_node, compute_name):
        idx = -1
        for i, dm in enumerate(self.data_moves):
            if dm.op_name == compute_name and dm.dst_node == dst_node:
                idx = i
                break

        while idx >= 0:
            if self.data_moves[idx].src_node == src_node:
                return self.data_moves[idx]
            idx -= 1

        raise RuntimeError(f"Unable to find load movement between {src_node} "
                           f"and {dst_node} for {compute_name}")


    def get_store_transfer(self, src_node, dst_node, compute_name):
        idx = len(self.data_moves)
        for i, dm in enumerate(self.data_moves):
            if dm.op_name == compute_name and dm.src_node == src_node:
                idx = i
                break

        while idx < len(self.data_moves):
            if self.data_moves[idx].dst_node == dst_node:
                return self.data_moves[idx]
            idx += 1

        raise RuntimeError(f"Unable to find store movement between {src_node} "
                           f"and {dst_node} for {self.name} in {compute_name}")

    def evaluate_operand(self, node: pm.Node, hag, cdlt):

        initial_size = [self.shape_symbols[symb] for symb in self.shape_list]
        level_shapes = {}
        assert len(hag.node_levels[0]) == 1
        level_shapes[0] = initial_size
        for i, access in enumerate(self.data_moves):
            if access.dst_node is None:
                raise RuntimeError(f"Unset destination node for access for operand {self.name}:\n"
                                   f"Source: {access.src_node}\n"
                                   f"Dest: {access.dst_node}")
            access_level = hag.get_node_level(access.dst_node)

            if access_level in level_shapes:
                access.resolve_shape_map(level_shapes[access_level])
            else:
                #TODO: Come back to this code
                access.resolve_offsets(cdlt)

        first_tile = True

        # TODO: Add checks here
        for path_key, tiling_values in self.tiling.items():
            if first_tile:
                self.evaluated_tiling.append((path_key[0], initial_size))
                first_tile = False
            dest_tiling = []
            for k in tiling_values:
                if isinstance(k, str):
                    dest_tiling.append(cdlt.params[k].value)
                else:
                    assert isinstance(k, Integral)
                    dest_tiling.append(k)

            self.evaluated_tiling.append((path_key[1], dest_tiling))


    def copy(self):
        # TODO: Fix this
        op_temp = Operand(name=self.name,
                          supported_dtypes=self.supported_dtypes,
                          shape_list=self.shape_list,
                          shape_symbols=self.shape_symbols.copy(),
                          dependencies=self.dependencies.copy(),
                          tiling=deepcopy(self.tiling),
                          data_path=self.data_path.copy(),
                          mem_locations=self.mem_locations.copy(),
                          dtype=self.dtype,
                          node_name=self.node_name,
                          permutation=self.permutation,
                          data_moves=deepcopy(self.data_moves),
                          write_destination=self.write_destination)
        op_temp.evaluated_tiling = deepcopy(self.evaluated_tiling)
        return op_temp

    def emit(self, output_type):

        if output_type == "json":

            blob = {"name": self.name,
                    "unique_name": self.node_name,
                    "dtype": str(self.dtype),
                    "shape_symbols": {k: v for k, v in self.shape_symbols.items()},
                    "data_path": self.data_path,
                    "tiling": {k: v for k, v in self.tiling.items()}
                    }
        else:
            raise TypeError(f"Unable to support output type for operand: {output_type}")
        return blob

@dataclass
class IndexedOperandTemplate:
    operand_template: Operand
    offsets: List[Union[int, str, Basic]]

    @property
    def data_moves(self):
        return self.operand_template.data_moves

    def add_transfer_access(self, path, op_name, sizes):
        return self.operand_template.add_transfer_access(path, op_name, sizes, self.offsets)

    def add_compute_access(self, target, op_name, operand_type):
        return self.operand_template.add_compute_access(target, op_name, operand_type, self.offsets)

    @property
    def offset_names(self):
        return [str(o) for o in self.offsets]


    @property
    def atomic_offsets(self):
        offs = []
        for o in self.offsets:
            if isinstance(o, Basic):
                offs += [str(off) for off in o.atoms(Idx)]
                offs += [str(i) for i in list(o.free_symbols) if str(i) not in offs]
        return offs

    @property
    def atomic_loop_offsets(self):
        offs = []
        for o in self.offsets:
            if isinstance(o, Basic):
                offs += [str(off) for off in o.atoms(Idx)]
        return offs



