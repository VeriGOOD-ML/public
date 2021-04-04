from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union, ClassVar
from collections import namedtuple
from functools import partial
from pytools import memoize
from collections import defaultdict
import numpy as np
from . import pairwise
import polymath as pm
from numbers import Integral
from copy import deepcopy
from sympy import Basic, Idx, symbols, Integer
from codelets.adl import util
from dataclasses import dataclass, field

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

@dataclass(frozen=True)
class Datatype:
    type: str
    bitwidth: int

    def __str__(self):
        return f"{self.type}{self.bitwidth}"

    def to_json(self):
        blob = {}
        blob['type'] = self.type
        blob['bitwidth'] = self.bitwidth
        return blob

    @staticmethod
    def from_json(dt_obj: Dict):
        return Datatype(type=dt_obj['type'], bitwidth=dt_obj['bitwidth'])

    @staticmethod
    def from_str(dt_str: str):
        idx = 0

        while not dt_str[idx].isdigit() and idx < len(dt_str):
            idx += 1
        type_part = dt_str[:idx].upper()
        bit_part = int(dt_str[idx:])
        return Datatype(type=type_part, bitwidth=bit_part)

    def bytes(self):
        return self.bitwidth // 8

    def bits(self):
        return self.bitwidth

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
                          deepcopy(self.evaluated_domain_offsets)
                          )
    def substitute_offset_symbols(self, cdlt, dep_map, replacements):
        new_offset_map = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                for idx, i in enumerate(indices):
                    if str(i) in replacements and str(i) in dep_map:
                        replacement_op = cdlt.op_map[dep_map[str(i)]]
                        assert replacement_op.op_type == "loop"
                        o.subs(i, replacement_op.get_symbol())
                new_offset_map[name] = o
            else:
                new_offset_map[name] = 0
        self.offset_map = new_offset_map
        self.set_size_from_splits(cdlt, cdlt.domain_tiling)
        self.set_offset_map(cdlt, cdlt.domain_loop_map, dep_map)

    def get_size_from_splits(self, cdlt, splits):
        sizes = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                for idx, i in enumerate(indices):
                    assert cdlt.op_map[str(i)].end % splits[str(i)] == 0
                    max_vals[str(i)] = cdlt.op_map[str(i)].end // splits[str(i)] - 1
                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})

                size = self.resolve_offset(o, max_vals) + 1
                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size

        return sizes


    def get_size_from_loops(self, cdlt, loops):
        sizes = {}

        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                for idx, i in enumerate(indices):
                    assert str(i) in loops
                    max_vals[str(i)] = loops[str(i)] - 1

                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
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
                max_vals = {str(i): cdlt.op_map[str(i)].end - 1 for i in indices}
                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
            else:
                size = o
            self.evaluated_offsets[name] = size

    def resolve_offset(self, expr: Basic, values: Dict[str, int]):
        for f_sym in list(expr.free_symbols):
            if str(f_sym) in values:
                expr = expr.subs(f_sym, values[str(f_sym)])
        if not isinstance(expr, (Integer, Integral)):
            raise TypeError(f"Unable to compute domain domain_offsets because offset is not an integer:"
                            f"Offset: {expr}\tType: {type(expr)}")
        return int(expr)




@dataclass
class OperandTemplate:
    name: str
    supported_dtypes: Union[List[Datatype]]
    shape_list: List[str]
    shape_symbols: Dict = field(default_factory=dict)
    tiling: Dict[str, List[int]] = field(default_factory=dict)
    data_path: List[str] = field(default_factory=list)
    dtype: Datatype = field(default=None)
    node_name: str = field(default=None)
    evaluated_tiling: List[Tuple[str, List]] = field(default_factory=list, init=False)
    dependencies: List[str] = field(default_factory=list)
    data_moves: List[DataMovement] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    current_codelet: ClassVar = field(default=None)
    compute_pad_dim: int = field(default=-1)
    static_padding: Dict[str, int] = field(default_factory=dict)
    dynamic_padding: Dict[str, int] = field(default_factory=dict)
    dim_order: List[int] = field(default=None)
    operand_type: str = field(default='variable')

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
                dm.offset_map[dim] = val

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

    # 'up' -> dram -> compute unit
    # 'down' -> compute unit -> dram
    def get_offset(self, cdlt, level, loop_id, movement_type='up', zero_not_found=True):
        if movement_type == 'up':
            prev_level = level - 1
        else:
            prev_level = level + 1
        assert prev_level >= 0
        target_movement = None
        for dm in self.data_moves:

            if cdlt.get_tile_level(dm.dst_node) == level:
                # assert cdlt.get_tile_level(dm.src_node) == prev_level, f"{self.name}: {dm.src_node}({cdlt.get_tile_level(dm.src_node)})" \
                #                                                        f"-> {dm.dst_node}({cdlt.get_tile_level(dm.dst_node)})\n" \
                #                                                        f"Levels: {cdlt.tile_levels}"
                target_movement = dm
                break
            elif cdlt.get_tile_level(dm.src_node) == level:
                assert cdlt.get_tile_level(dm.dst_node) == prev_level, f"{self.name}: {dm.src_node} -> {dm.dst_node}"
                target_movement = dm
                break

        if target_movement is None:
            raise RuntimeError(f"Could not find data movement for level {level} in operand "
                               f"{self.name}")

        offset_val = None
        for o in target_movement.domain_offsets():
            if o.loop_id == loop_id:
                offset_val = o
                break

        if offset_val is None:
            if zero_not_found:
                return 0
            else:
                raise RuntimeError(f"Could not find offset movement {level} from "
                                   f"{target_movement.src_node}->{target_movement.dst_node} "
                                   f"in operand {self.name}")
        return offset_val.stride

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
                self.required_params.append(idx)
            elif isinstance(idx, int):
                off = idx
            else:
                try:
                    off = idx.param_symbols[idx.op_str]
                except TypeError:
                    off = idx
                self.dependencies.append(idx.op_str)
            offsets.append(off)
        assert len(offsets) == len(self.shape_list)
        return IndexedOperandTemplate(self, offsets)

    def get_access_offsets(self, offsets):
        if len(offsets) == 0 and len(self.data_moves) > 0:
            a_offsets = {self.shape_list[i]: list(self.data_moves[-1].offset_map.values())[i].copy() for i in range(len(self.shape_list))}
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

                    if self.data_moves[-1].unset_offsets:
                        self.data_moves[-1].offset_map = self.get_access_offsets(offsets)


            dm_offsets = self.get_access_offsets(offsets)
            shape = self.get_shape_map(sizes[i])
            movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, dm_offsets)
            self.data_moves.append(movement)

        return self

    def add_compute_access(self, target, op_name, operand_type, offsets=None):

        assert operand_type in ["source", "dest"]
        offsets = offsets or []

        offsets = self.get_access_offsets(offsets)
        self.add_dependency(op_name)

        shape = self.get_shape_map([0] * len(self.shape_list))
        if operand_type == "source":
            src = self.current_location
            dst = target
        else:
            src = target
            dst = None

        movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, offsets)
        self.data_moves.append(movement)

        if self.current_location != target:
            self.data_path.append(target)

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

    def get_ld_storage_location(self, cdlt, level):
        prev_level = level - 1
        assert prev_level >= 0
        for dm in self.data_moves:
            if cdlt.get_tile_level(dm.src_node) == prev_level:
                assert cdlt.get_tile_level(dm.dst_node) == level
                return dm.dst_node
            elif cdlt.get_tile_level(dm.dst_node) == prev_level:
                assert cdlt.get_tile_level(dm.src_node) == level
                return dm.src_node
        raise RuntimeError(f"Unable to find storage node for level {level} in operand"
                           f" {self.name}.")


    def set_start_location(self, location: str):
        if len(self.data_path) > 0:
            raise RuntimeError(f"Unable to set default location for operand {self.name} because "
                               f"path has already been initialized: {self.data_path}")
        self.data_path.append(location)

    def evaluate_operand(self, node: pm.Node, hag, cdlt):

        initial_size = [self.shape_symbols[symb] for symb in self.shape_list]
        level_shapes = {}
        assert len(hag.node_levels[0]) == 1
        level_shapes[0] = initial_size

        for i, access in enumerate(self.data_moves):
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
        op_temp = OperandTemplate(name=self.name,
                                  supported_dtypes=self.supported_dtypes,
                                  shape_list=self.shape_list,
                                  shape_symbols=self.shape_symbols.copy(),
                                  dependencies=self.dependencies.copy(),
                                  tiling=deepcopy(self.tiling),
                                  data_path=self.data_path.copy(),
                                  dtype=self.dtype,
                                  node_name=self.node_name,
                                  data_moves=deepcopy(self.data_moves),
                                  operand_type=self.operand_type)
        op_temp.evaluated_tiling = deepcopy(self.evaluated_tiling)
        return op_temp

    def emit(self, output_type):

        if output_type == "json":

            blob = {"name": self.name,
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
    operand_template: OperandTemplate
    offsets: List[Union[int, str, Basic]]

    @property
    def data_moves(self):
        return self.operand_template.data_moves

    def add_transfer_access(self, path, op_name, sizes):
        return self.operand_template.add_transfer_access(path, op_name, sizes, self.offsets)

    def add_compute_access(self, target, op_name, operand_type):
        return self.operand_template.add_compute_access(target, op_name, operand_type, self.offsets)





