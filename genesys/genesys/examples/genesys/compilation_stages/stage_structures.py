from pytools import memoize_method
from typing import List, Dict, Tuple, Any
from codelets.adl.operation.operand import DataMovement
from itertools import product
from collections import defaultdict
from codelets.adl.flex_param import FlexParam
from codelets.compiler.transformations import factors, factors_rand_sort,\
    factors_reversed
from dataclasses import dataclass, field
import numpy as np


FACTOR_FN_MAP = {'default': factors, 'random': factors_rand_sort, 'reversed': factors_reversed
                 }
@dataclass
class Tiling:
    levels: List[int]
    splits: List[Dict[str, Tuple]]
    shapes: List[Dict[str, Tuple]]

@dataclass
class TilingInfo:
    name: str
    loop_dim_map: Dict[str, str]
    levels: int
    loop_dependencies: List[str]
    accesses: Dict[Any, List]
    level_map: Dict[Tuple[str, str], int] = field(default_factory=dict)
    constraint_fps: Dict[str, FlexParam] = field(default_factory=dict)
    tile_hints: Dict[int, Dict[str, FlexParam]] = field(default_factory=dict)
    valid_tilings: List[Tiling] = field(default_factory=list)
    selected_splits: Dict[int, Dict] = field(default_factory=lambda: defaultdict(dict))
    shapes: Dict[int, Dict] = field(default_factory=lambda: defaultdict(dict))
    accumulated_splits: Dict[str, int] = field(default_factory=dict)
    level_factors: Dict[int, Dict] = field(default_factory=lambda: defaultdict(dict))
    factor_fn_name: str = field(default='default')
    print_debug: bool = field(default=True)
    dims: List[str] = field(default_factory=list)
    loop_idx_mapping: Dict[str, int] = field(default_factory=dict)
    cdlt_params: Dict[str, int] = field(default_factory=dict)
    derived_tilings: Dict[str, FlexParam] = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        for i in range(self.levels):
            self.tile_hints[i] = {}


    def initialize_shapes(self, cdlt):
        if 'fixed_tile_dims' in cdlt.compilation_params:
            fixed_dims = cdlt.compilation_params['fixed_tile_dims']
        else:
            fixed_dims = []
        assert len(fixed_dims) == 0

        # TODO: Need to validate this across all loops
        for l in self.loop_dependencies:
            dim = self.loop_dim_map[l]
            if dim in self.level_factors[0]:
                continue
            self.dims.append(dim)
            # TODO: update this for when dims are fixed
            loop = cdlt.op_map[l]

            # TODO: replace iter_count with dimension
            if l in self.derived_tilings[1]:
                self.level_factors[0][dim] = [1]
            else:
                self.level_factors[0][dim] = FACTOR_FN_MAP[self.factor_fn_name](loop.iter_count, 0)

            self.shapes[0][dim] = loop.iter_count
            self.selected_splits[0][dim] = 1
            self.accumulated_splits[dim] = 1
        assert 0 not in cdlt.domain_tiling

        self.loop_idx_mapping = {
            l: self.dims.index(self.loop_dim_map[l]) for l in self.loop_dependencies
        }

        first_perm = product(*tuple(self.level_factors[0].values()))

        return first_perm

    def add_constraint(self, src: str, dst: str, level: int, constraint_str: str):
        self.constraint_fps[src, dst] = FlexParam(f"{self.name}_{src}_{dst}", ["size"], constraint_str)
        self.level_map[(src, dst)] = level

    def evaluate_constraint(self, key: Tuple[str, str], sizes: Dict[str, int], dtype_bits: int):
        total_size = np.prod(list(sizes.values())) * dtype_bits
        constraint_sat = self.constraint_fps[key].evaluate_fn(total_size)
        return constraint_sat

    def add_tile_hint(self, level: int, loop_name: str, hint_str):
        hint = FlexParam(f"{loop_name}_lvl{level}_hint", ["size", "split", "params"], hint_str)
        self.tile_hints[level][loop_name] = hint

    def add_level_hint(self, level: int, hint_str):
        name = f"LEVEL{level}_hint"
        hint = FlexParam(name, ["sizes", "splits", "params"], hint_str)
        assert name not in self.tile_hints
        self.tile_hints[name] = hint

    def check_tile_hints(self, level, loop_deps, sizes, splits):
        #TODO: Check if this works when there are actual tile hints
        for l, th in self.tile_hints[level].items():
            # idx = self.dims.index(self.loop_dim_map[l])
            idx = self.loop_idx_mapping[l]
            size = sizes[idx]
            split = splits[idx]
            valid = th.evaluate_fn(size, split, self.cdlt_params)
            if not valid:
                return False

        level_name = f"LEVEL{level}_hint"

        if level_name in self.tile_hints:
            sizes = {self.loop_dim_map[l]: sizes[self.loop_idx_mapping[l]] for l in loop_deps}
            splits = {self.loop_dim_map[l]: splits[self.loop_idx_mapping[l]] for l in loop_deps}
            valid = self.tile_hints[level_name].evaluate_fn(sizes, splits, self.cdlt_params)
            if not valid:
                return False
        return True

    def add_valid_tiling(self, valid_tiling):
        pass

    def get_permutation_map(self, perm):
        pmap = {}
        for i, l in enumerate(self.loop_dependencies):
            pmap[l] = perm[self.dims.index(self.loop_dim_map[l])] * self.accumulated_splits[self.loop_dim_map[l]]
        return pmap

    def are_splits_valid(self, cdlt, perm, level, hag):

        for o in cdlt.all_operands:
            pass

    def validate_splits(self, cdlt, perm, level, hag):
        valid_splits = perm

        perm_map = self.get_permutation_map(perm)
        size_map = {}
        access_setters = {}
        checked_accesses = []
        operand_sizes = {}
        loc_sizes = defaultdict(int)
        op_loc_set = []
        for level_access in self.accesses[level]:
            key = (level_access.src_node, level_access.dst_node)
            if (key, level_access.operand_name) in checked_accesses:
                continue
            checked_accesses.append((key, level_access.operand_name))
            size = level_access.get_size_from_splits(cdlt, perm_map)
            for k, v in size.items():
                if k in size_map and v != size_map[k]:
                    return None
                else:
                    size_map[k] = v
                    access_setters[k] = level_access

            operand = cdlt.get_operand(level_access.operand_name)
            dtype_size = operand.dtype.bits()
            if (key[1], level_access.operand_name) not in op_loc_set:
                loc_sizes[key[1]] += dtype_size*np.prod([size[s] for s in operand.shape_symbols])
                op_loc_set.append((key[1], level_access.operand_name))
                operand_sizes[(key[1], level_access.operand_name)] = {s: size[s] for s in operand.shape_symbols}
            constraint_sat = self.evaluate_constraint(key, size, dtype_size)

            if not constraint_sat:
                valid_splits = None
                break

        if valid_splits is not None:
            for node_name, val in loc_sizes.items():
                node = hag.get_subgraph_node(node_name)
                if node.node_type == "storage" and val > node.size:
                    return None


        self.print_debug = False
        return valid_splits

    def validate_derived_splits(self, cdlt, perm, level, hag):
        valid_splits = perm

        perm_map = self.get_permutation_map(perm)
        size_map = {}
        access_setters = {}
        checked_accesses = []
        operand_sizes = {}
        loc_sizes = defaultdict(int)
        op_loc_set = []
        derived_sizes = {k: v['size'].value for k,v in self.derived_tilings[level].items()}

        for level_access in self.accesses[level]:
            key = (level_access.src_node, level_access.dst_node)
            if (key, level_access.operand_name) in checked_accesses:
                continue
            checked_accesses.append((key, level_access.operand_name))
            size = level_access.get_size_from_splits_derived(cdlt, perm_map, derived_sizes)
            for k, v in size.items():
                if k in size_map and v != size_map[k]:
                    return None
                else:
                    size_map[k] = v
                    access_setters[k] = level_access

            operand = cdlt.get_operand(level_access.operand_name)
            dtype_size = operand.dtype.bits()
            if (key[1], level_access.operand_name) not in op_loc_set:
                loc_sizes[key[1]] += dtype_size*np.prod([size[s] for s in operand.shape_symbols])
                op_loc_set.append((key[1], level_access.operand_name))
                operand_sizes[(key[1], level_access.operand_name)] = {s: size[s] for s in operand.shape_symbols}
            constraint_sat = self.evaluate_constraint(key, size, dtype_size)

            if not constraint_sat:
                valid_splits = None
                break

        if valid_splits is not None:
            for node_name, val in loc_sizes.items():
                node = hag.get_subgraph_node(node_name)
                if node.node_type == "storage" and val > node.size:
                    return None


        self.print_debug = False
        return valid_splits

    def update_loop_order(self, cdlt):
        if "LOOP_TILE_ORDER" in cdlt.compilation_params:
            dim_order = cdlt.compilation_params["LOOP_TILE_ORDER"]

            reversed_dom_map = {v: k for k, v in cdlt.domain_loop_map.items()}
            assert len(dim_order) == len(self.loop_dependencies), f"Invalid loop order specification due to missing loop names: " \
                                                                  f"All loops: {[(l, cdlt.loop_param_map[l]) for l in self.loop_dependencies]}\n" \
                                                                  f"Specified loops: {[(reversed_dom_map[d], d) for d in dim_order]}"
            self.loop_dependencies = [reversed_dom_map[d] for d in dim_order]

    def get_tile_permutations(self, level, perm_stack, cdlt):
        if level in cdlt.domain_tiling:
            perms = [tuple(cdlt.domain_tiling[level][ld] for ld in self.dims)]
        else:
            perms = perm_stack[level - 1]

        return perms

    def initialize_factors(self, cdlt, fixed_dims, factor_fn):
        level_factors = defaultdict(dict)
        for l in self.loop_dependencies:
            loop = cdlt.op_map[l]
            if cdlt.domain_loop_map[l] in fixed_dims:
                level_factors[0][loop.op_str] = [1]
            else:
                level_factors[0][loop.op_str] = factor_fn(loop.iter_count, 0)
        return level_factors

    def move_up_tile_level(self, prev_level):
        self.shapes.pop(prev_level)
        self.level_factors.pop(prev_level)
        prev_splits = self.selected_splits.pop(prev_level)
        self.accumulated_splits = {k: v//prev_splits[k] for k, v in self.accumulated_splits.items()}
        return prev_splits

    def run_factor_fn(self, shapes, level):
        return FACTOR_FN_MAP[self.factor_fn_name](shapes, level)

    def move_down_tile_level(self, cdlt, level, valid_splits):
        self.selected_splits[level] = valid_splits.copy()
        self.accumulated_splits = {k: v * self.selected_splits[level][k] for k, v in self.accumulated_splits.items()}
        new_perms = self.get_level_tiling(cdlt, valid_splits, level)
        return new_perms

    def move_down_tile_level_derived(self, cdlt, level, valid_splits):
        self.selected_splits[level] = valid_splits.copy()
        self.accumulated_splits = {k: v * self.selected_splits[level][k] for k, v in self.accumulated_splits.items()}
        new_perms = self.get_level_tiling_derived(cdlt, valid_splits, level)
        return new_perms

    def get_level_tiling(self, cdlt, splits, level):

        for l in self.dims:
            self.shapes[level][l] = self.shapes[level-1][l] // splits[l]
            self.level_factors[level][l] = self.run_factor_fn(self.shapes[level][l], level)

        perms = product(*tuple(self.level_factors[level].values()))
        return perms

    def get_level_tiling_derived(self, cdlt, splits, level):
        derived_tilings = self.derived_tilings[level]
        mappings = {cdlt.loop_param_map[l]: l for l in derived_tilings.keys()}
        for l in self.dims:
            if l in mappings:
                self.shapes[level][l] = derived_tilings[mappings[l]]['size'].value
            else:
                self.shapes[level][l] = self.shapes[level-1][l] // splits[l]
            self.level_factors[level][l] = self.run_factor_fn(self.shapes[level][l], level)

        perms = product(*tuple(self.level_factors[level].values()))
        return perms

    def loop_from_dim(self, dimname):
        for k, v in self.loop_dim_map.items():
            if v == dimname:
                return k
        raise RuntimeError(f"Unable to find loop name for {dimname}")


    def add_derived_tiling(self, loop, level, split_eq, size_eq):
        if "loop" not in loop:
            loop = self.loop_from_dim(loop)
        split_hint = FlexParam(f"{loop}_split{level}", ["sizes", "splits", "params"], split_eq)
        size_hint = FlexParam(f"{loop}_size{level}", ["sizes", "splits", "params"], size_eq)

        self.derived_tilings[level][loop] = {"split": split_hint, "size": size_hint}


    def evaluate_derived_param(self, level, loop_deps, sizes, splits):
        if level not in self.derived_tilings:
            return sizes, splits
        mut_sizes = list(sizes)
        mut_splits = list(splits)
        map_sizes = {self.loop_dim_map[l]: sizes[self.loop_idx_mapping[l]] for l in loop_deps}
        map_splits = {self.loop_dim_map[l]: splits[self.loop_idx_mapping[l]] for l in loop_deps}
        for loop, l in self.derived_tilings[level].items():
            mut_splits[self.loop_idx_mapping[loop]] = l['split'].evaluate_fn(map_sizes, map_splits, self.cdlt_params, force_evaluate=True)
            mut_sizes[self.loop_idx_mapping[loop]] = l['size'].evaluate_fn(map_sizes, map_splits, self.cdlt_params, force_evaluate=True)

        return tuple(mut_sizes), tuple(mut_splits)

    def finalize_derived_param(self, level, loop, loop_deps, splits, sizes):

        map_sizes = {self.loop_dim_map[l]: sizes[l] for l in loop_deps}
        map_splits = {self.loop_dim_map[l]: splits[l] for l in loop_deps}
        sz = self.derived_tilings[level][loop]['size'].evaluate_fn(map_sizes, map_splits, self.cdlt_params, force_evaluate=True)
        split = self.derived_tilings[level][loop]['split'].evaluate_fn(map_sizes, map_splits, self.cdlt_params, force_evaluate=True)

        return {"size": sz, "split": split}

