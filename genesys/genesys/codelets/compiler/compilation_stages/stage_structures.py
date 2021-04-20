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

    def __post_init__(self):
        for i in range(self.levels):
            self.tile_hints[i] = {}

    def initialize_shapes(self, cdlt):
        if 'fixed_tile_dims' in cdlt.compilation_params:
            fixed_dims = cdlt.compilation_params['fixed_tile_dims']
        else:
            fixed_dims = []

        for l in self.loop_dependencies:
            loop = cdlt.op_map[l]
            if cdlt.domain_loop_map[l] in fixed_dims:
                self.level_factors[0][loop.op_str] = [1]
            else:
                self.level_factors[0][loop.op_str] = FACTOR_FN_MAP[self.factor_fn_name](loop.iter_count, 0)

            self.shapes[0][loop.op_str] = loop.iter_count
            self.selected_splits[0][loop.op_str] = 1
            self.accumulated_splits[loop.op_str] = 1

        if 0 in cdlt.domain_tiling:
            first_perm = [tuple(cdlt.domain_tiling[0][ld] for ld in self.loop_dependencies)]
        else:
            first_perm = product(*tuple(self.level_factors[0].values()))

        return first_perm

    def add_constraint(self, src: str, dst: str, level: int, constraint_str: str):
        self.constraint_fps[src,dst] = FlexParam(f"{self.name}_{src}_{dst}", ["size"], constraint_str)
        self.level_map[(src, dst)] = level

    def evaluate_constraint(self, key: Tuple[str, str], sizes: Dict[str, int], dtype_bits: int):
        total_size = np.prod(list(sizes.values())) * dtype_bits
        constraint_sat = self.constraint_fps[key].evaluate_fn(total_size)
        return constraint_sat

    def add_tile_hint(self, level: int, loop_name: str, hint_str):
        hint = FlexParam(f"{loop_name}_lvl{level}_hint", ["size", "split"], hint_str)
        self.tile_hints[level][loop_name] = hint

    def add_level_hint(self, level: int, hint_str):
        name = f"LEVEL{level}_hint"
        hint = FlexParam(name, ["sizes", "splits"], hint_str)
        assert name not in self.tile_hints
        self.tile_hints[name] = hint

    def check_tile_hints(self, level, loop_deps, sizes, splits):

        for l, th in self.tile_hints[level].items():
            idx = loop_deps.index(l)
            size = sizes[idx]
            split = splits[idx]
            valid = th.evaluate_fn(size, split)
            if not valid:
                return False

        level_name = f"LEVEL{level}_hint"
        if level_name in self.tile_hints:
            sizes = {self.loop_dim_map[l]: sizes[i] for i, l in enumerate(loop_deps)}
            splits = {self.loop_dim_map[l]: splits[i] for i, l in enumerate(loop_deps)}
            valid = self.tile_hints[level_name].evaluate_fn(sizes, splits)
            if not valid:
                return False
        return True

    def add_valid_tiling(self, valid_tiling):
        pass

    def get_permutation_map(self, perm):
        return {l: perm[i] * self.accumulated_splits[l] for i, l in enumerate(self.loop_dependencies)}

    def validate_splits(self, cdlt, perm, level):
        valid_splits = perm
        perm_map = self.get_permutation_map(perm)
        size_map = {}

        for level_access in self.accesses[level]:

            size = level_access.get_size_from_splits(cdlt, perm_map)
            key = (level_access.src_node, level_access.dst_node)

            for k, v in size.items():
                if k in size_map and v != size_map[k]:
                    raise RuntimeError(f"Size is not equal to collected sizes for access:\n"
                                       f"Size from splits: {size}\n"
                                       f"Size map: {size_map}\n"
                                       f"Level: {level}\n"
                                       f"Key: {key}\n")

                else:
                    size_map[k] = v

            dtype_size = cdlt.get_operand(level_access.operand_name).dtype.bits()
            constraint_sat = self.evaluate_constraint(key, size, dtype_size)

            if not constraint_sat:
                valid_splits = None
                break

        return valid_splits

    def update_loop_order(self, cdlt):
        if "LOOP_TILE_ORDER" in cdlt.compilation_params:
            dim_order = cdlt.compilation_params["LOOP_TILE_ORDER"]
            reversed_dom_map = {v: k for k, v in cdlt.domain_loop_map.items()}
            self.loop_dependencies = [reversed_dom_map[d] for d in dim_order]

    def get_tile_permutations(self, level, perm_stack, cdlt):
        if level in cdlt.domain_tiling:
            perms = [tuple(cdlt.domain_tiling[level][ld] for ld in self.loop_dependencies)]
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

    def get_level_tiling(self, cdlt, splits, level):

        if 'fixed_tile_dims' in cdlt.compilation_params:
            fixed_dims = cdlt.compilation_params['fixed_tile_dims']
        else:
            fixed_dims = []

        for l in self.loop_dependencies:
            self.shapes[level][l] = self.shapes[level-1][l] // splits[l]

            if cdlt.domain_loop_map[l] in fixed_dims:
                self.level_factors[level][l] = [1]
            else:
                self.level_factors[level][l] = self.run_factor_fn(self.shapes[level][l], level)

        perms = product(*tuple(self.level_factors[level].values()))
        return perms