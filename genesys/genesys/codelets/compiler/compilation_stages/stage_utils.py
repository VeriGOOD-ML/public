from typing import TYPE_CHECKING
from pathlib import Path
import json
if TYPE_CHECKING:
    from codelets.adl import ArchitectureNode
    from codelets.codelet_impl import Codelet

from codelets.compiler.transformations import factors, factors_rand_sort,\
    factors_reversed, level_factors

FACTOR_FN_MAP = {'default': factors, 'random': factors_rand_sort, 'reversed': factors_reversed,
                 'level': level_factors
                 }


def default_tile_heuristic(hag: 'ArchitectureNode', cdlt: 'Codelet', tiling_splits):
    total_accesses = 0
    for l, splits in tiling_splits.items():
        for _, s in splits.items():
            total_accesses += s
    return total_accesses

def update_shape_from_arch(node, shaped_nodes, arch_constraint, dim_index,
                           layout_nhwc=False, force_reshape=False):

    if node.name not in shaped_nodes or force_reshape:
        if node.shape[dim_index] % arch_constraint != 0:
            dim_shape = node.shape[dim_index] + (arch_constraint - (node.shape[dim_index] % arch_constraint))
        else:
            dim_shape = node.shape[dim_index]
        new_shape = list(node.shape)
        new_shape[dim_index] = dim_shape
        if layout_nhwc:
            node.shape = (new_shape[0], new_shape[2], new_shape[3], new_shape[1])
            new_shape = list(node.shape)
        else:
            node.shape = tuple(new_shape)
        shaped_nodes[node.name] = node.shape
    elif shaped_nodes[node.name] != node.shape:
        new_shape = list(shaped_nodes[node.name])
        node.shape = tuple(new_shape)
    else:
        new_shape = list(node.shape)

    return tuple(new_shape)


def find_tiling(cdlt, level, perm_stack):
    if level > list(cdlt.tile_levels.keys())[-1] or level <= 0:
        return level

    prev_level = level - 1
    perms = perm_stack[prev_level]
    assert perms is not None
    valid_splits = None

def store_tile_checkpoint(cdlt, checkpoint_path):
    abs_path = Path(checkpoint_path).absolute()
    if abs_path.exists():
        with open(f'{abs_path}') as f:
            tiling = json.load(f)
    else:
        tiling = {}
    tile_key = f"{cdlt.op_name}{cdlt.instance_id}"
    tiling[tile_key] = cdlt.domain_tiling

    with open(f'{abs_path}', "w") as outfile:
        json.dump(tiling, outfile, indent=4)
