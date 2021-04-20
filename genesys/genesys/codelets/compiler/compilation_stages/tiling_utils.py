from typing import TYPE_CHECKING, List
from collections import defaultdict, deque
from itertools import product
from pytools import memoize

if TYPE_CHECKING:
    from codelets.adl import ArchitectureNode
    from codelets.codelet_impl import Codelet


from . import TilingInfo
from codelets.compiler.transformations import factors, factors_rand_sort,\
    factors_reversed, level_factors

FACTOR_FN_MAP = {'default': factors, 'random': factors_rand_sort, 'reversed': factors_reversed,
                 'level': level_factors
                 }

@memoize
def get_sizes_from_splits(loops, shapes, splits):
    out_shapes = []

    for i, l in enumerate(loops):
        out_shapes.append(shapes[i] // splits[i])

    return tuple(out_shapes)

# TODO: THis needs to return a list of functions with the same function signature
def get_tile_constraints(cdlt: 'Codelet', hag: 'ArchitectureNode', tile_info: TilingInfo):
    path_constraints = {}

    for o in cdlt.operands:
        for access in o.data_moves:
            if (access.src_node, access.dst_node) in path_constraints or access.src_node == access.dst_node:
                continue

            src_node = hag.get_subgraph_node(access.src_node)
            dst_node = hag.get_subgraph_node(access.dst_node)
            edge = hag.get_subgraph_edge(access.src_node, access.dst_node)
            if dst_node.node_type == 'compute':
                constraint = f"size == {edge.bandwidth}"
                assert src_node.node_type == 'storage'
                # TODO: Need to add something which adds padding function here and uses a function constraint
            elif dst_node.node_type == 'storage':
                if src_node.node_type == 'compute':
                    constraint = f"size == {edge.bandwidth}"
                else:
                    assert src_node.node_type == 'storage'
                    constraint = f"size <= {dst_node.size} and size >= 0"
            else:
                raise TypeError(f"Unable to handle architecture node type {type(dst_node)}")
            level = cdlt.get_tile_level(access.dst_node)
            tile_info.add_constraint(access.src_node, access.dst_node, level, constraint)

    for loop_name, dim_name in cdlt.domain_loop_map.items():
        for level in cdlt.tile_levels.keys():
            tile_hint_key = f"{dim_name}_hint{level}"
            if tile_hint_key in cdlt.compilation_params:
                tile_info.add_tile_hint(level, loop_name, cdlt.compilation_params[tile_hint_key])

    for level in cdlt.tile_levels.keys():
        level_hint_key = f"LEVEL{level}_hint"
        if level_hint_key in cdlt.compilation_params:
            tile_info.add_level_hint(level, cdlt.compilation_params[level_hint_key])

    return tile_info

def get_level_tiling(cdlt, loop_dependencies, shapes, splits, factor_fn, level):
    out_shapes = {}
    out_factors = {}

    if 'fixed_tile_dims' in cdlt.compilation_params:
        fixed_dims = cdlt.compilation_params['fixed_tile_dims']
    else:
        fixed_dims = []


    for l in loop_dependencies:
        out_shapes[l] = shapes[l] // splits[l]

        if cdlt.domain_loop_map[l] in fixed_dims:
            out_factors[l] = [1]
        else:
            out_factors[l] = factor_fn(out_shapes[l], level)

    perms = product(*tuple(out_factors.values()))

    return out_shapes, out_factors, perms


def find_valid_splits(cdlt, p, lvl,
                      accumulated_splits,
                      loop_dependencies,
                      level_accesses,
                      tile_constraints):
    valid_splits = p

    perm_map = {l: p[i]*accumulated_splits[l] for i, l in enumerate(loop_dependencies)}
    size_map = {}


    for level_access in level_accesses[lvl]:

        size = level_access.get_size_from_splits(cdlt, perm_map)
        key = (level_access.src_node, level_access.dst_node)

        for k, v in size.items():
            if k in size_map and v != size_map[k]:
                raise RuntimeError(f"Size is not equal to collected sizes for access:\n"
                                   f"Size from splits: {size}\n"
                                   f"Size map: {size_map}\n"
                                   f"Level: {lvl}\n"
                                   f"Key: {key}\n")

            else:
                size_map[k] = v


        dtype_size = cdlt.get_operand(level_access.operand_name).dtype.bits()
        constraint_sat = tile_constraints.evaluate_constraint(key, size, dtype_size)

        if not constraint_sat:
            valid_splits = None
            break
    return valid_splits

def set_codelet_tiling(cdlt: 'Codelet', hag: 'ArchitectureNode', factor_fn_name):
    # TODO: Try to look ahead and see if all paths lead to node, in which case
    # we can add additional constraints to the first level
    level_accesses = defaultdict(list)
    loop_dependencies = []
    # Collect accesses and loop dependencies
    for o in cdlt.operands:
        for i, access in enumerate(o.data_moves):
            if access.src_node != access.dst_node:
                level_accesses[cdlt.get_tile_level(access.dst_node)].append(access)

        loop_dependencies += [dp for dp in o.dependencies if dp not in loop_dependencies and "loop" in dp]


    tile_info = TilingInfo(f"{cdlt.op_name}{cdlt.instance_id}_tile_info",
                           cdlt.domain_loop_map,
                           len(list(cdlt.tile_levels.keys())),
                           loop_dependencies,
                           level_accesses, factor_fn_name=factor_fn_name)
    tile_info.update_loop_order(cdlt)
    tile_info = get_tile_constraints(cdlt, hag, tile_info)
    first_perm = tile_info.initialize_shapes(cdlt)
    perm_stack = deque()

    perm_stack.append(first_perm)
    level = 1
    level_counter = defaultdict(int)
    loop_deps_fixed = tuple(tile_info.loop_dependencies)
    parent_perms = deque()
    prev_perm = None
    parent_perms.append(prev_perm)

    while level < tile_info.levels and level > 0:

        prev_level = level - 1
        perms = tile_info.get_tile_permutations(level, perm_stack, cdlt)
        assert perms is not None
        valid_splits = None
        fixed_shapes = tuple([tile_info.shapes[prev_level][l] for l in tile_info.loop_dependencies])

        for p in perms:
            level_counter[level] += 1
            perm_shapes = get_sizes_from_splits(loop_deps_fixed, fixed_shapes, p)
            passes_hint = tile_info.check_tile_hints(level, loop_deps_fixed, perm_shapes, p)
            if not passes_hint:
                continue
            valid_splits = tile_info.validate_splits(cdlt, p, level)

            if valid_splits:
                prev_perm = p
                valid_splits = {list(tile_info.level_factors[level - 1].keys())[i]: v for i, v in enumerate(valid_splits)}
                break

        if not valid_splits:
            prev_perm = parent_perms.pop()
            perm_stack.pop()
            prev_splits = tile_info.move_up_tile_level(prev_level)
            level -= 1
        else:
            parent_perms.append(prev_perm)
            new_perms = tile_info.move_down_tile_level(cdlt, level, valid_splits)
            perm_stack.append(new_perms)
            level += 1

    if level == 0:
        raise RuntimeError(f"Unable to find adequate tiling for Codelet:"
                       f"Codelet Dimensions: {cdlt.operand_dim_mapping()}\n"
                           f"Times per level: {level_counter}\n"
                       f"Op: {cdlt.op_name}{cdlt.instance_id}\n"
                       f"constraints:{[(k, t.fn_body_str) for k, t in tile_info.constraint_fps.items()]}\n")
    # Lastly, update operands
    for o in cdlt.operands:

        for idx, a in enumerate(o.data_moves):
            if all(a in [None, 0] for a in list(a.offset_map.values())):
                assert idx > 0
                a.reinit_offset_map(o.data_moves[idx - 1].offset_map.copy())

            if len(a.shape_map) == 0:
                a.set_size_from_splits(cdlt, tile_info.selected_splits)

            a.set_offset_map(cdlt, tile_info.shapes)

    # TODO: Store all information int he codelet
    cdlt._domain_tiling = tile_info.selected_splits
    cdlt._domain_loop_map = tile_info.shapes

    return cdlt
