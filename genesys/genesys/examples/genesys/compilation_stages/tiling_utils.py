from typing import TYPE_CHECKING, List
from collections import defaultdict, deque
from itertools import product, tee
from pytools import memoize
from sympy import Basic, Idx, symbols, Integer, lambdify


if TYPE_CHECKING:
    from codelets.adl import ArchitectureNode
    from codelets.codelet_impl import Codelet

from .stage_structures import TilingInfo
from . import CUSTOM_TILE_OPS
from codelets.compiler.transformations import factors, factors_rand_sort, \
    factors_reversed, level_factors



FACTOR_FN_MAP = {'default': factors, 'random': factors_rand_sort, 'reversed': factors_reversed,
                 'level': level_factors
                 }

# @memoize
def get_sizes_from_splits(loops, shapes, splits):
    out_shapes = []

    for i, l in enumerate(loops):
        out_shapes.append(shapes[i] // splits[i])

    return tuple(out_shapes)


# TODO: THis needs to return a list of functions with the same function signature
def get_tile_constraints(cdlt: 'Codelet', hag: 'ArchitectureNode', tile_info: TilingInfo):
    path_constraints = {}

    # for n in hag

    for o in cdlt.all_operands:
        for access in o.data_moves:
            if (access.src_node, access.dst_node) in path_constraints or access.src_node == access.dst_node:
                continue
            if access.src_node is None or access.dst_node is None:
                raise RuntimeError(f"Source node for access in operand {access.operand_name} is not set:\n"
                                   f"Source: {access.src_node}\n"
                                   f"Dst: {access.dst_node}\n")
            src_node = hag.get_subgraph_node(access.src_node)
            dst_node = hag.get_subgraph_node(access.dst_node)
            edge = hag.get_subgraph_edge(access.src_node, access.dst_node)
            if dst_node.node_type == 'compute':
                constraint = f"size == {edge.bandwidth}"
                assert src_node.node_type == 'storage'
                # TODO: Need to add something which adds padding function here and uses a function constraint
            elif dst_node.node_type == 'storage':
                if src_node.node_type == 'compute':
                    constraint = f"size <= {dst_node.capacity}"
                else:
                    assert src_node.node_type == 'storage'
                    constraint = f"size <= {dst_node.capacity} and size >= 0"
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

    perm_map = {l: p[i] * accumulated_splits[l] for i, l in enumerate(loop_dependencies)}
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

def get_tile_info(cdlt, hag, factor_fn_name) -> TilingInfo:
    level_accesses = defaultdict(list)
    loop_dependencies = []

    # Collect accesses and loop dependencies
    for o in cdlt.all_operands:
        for i, access in enumerate(o.data_moves):
            if access.src_node != access.dst_node:
                level_accesses[cdlt.get_tile_level(access.dst_node)].append(access)

        loop_dependencies += [dp for dp in list(set(o.dependencies)) if dp not in loop_dependencies and "loop" in dp]

    op_params = {}
    loop_order = [v for v in cdlt.loop_param_map.values()]
    for k, v in cdlt.required_params.items():
        if k not in loop_order:
            op_params[k] = v.value

    tile_info = TilingInfo(f"{cdlt.op_name}{cdlt.instance_id}_tile_info",
                   cdlt.domain_loop_map,
                   len(list(cdlt.tile_levels.keys())),
                   loop_dependencies,
                   level_accesses,
                   factor_fn_name=factor_fn_name,
                   cdlt_params=op_params
                   )
    tile_info.update_loop_order(cdlt)
    tile_info = get_tile_constraints(cdlt, hag, tile_info)
    return tile_info

def set_codelet_tiling(cdlt: 'Codelet',
                       hag: 'ArchitectureNode',
                       factor_fn_name,
                       stopping_condition,
                       selection_metric,
                       heuristic_fn):

    if stopping_condition is None:
        RuntimeError("Stopping condition for codelet tiling is not specified")
    if selection_metric is None:
        RuntimeError("Selection metric for codelet tiling is not specified")
    if cdlt.op_name in CUSTOM_TILE_OPS:
        return set_dw_conv_tiling(cdlt, hag, factor_fn_name, stopping_condition, selection_metric, heuristic_fn)
    # TODO: Try to look ahead and see if all paths lead to node, in which case
    # we can add additional constraints to the first level
    tile_info = get_tile_info(cdlt, hag, factor_fn_name)


    # TODO: IF loop ordering is specified, need to figure out how to handle multiple loop blocks over the same
    # dimension
    first_perm = tile_info.initialize_shapes(cdlt)

    perm_stack = deque()

    perm_stack.append(first_perm)
    level = 1
    level_counter = defaultdict(int)
    loop_deps_fixed = tuple(tile_info.loop_dependencies)
    loop_dims_fixed = tuple(tile_info.dims)
    parent_perms = deque()
    prev_perm = None
    parent_perms.append(prev_perm)
    invalid_permutations = {}

    eval_params = {}

    for k, v in cdlt.required_params.items():
        if k not in list(cdlt.loop_param_map.values()):
            assert v.value is not None
            eval_params[k] = v.value

    while tile_info.levels > level > 0:
        prev_level = level - 1
        perms = tile_info.get_tile_permutations(level, perm_stack, cdlt)
        perms, perms_copy = tee(perms)
        assert perms is not None
        fixed_shapes = tuple([tile_info.shapes[prev_level][l] for l in tile_info.dims])
        search_space = {}
        stop_search = False
        last_valid_permutation = None
        selected_permutation = None
        for p in perms:
            if p in invalid_permutations:
                continue
            level_counter[level] += 1

            perm_shapes = get_sizes_from_splits(loop_dims_fixed, fixed_shapes, p)
            passes_hint = tile_info.check_tile_hints(level, loop_deps_fixed, perm_shapes, p)
            if not passes_hint:
                continue
            valid_splits = tile_info.validate_splits(cdlt, p, level, hag)
            if valid_splits is None:
                continue
            last_valid_permutation = p
            search_space[p] = heuristic_fn(p)
            stop_search = stopping_condition(search_space)
            if stop_search:
                selected_permutation = selection_metric(search_space, p)
                break
        # Explored all permutations
        if not stop_search:
            selected_permutation = selection_metric(search_space, last_valid_permutation)
            # Need to reset permutation generator to restart search if return to this level
            perm_stack[level-1] = perms_copy
        # If no split available, move up a level and restart search.
        # Else store current permutation and move down a level.
        if selected_permutation is None:
            prev_perm = parent_perms.pop()
            # Add prev permutation to list of permutations which are known to be invalid
            invalid_permutations[prev_perm] = 1
            perm_stack.pop()
            prev_splits = tile_info.move_up_tile_level(prev_level)
            level -= 1
        else:
            selected_splits = {list(tile_info.level_factors[level - 1].keys())[i]: v for i, v in
                               enumerate(selected_permutation)}
            parent_perms.append(selected_permutation)
            new_perms = tile_info.move_down_tile_level(cdlt, level, selected_splits)
            perm_stack.append(new_perms)
            level += 1

    if level == 0:
        other_hint_str = []
        for k, lh in tile_info.tile_hints.items():
            if isinstance(lh, dict):
                for key, hint in lh.items():
                    other_hint_str.append(f"Level {k}, {cdlt.loop_param_map[key]}: {hint.fn_body_str}")
        other_hint_str = "\n".join(other_hint_str)
        hint_str = "\n".join([f"{k} : {v.fn_body_str}" for k, v in tile_info.tile_hints.items() if hasattr(v, 'fn_body_str')])
        raise RuntimeError(f"Unable to find adequate tiling for Codelet {cdlt.cdlt_uid}:"
                           f"Dimensions: {cdlt.operand_dim_mapping()}\n"
                           f"Operands: {[f'{o.name}: {o.shape}' for o in cdlt.operands]}\n"
                           f"Times per level: {level_counter}\n"
                           f"Op: {cdlt.op_name}{cdlt.instance_id}\n"
                           f"Constraints:{[(k, t.fn_body_str) for k, t in tile_info.constraint_fps.items()]}\n\n"
                           f"Level Hints: {hint_str}\n"
                           f"Loop hints: {other_hint_str}"
                           )

    # Lastly, update operands
    for o in cdlt.all_operands:
        for idx, a in enumerate(o.data_moves):
            if all(a in [None, 0] for a in list(a.offset_map.values())):
                assert idx > 0
                a.reinit_offset_map(o.data_moves[idx - 1].offset_map.copy())

            if len(a.shape_map) == 0:
                a.set_size_from_splits(cdlt, tile_info.selected_splits)

            a.set_offset_map(cdlt, tile_info.shapes)


    ## Testing temporary

    # TODO: Store all information in the codelet
    cdlt._domain_tiling = {}
    cdlt._domain_loop_map = {}
    for l, dim_splits in tile_info.selected_splits.items():
        cdlt._domain_tiling[l] = {}
        cdlt._domain_loop_map[l] = {}
        for ld in tile_info.loop_dependencies:
            cdlt._domain_tiling[l][ld] = dim_splits[tile_info.loop_dim_map[ld]]
            cdlt._domain_loop_map[l][ld] = tile_info.shapes[l][tile_info.loop_dim_map[ld]]
    return cdlt



def set_dw_conv_tiling(cdlt: 'Codelet',
                       hag: 'ArchitectureNode',
                       factor_fn_name,
                       stopping_condition,
                       selection_metric,
                       heuristic_fn):

    if stopping_condition is None:
        RuntimeError("Stopping condition for codelet tiling is not specified")
    if selection_metric is None:
        RuntimeError("Selection metric for codelet tiling is not specified")
    # TODO: Try to look ahead and see if all paths lead to node, in which case
    # we can add additional constraints to the first level


    tile_info = get_tile_info(cdlt, hag, factor_fn_name)

    # TODO: IF loop ordering is specified, need to figure out how to handle multiple loop blocks over the same
    # dimension
    tile_info.add_derived_tiling("OH", 1, "splits['OH1']", "(sizes['OH1'] - 1)*params['s2'] + sizes['KH1']")
    tile_info.add_derived_tiling("OW", 1, "splits['OW1']", "(sizes['OW1'] - 1)*params['s2'] + sizes['KW1']")
    first_perm = tile_info.initialize_shapes(cdlt)

    perm_stack = deque()

    perm_stack.append(first_perm)
    level = 1
    level_counter = defaultdict(int)
    loop_deps_fixed = tuple(tile_info.loop_dependencies)
    loop_dims_fixed = tuple(tile_info.dims)
    parent_perms = deque()
    prev_perm = None
    parent_perms.append(prev_perm)
    invalid_permutations = {}

    eval_params = {}

    for k, v in cdlt.required_params.items():
        if k not in list(cdlt.loop_param_map.values()):
            assert v.value is not None
            eval_params[k] = v.value

    while tile_info.levels > level > 0:
        prev_level = level - 1
        perms = tile_info.get_tile_permutations(level, perm_stack, cdlt)
        perms, perms_copy = tee(perms)
        assert perms is not None
        fixed_shapes = tuple([tile_info.shapes[prev_level][l] for l in tile_info.dims])
        search_space = {}
        stop_search = False
        last_valid_permutation = None
        selected_permutation = None
        for p in perms:
            if p in invalid_permutations:
                continue
            level_counter[level] += 1
            perm_shapes = get_sizes_from_splits(loop_dims_fixed, fixed_shapes, p)
            perm_shapes, p = tile_info.evaluate_derived_param(level, loop_deps_fixed, perm_shapes, p)
            passes_hint = tile_info.check_tile_hints(level, loop_deps_fixed, perm_shapes, p)
            if not passes_hint:
                continue
            valid_splits = tile_info.validate_derived_splits(cdlt, p, level, hag)
            if valid_splits is None:
                continue
            last_valid_permutation = p
            search_space[p] = heuristic_fn(p)
            stop_search = stopping_condition(search_space)
            if stop_search:
                selected_permutation = selection_metric(search_space, p)
                break
        # Explored all permutations
        if not stop_search:
            selected_permutation = selection_metric(search_space, last_valid_permutation)
            # Need to reset permutation generator to restart search if return to this level
            perm_stack[level-1] = perms_copy
        # If no split available, move up a level and restart search.
        # Else store current permutation and move down a level.
        if selected_permutation is None:
            prev_perm = parent_perms.pop()
            # Add prev permutation to list of permutations which are known to be invalid
            invalid_permutations[prev_perm] = 1
            perm_stack.pop()
            prev_splits = tile_info.move_up_tile_level(prev_level)
            level -= 1
        else:

            selected_splits = {list(tile_info.level_factors[level - 1].keys())[i]: v for i, v in
                               enumerate(selected_permutation)}
            parent_perms.append(selected_permutation)
            new_perms = tile_info.move_down_tile_level(cdlt, level, selected_splits)
            perm_stack.append(new_perms)
            level += 1

    if level == 0:
        other_hint_str = []
        for k, lh in tile_info.tile_hints.items():
            if isinstance(lh, dict):
                for key, hint in lh.items():
                    other_hint_str.append(f"Level {k}, {cdlt.loop_param_map[key]}: {hint.fn_body_str}")
        other_hint_str = "\n".join(other_hint_str)
        hint_str = "\n".join([f"{k} : {v.fn_body_str}" for k, v in tile_info.tile_hints.items() if hasattr(v, 'fn_body_str')])
        raise RuntimeError(f"Unable to find adequate tiling for Codelet {cdlt.cdlt_uid}:"
                           f"Dimensions: {cdlt.operand_dim_mapping()}\n"
                           f"Times per level: {level_counter}\n"
                           f"Op: {cdlt.op_name}{cdlt.instance_id}\n"
                           f"Constraints:{[(k, t.fn_body_str) for k, t in tile_info.constraint_fps.items()]}\n\n"
                           f"Level Hints: {hint_str}\n"
                           f"Loop hints: {other_hint_str}"
                           )

    # Lastly, update operands


    ## Testing temporary

    # TODO: Store all information in the codelet
    cdlt._domain_tiling = {}
    cdlt._domain_loop_map = {}
    for l, dim_splits in tile_info.selected_splits.items():
        cdlt._domain_tiling[l] = {}
        cdlt._domain_loop_map[l] = {}
        for ld in tile_info.loop_dependencies:
            cdlt._domain_tiling[l][ld] = dim_splits[tile_info.loop_dim_map[ld]]
            cdlt._domain_loop_map[l][ld] = tile_info.shapes[l][tile_info.loop_dim_map[ld]]

    for o in cdlt.all_operands:
        for idx, a in enumerate(o.data_moves):
            if all(a in [None, 0] for a in list(a.offset_map.values())):
                assert idx > 0
                a.reinit_offset_map(o.data_moves[idx - 1].offset_map.copy())

            if len(a.shape_map) == 0:
                a.set_size_from_splits(cdlt, tile_info.selected_splits)

            a.set_offset_map(cdlt, tile_info.shapes)

            src_level = cdlt.get_tile_level(a.src_node)
            dst_level = cdlt.get_tile_level(a.dst_node)
            if src_level > dst_level:
                level = dst_level
            else:
                level = src_level

            if level in tile_info.derived_tilings:
                for name, off in a.offset_map.items():
                    if isinstance(off, Basic):
                        indices = a.get_symbol_atoms(off)
                        for i in indices:
                            i_as_str = a.get_symbol_str(i)
                            if i_as_str in tile_info.derived_tilings[level]:
                                a.derived_sizes[i_as_str] = tile_info.derived_tilings[level][i_as_str]['size'].value

    dom_tile = cdlt.domain_tiling[1]
    dom_loop = cdlt.domain_loop_map[1]
    for l in tile_info.selected_splits.keys():
        for ld in tile_info.loop_dependencies:
            if ld in tile_info.derived_tilings[l]:
                cdlt._derived_fps[l][ld] = tile_info.finalize_derived_param(l, ld, tile_info.loop_dependencies, cdlt._domain_tiling[l],
                                                                                         cdlt._domain_loop_map[l])


    return cdlt
