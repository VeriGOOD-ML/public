from typing import List
from codelets.adl.graph import ArchitectureNode, ComputeNode, StorageNode
from codelets.adl import Codelet
from codelets.adl.backups.operand import Datatype

import numpy as np
from itertools import product
from codelets.compiler.transformations.util import factors

def get_compilation_parameters(hag: ArchitectureNode, cdlt: Codelet):
    tiling_options = get_tiling_options(hag, cdlt)
    return tiling_options


def get_capacities(hag_nodes: List[ArchitectureNode], cdlt_dtype_size: int):
    capacities = []
    for hn in hag_nodes:
        if isinstance(hn, StorageNode) and hn.on_chip:
            capacities.append(hn.size_bytes)
        elif isinstance(hn, ComputeNode):
            capacities.append(np.prod(hn.dimensions)*cdlt_dtype_size)
    return capacities

def get_tiled_dimensions(untiled_sizes, loop_order, perm):
    tiled_sizes = {}
    for n, i in enumerate(loop_order):
        tiled_sizes[i] = untiled_sizes[i]/perm[n]
    return tiled_sizes

def filter_tiling(hag: ArchitectureNode, cdlt: Codelet, looped_dimensions, dim_factors):
    untiled_args = looped_dimensions.copy()
    untiled_args.update(cdlt.op_params)
    all_ops = cdlt.inputs + cdlt.outputs
    ordered_perms = []
    for lo in cdlt.loop_order:
        ordered_perms.append(tuple(dim_factors[lo]))
    ordered_perms = product(*tuple(ordered_perms))
    valid_perms = []
    op_shapes = {}
    op_constraints = {}
    for a in all_ops:
        mem_nodes = [hag.get_subgraph_node(m_node) for m_node in a.memory_path]
        assert isinstance(a.supported_dtypes, Datatype)
        dtype_size = a.supported_dtypes.bitwidth // 8
        mem_capacities = get_capacities(mem_nodes, dtype_size)
        op_constraints[a.name] = mem_capacities

    # for o_perm in ordered_perms:
    #     eval_args = get_tiled_dimensions(untiled_args, cdlt.loop_order, o_perm)
    #     eval_args.update(cdlt.op_params)
    #     op_shape_list = {}
    #     for a in all_ops:
    #         for i in a.iteration_domain:
    #             # TODO: Add try-catch statement here
    #             op_shape = eval(i, eval_args)
    #             op_shape_list.append(op_shape)
    #         dtype_size = a.supported_dtypes.bitwidth // 8
    #         if all([np.prod(op_shape_list)*dtype_size <= ]):
    #             valid_perms.append(op_shapes[a.field_name] = op_shape_list
    return op_shapes

def get_tiling_options(hag: ArchitectureNode, cdlt: Codelet):
    all_ops = cdlt.inputs + cdlt.outputs
    tiling_dims = {}

    for op in all_ops:
        tiling_dims.update(op.shape_symbols)

    for k in tiling_dims.keys():
        tiling_dims[k] = tiling_dims[k]['value']

    looped_dimensions = {s: tiling_dims[s] for s in cdlt.loop_order}
    dim_factors = {k: factors(v) for k, v in looped_dimensions.items()}
    filter_tile_factors = filter_tiling(hag, cdlt, looped_dimensions, dim_factors)

    return looped_dimensions


