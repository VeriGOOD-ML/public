from typing import List, Union, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from codelets.adl.operation import Operation, Loop, Compute, Configure, Transfer
    from codelets.codelet_impl.codelet import Codelet


from sympy import Basic, Idx

TileConstraint = Dict[Tuple[str, str], Tuple[int, int]]


def unroll(loop):
    pass

def fuse(loops):
    pass

def reorder(loops, loop_permutation):
    pass

def find_minimum_idx(op: 'Operation', op_idx_map, op_list):
    dep_indices = [op_idx_map[o] for o in op.dependencies]
    if len(dep_indices) > 0:
        min_idx = max(dep_indices)
    else:
        min_idx = op_idx_map[op.op_str]

    return min_idx + 1

def split_loop(cdlt: 'Codelet', outer_loop: 'Loop', inner_loop: 'Loop', inner_tile_level: int):
    loop_domain_key = cdlt.domain_loop_map[outer_loop.op_str]
    cdlt.domain_loop_map[inner_loop.op_str] = loop_domain_key
    split_factor = cdlt.domain_tiling[inner_tile_level][loop_domain_key]
    initial_size = outer_loop.max() - outer_loop.min()

    if initial_size % split_factor != 0:
        raise RuntimeError(f"Invalid split factor for iterator:\n"
                           f"Split factor: {split_factor}\n"
                           f"Size: {initial_size}\n"
                           f"Loop key: {loop_domain_key}\n"
                           f"Loop min/max: {outer_loop.min()}, {outer_loop.max()}")

    outer_loop.start = 0
    outer_loop.end = initial_size
    outer_loop.stride = initial_size // split_factor
    outer_loop.offset = 0
    inner_loop.start = 0
    inner_loop.end = initial_size // split_factor
    inner_loop.stride = 1
    inner_loop.offset = 0


    return inner_loop

# TODO: THis function needs to be fixed, too complicated and not generalizeable
def split_transfer(cdlt: 'Codelet', outer_xfer: 'Transfer', inner_xfer: 'Transfer'):
    full_path = outer_xfer.path.copy()
    all_transfers = outer_xfer.transfers.copy()

    outer_xfer.path = full_path[:2]
    inner_xfer.path = full_path[1:]


    outer_xfer_key = tuple(full_path[:2])

    outer_xfer.transfers = {outer_xfer_key: all_transfers[outer_xfer_key]}
    inner_xfer.transfers.pop(outer_xfer_key)

    # Update dependencies
    new_inner_deps = []
    dep_map = {}
    dep_symbols = {}
    if inner_xfer.loop_level > outer_xfer.loop_level:

        for d in inner_xfer.dependencies:
            dep_op = cdlt.op_map[d]
            for level, name in dep_op.split_map.items():
                dep_map[name] = d
                dep_symbols[d] = Idx(name, (dep_op.start, dep_op.end))
                new_inner_deps.append(name)
        inner_xfer.dependencies = new_inner_deps

        new_offset = []

        for o in outer_xfer.transfers[outer_xfer_key]._src_offset:
            if isinstance(o, Basic):
                sym_map = {i: dep_symbols[str(i)] for i in list(o.atoms(Idx))}
                new_offset.append(o.subs(sym_map))

        inner_xfer.transfers[tuple(full_path[1:3])]._src_offset = new_offset
        outer_xfer.transfers[outer_xfer_key].compute_src_size(cdlt)
        for _, v in inner_xfer.transfers.items():
            v.compute_src_size(cdlt)
    else:
        for d in outer_xfer.dependencies:
            dep_op = cdlt.op_map[d]
            if dep_op.op_type == "compute":
                new_inner_deps.append(d)
                inner_xfer.dependencies.remove(d)
            else:
                for level, name in dep_op.split_map.items():
                    dep_map[name] = d
                    new_inner_deps.append(name)
                    if dep_op.op_type == "loop":
                        dep_symbols[d] = Idx(name, (dep_op.start, dep_op.end))

        outer_xfer.dependencies = new_inner_deps


        for path, xfer in inner_xfer.transfers.items():

            new_offset = []
            for o in xfer._dst_offset:
                if isinstance(o, Basic):
                    sym_map = {i: dep_symbols[str(i)] for i in list(o.atoms(Idx))}
                    new_offset.append(o.subs(sym_map))
                outer_xfer.transfers[outer_xfer_key]._dst_offset = new_offset
            xfer.compute_dst_size(cdlt)

        outer_xfer.transfers[outer_xfer_key].compute_dst_size(cdlt)

    return inner_xfer


def split_operation(cdlt: 'Codelet', op: 'Operation', loop_level: int, tile_level: int):
    # if isinstance(op, Compute):
    if op.op_type == 'compute':
        inner_op = op
        inner_op.loop_level = loop_level
    else:
        inner_op = op.copy(cdlt)
        inner_op.op_id = cdlt.op_id_counters[op.op_type]
        inner_op.global_op_id = cdlt.id_counter
        inner_op.loop_level = loop_level
        op.set_split_mapping(tile_level, inner_op.op_str)
        cdlt.op_id_counters[op.op_type] += 1
        cdlt.id_counter = cdlt.id_counter + 1

        # if isinstance(op, Transfer):
        if op.op_type == 'transfer':
            inner_op = split_transfer(cdlt, op, inner_op)
        # elif isinstance(op, Loop):
        elif op.op_type == 'loop':
            inner_op = split_loop(cdlt, op, inner_op, tile_level)

    return inner_op


def lift_op(new_index, old_index, op_list: List[Union['Compute', 'Loop', 'Transfer', 'Configure', 'Operation']]):
    op = op_list[old_index]
    op._loop_id = op_list[new_index-1].loop_id
    op._loop_level = op_list[new_index-1].loop_level if op_list[new_index-1].op_type != "loop" else op_list[new_index-1].loop_level + 1
    op_list.insert(new_index, op_list.pop(old_index))

# TODO: The ordering relative to other operations needs to consider the loop level
def lift_operations(cdlt: 'Codelet'):
    dep_indices = {l.op_str: i
                    for i, l in enumerate(cdlt.ops)}
    lifted_ops = cdlt.ops.copy()

    for o in cdlt.ops:
        if o.op_type != "loop" and len(o.dependencies) > 0:
            min_idx = find_minimum_idx(o, dep_indices, lifted_ops)
            if min_idx < dep_indices[o.op_str]:
                lift_op(min_idx, dep_indices[o.op_str], lifted_ops)
                dep_indices = {l.op_str: i
                               for i, l in enumerate(lifted_ops)}
    cdlt._ops = lifted_ops
    return cdlt

