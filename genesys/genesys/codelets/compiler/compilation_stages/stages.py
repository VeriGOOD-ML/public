from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from codelets.codelet_impl import Codelet


from .stage_utils import default_tile_heuristic, set_codelet_tiling
import polymath as pm
import json

SYSTOLIC_ARRAY_CDLTS = ['conv_bias', 'conv', 'gemm']
SIMD_CDLTS = ['max_pool', 'elem_add', 'relu', 'global_avg_pool', 'batch_normalization']
POOL_OPS = ['max_pool', 'global_avg_pool']
BINARY_SIMD = ['elem_add']
UNARY_SIMD = ['relu']

def update_operand_dtypes(program, node: pm.Node, cdlt: 'Codelet', dtype_map=None) -> 'Codelet':
    if cdlt.op_name in SYSTOLIC_ARRAY_CDLTS:
        cdlt.inputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        cdlt.inputs[1].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        if len(cdlt.inputs) == 3:
            cdlt.inputs[2].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
        cdlt.outputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
    else:
        assert cdlt.op_name in SIMD_CDLTS
        for o in cdlt.operands:
            o.set_dtype(dtype_map['SIMD'])
    return cdlt

def add_backprop(program, node: pm.Node, cdlt: 'Codelet'):
    pass

def update_batch_size(program, node: pm.Node, cdlt: 'Codelet', batch_size=None) -> 'Codelet':
    if cdlt.op_name == 'conv':
        pass

    return cdlt

def pad_operands(program, node: pm.Node, cdlt: 'Codelet', shaped_nodes=None) -> 'Codelet':
    if cdlt.op_name in "conv_bias":
        assert isinstance(shaped_nodes, list)
        activation = node.inputs[0]
        weight = node.inputs[1]
        bias = node.inputs[2]
        out = node.outputs[0]
        sys_array_dims = program.hag.get_subgraph_node("pe_array").dimensions

        if out.name not in shaped_nodes:
            if out.shape[1] % sys_array_dims[1] != 0:
                oc_shape = out.shape[1] + (sys_array_dims[1] - (out.shape[1] % sys_array_dims[1]))
            else:
                oc_shape = out.shape[1]
            out.shape = tuple([out.shape[0], out.shape[2], out.shape[3], oc_shape])
            shaped_nodes.append(out.name)
        else:
            oc_shape = out.shape[-1]

        if bias.name not in shaped_nodes:
            bias.shape = tuple([oc_shape])
            shaped_nodes.append(bias.name)

        if activation.name not in shaped_nodes:
            if activation.shape[1] % sys_array_dims[0] != 0:
                ic_shape = activation.shape[1] + (sys_array_dims[0] - (activation.shape[1] % sys_array_dims[0]))
            else:
                ic_shape = activation.shape[1]
            activation.shape = tuple([activation.shape[0], activation.shape[2] + 2*node.kwargs['pad'], activation.shape[3] + 2*node.kwargs['pad'], ic_shape])
            # activation.shape = tuple([activation.shape[0], activation.shape[2], activation.shape[3], ic_shape])
            shaped_nodes.append(activation.name)
        else:
            ic_shape = activation.shape[-1]



        if weight.name not in shaped_nodes:
            weight.shape = tuple([weight.shape[2], weight.shape[3], oc_shape, ic_shape])
            shaped_nodes.append(weight.name)

        # assert 'pad' in node.kwargs.keys()

        cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'IC'])
        cdlt.inputs[0].add_padding('IH', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.inputs[0].add_padding('IW', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'OC'])
        cdlt.inputs[1].set_dim_order(['KH', 'KW', 'OC', 'IC'])


    elif cdlt.op_name == "gemm":
        sys_array_dims = program.hag.get_subgraph_node("pe_array").dimensions

        activation = node.inputs[0]
        weight = node.inputs[1]
        bias = node.inputs[2]
        out = node.outputs[0]

        if 'transB' in node.kwargs and node.kwargs['transB'] == 1:
            weight.shape = (weight.shape[1], weight.shape[0])

        if 'transA' in node.kwargs and node.kwargs['transA'] == 1:
            activation.shape = (activation.shape[1], activation.shape[0])

        if activation.name not in shaped_nodes:
            if activation.shape[1] % sys_array_dims[0] != 0:
                ic_shape = activation.shape[1] + (sys_array_dims[0] - (activation.shape[1] % sys_array_dims[0]))
            else:
                ic_shape = activation.shape[1]
            activation.shape = tuple([activation.shape[0], ic_shape])
            shaped_nodes.append(activation.name)
        else:
            ic_shape = activation.shape[-1]

        if weight.name not in shaped_nodes:
            if weight.shape[1] % sys_array_dims[0] != 0:
                oc_shape = weight.shape[1] + (sys_array_dims[0] - (weight.shape[1] % sys_array_dims[0]))
            else:
                oc_shape = weight.shape[1]
            weight.shape = tuple([ic_shape, oc_shape])
            shaped_nodes.append(weight.name)
        else:
            oc_shape = weight.shape[1]

        if out.name not in shaped_nodes:
            out.shape = tuple([out.shape[0], oc_shape])
            shaped_nodes.append(out.name)

        if bias.name not in shaped_nodes:
            bias.shape = tuple([oc_shape])
            shaped_nodes.append(bias.name)
    elif cdlt.op_name == 'max_pool':
        activation = node.inputs[0]
        out = node.outputs[0]
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions

        if 'KH' not in node.kwargs:
            assert len(node.args) == 4 and isinstance(node.args[2], int)
            node.add_attribute('KH', node.args[2])

        if 'KW' not in node.kwargs:
            assert len(node.args) == 4 and isinstance(node.args[3], int)
            node.add_attribute('KW', node.args[3])

        if 'stride' in node.kwargs and isinstance(node.kwargs['stride'], list):
            sy, sx = node.kwargs['stride'][0], node.kwargs['stride'][1]
            assert isinstance(sy, int)
            assert isinstance(sx, int)
            node.add_attribute('sy', sy)
            node.add_attribute('sx', sx)

        if out.name not in shaped_nodes:
            if out.shape[1] % simd_dims[0] != 0:
                oc_shape = out.shape[1] + (simd_dims[1] - (out.shape[1] % simd_dims[0]))
            else:
                oc_shape = out.shape[1]
            out.shape = tuple([out.shape[0], out.shape[2], out.shape[3], oc_shape])
            shaped_nodes.append(out.name)
        else:
            oc_shape = out.shape[-1]

        if activation.name not in shaped_nodes:
            if activation.shape[1] % simd_dims[0] != 0:
                ic_shape = activation.shape[1] + (simd_dims[0] - (activation.shape[1] % simd_dims[0]))
            else:
                ic_shape = activation.shape[1]
            py, px = node.kwargs['pad'][0], node.kwargs['pad'][1]
            activation.shape = tuple([activation.shape[0], activation.shape[2] + 2 * py,
                                      activation.shape[3] + 2 * px, ic_shape])
            shaped_nodes.append(activation.name)
        else:
            ic_shape = activation.shape[-1]

        assert ic_shape == oc_shape

        cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
        cdlt.inputs[0].add_padding('IH', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.inputs[0].add_padding('IW', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'C'])
    elif cdlt.op_name in ['elem_add', 'relu', 'global_avg_pool']:
        activation = node.inputs[0]
        out = node.outputs[0]
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions
        if out.name not in shaped_nodes:
            if out.shape[1] % simd_dims[0] != 0:
                oc_shape = out.shape[1] + (simd_dims[1] - (out.shape[1] % simd_dims[0]))
            else:
                oc_shape = out.shape[1]
            out.shape = tuple([out.shape[0], out.shape[2], out.shape[3], oc_shape])
            shaped_nodes.append(out.name)
        else:
            oc_shape = out.shape[-1]

        if activation.name not in shaped_nodes:
            if activation.shape[1] % simd_dims[0] != 0:
                ic_shape = activation.shape[1] + (simd_dims[0] - (activation.shape[1] % simd_dims[0]))
            else:
                ic_shape = activation.shape[1]
            activation.shape = tuple([activation.shape[0], activation.shape[2], activation.shape[3], ic_shape])
            shaped_nodes.append(activation.name)
        else:
            ic_shape = activation.shape[-1]

        if cdlt.op_name in BINARY_SIMD:
            op2 = node.inputs[1]
            if op2.name not in shaped_nodes:
                if op2.shape[1] % simd_dims[0] != 0:
                    ic_shape2 = op2.shape[1] + (simd_dims[0] - (op2.shape[1] % simd_dims[0]))
                else:
                    ic_shape2 = op2.shape[1]
                op2.shape = tuple([op2.shape[0], op2.shape[2], op2.shape[3], ic_shape2])
                shaped_nodes.append(op2.name)
            else:
                ic_shape2 = op2.shape[-1]

            assert ic_shape2 == ic_shape
            cdlt.inputs[1].set_dim_order(['N', 'H', 'W', 'C'])
        assert ic_shape == oc_shape
        if cdlt.op_name == 'global_avg_pool':
            cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'C'])
        else:
            cdlt.inputs[0].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'H', 'W', 'C'])

    return cdlt

def tile(program, node: pm.Node, cdlt: 'Codelet', heuristic_fn=None) -> 'Codelet':
    hag = program.hag
    cdlt.set_tile_levels()
    heuristic_fn = heuristic_fn or default_tile_heuristic
    # Find amount of splits for each loop by looking at dependencies
    loop_splits = {}
    for i, o in enumerate(cdlt.operands):
        loops = [d for d in o.dependencies if "loop" in d]
        max_level = max(cdlt.get_tile_level(dp) for dp in o.data_path)
        for l in loops:
            if l in loop_splits and loop_splits[l] < max_level:
                loop_splits[l] = max_level
            else:
                loop_splits[l] = max_level


    bands = cdlt.extract_bands()

    cdlt = set_codelet_tiling(cdlt, hag, heuristic_fn)

    for start, end in bands:
        idx = start
        splits = loop_splits[cdlt.ops[idx].op_str] - 1
        dep_mapping = {}
        for split in range(splits):
            op_band = cdlt.ops[start: end + 1]
            offset = (end - start)
            num_splits = 0
            for op in op_band:
                i = cdlt.ops.index(op)
                target_idx = offset + i

                # if isinstance(cdlt.ops[target_idx], Loop):
                if cdlt.ops[target_idx].op_type == 'loop':
                    inner_loop_level = cdlt.ops[target_idx].loop_level + 1
                else:
                    inner_loop_level = cdlt.ops[target_idx].loop_level

                if inner_loop_level < op.loop_level:
                    raise RuntimeError

                inner_deps = [dep_mapping[dp] for dp in op.dependencies]
                new_op_id, new_global_id = cdlt.get_new_op_ids(op)
                extra_kwargs = {}

                # if isinstance(op, Transfer):
                if op.op_type == 'transfer':
                    if len(op.path) <= 2:
                        dep_mapping[op.op_str] = op.op_str
                        offset -= 1
                        outgoing = False
                        if cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                            outgoing = True
                            cdlt.insert_op(op, target_idx)
                        # op.operand.update_op_accesses(cdlt, op, dep_mapping)
                        op.operand.update_transfer_access(op, outgoing=outgoing)
                        continue
                    elif cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                        outgoing = True
                        inner_path, outer_path = op.path[split: split + 2], op.path[split + 1:]
                        op._path = outer_path
                        extra_kwargs["path"] = inner_path
                        extra_kwargs["operand"] = op.operand
                        inner_op = cdlt.ops[i].copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                        assert id(op.operand) == id(inner_op.operand)
                        op.operand.update_op_accesses(cdlt, inner_op, dep_mapping)
                        op.operand.update_transfer_access(inner_op, outgoing=outgoing)

                        inner_idx = target_idx
                        dep_mapping[op.op_str] = inner_op.op_str

                        # Update outer op
                        op._dependencies.append(inner_op.op_str)
                        cdlt.insert_op(op, target_idx)
                    else:
                        outgoing = False
                        outer_path, inner_path = op.path[split: split + 2], op.path[split + 1:]
                        op._path = outer_path
                        extra_kwargs["path"] = inner_path
                        extra_kwargs["operand"] = op.operand
                        inner_deps.append(op.op_str)
                        inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                        assert id(op.operand) == id(inner_op.operand)
                        op.operand.update_op_accesses(cdlt, op, dep_mapping)

                        op.operand.update_transfer_access(inner_op, outgoing)

                        inner_idx = target_idx + 1
                        dep_mapping[op.op_str] = inner_op.op_str

                    num_splits += 1
                # elif isinstance(op, Loop):
                elif op.op_type == 'loop':

                    extra_kwargs['start'] = 0
                    extra_kwargs['end'] = cdlt.domain_loop_map[split + 1][op.op_str]
                    extra_kwargs['stride'] = 1

                    inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    loop_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                    cdlt._domain_loop_map[split+1][inner_op.op_str] = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.start = 0
                    op.stride = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.end = cdlt.domain_loop_map[split][op.op_str]
                    cdlt._domain_loop_map[split+1].pop(op.op_str)

                    dep_mapping[op.op_str] = inner_op.op_str
                    inner_idx = target_idx + 1
                    num_splits += 1
                else:
                    assert op.op_type == 'compute'
                    dep_mapping[op.op_str] = op.op_str
                    op.dependencies = inner_deps
                    op.loop_level = inner_loop_level
                    inner_op = op

                    for s in op.sources:
                        s.update_op_accesses(cdlt, inner_op, dep_mapping)
                        s.compute_tile(op, "source")
                    for d in op.dests:
                        d.update_op_accesses(cdlt, inner_op, dep_mapping)
                        d.compute_tile(op, "dest")

                    inner_idx = target_idx
                    num_splits += 1

                cdlt.insert_op(inner_op, inner_idx)


    for o in cdlt.operands:
        if len(o.data_moves) > 0 and o.data_moves[-1].dst_node not in o.tiling:

            last_move = o.data_moves[-1]
            dest_name = last_move.dst_node
            level = cdlt.get_tile_level(dest_name)
            level_sizes = cdlt.domain_loop_map[level]
            o.tiling[dest_name] = last_move.get_size_from_loops(cdlt, level_sizes)

        if o in cdlt.outputs and not o.is_tiled():
            missing_tiles = [l for l in o.unique_data_locations() if l not in list(o.tiling.keys())]
            prev_level = cdlt.get_tile_level(missing_tiles[0])
            for m in missing_tiles:
                level = cdlt.get_tile_level(m)
                level_sizes = cdlt.domain_loop_map[level]
                mmove = None
                for a in o.data_moves:
                    if a.src_node == m:
                        mmove = a
                        break
                prev_level = level
                if mmove is None:
                    raise RuntimeError(f"UNable to find movement for missing tile {m}\n"
                                       f"Moves: {o.movement_keys()}")
                o.tiling[m] = mmove.get_size_from_loops(cdlt, level_sizes)

        if not o.is_tiled():
            raise RuntimeError(f"Number of tilings does not match the data path size for {o.name}:\n"
                               f"Tiling keys: {list(o.tiling.keys())}\n"
                               f"Unique data path locations: {o.unique_data_locations()}\n"
                               f"Data path: {o.data_path}")
    return cdlt


def hoist(program, node: pm.Node, cdlt: 'Codelet') -> 'Codelet':

    for o in cdlt.ops:
        i = cdlt.ops.index(o)
        i_loop_level = o.loop_level
        idx = -1
        loop_level = -1

        for dep in o.dependencies:

            dep_idx = cdlt.ops.index(cdlt.op_map[dep])
            if cdlt.ops[dep_idx].op_type == "loop":
                dep_level = cdlt.ops[dep_idx].loop_level + 1
            else:
                dep_level = cdlt.ops[dep_idx].loop_level

            if dep_level > loop_level:
                loop_level = dep_level

            if dep_idx > idx:
                idx = dep_idx

        if idx < 0:
            idx = i

        if idx < i:
            cdlt.ops.insert(idx + 1, cdlt.ops.pop(i))
            idx += 1

        if loop_level < i_loop_level and loop_level > 0:
            cdlt.ops[idx].loop_level = loop_level
    return cdlt


def insert_dtype_cast(program, n, cdlt):
    pass


