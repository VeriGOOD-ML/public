from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from codelets.codelet_impl import Codelet

from .tiling_utils import set_codelet_tiling
from .stage_utils import default_tile_heuristic, update_shape_from_arch, store_tile_checkpoint
import polymath as pm
import json

SYSTOLIC_ARRAY_CDLTS = ['conv_bias', 'conv', 'gemm', 'gemm_no_bias']

# TODO: Update SIMD_CDLTS for dtypes
SIMD_CDLTS = ['max_pool', 'elem_add', 'relu', 'global_avg_pool', 'batch_normalization',
              'sgd4', 'elem_add_grad', 'sgd4d']
POOL_OPS = ['max_pool', 'global_avg_pool']
BINARY_SIMD = ['elem_add', 'sgd4d', 'relu_grad', 'elem_add_grad', 'global_average_pool_grad', 'relu_grad',
               'sgd4d', 'max_pool_grad']

UNARY_SIMD = ['relu', 'max_pool', 'global_avg_pool']
NOOPS = ['coarse_flatten']
STANDARD_SHAPE_OPS = ['elem_add', 'relu', 'global_avg_pool', 'batch_norm', 'sgd4d',
                      'max_pool_grad', 'global_average_pool_grad', 'relu_grad', 'elem_add_grad']

def update_operand_dtypes(program, node: pm.Node, cdlt: 'Codelet', dtype_map=None) -> 'Codelet':
    if cdlt.op_name in SYSTOLIC_ARRAY_CDLTS:
        cdlt.inputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        cdlt.inputs[1].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        if len(cdlt.inputs) == 3:
            cdlt.inputs[2].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
        cdlt.outputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
    else:
        for o in cdlt.operands:
            o.set_dtype(dtype_map['SIMD'])
    return cdlt

def pad_operands(program, node: pm.Node, cdlt: 'Codelet', shaped_nodes=None) -> 'Codelet':
    assert isinstance(shaped_nodes, dict)
    if cdlt.op_name in ['conv', 'conv_bias']:
        activation = node.inputs[0]
        weight = node.inputs[1]
        out = node.outputs[0]
        sys_array_dims = program.hag.get_subgraph_node("pe_array").dimensions
        out_shape = update_shape_from_arch(out, shaped_nodes, sys_array_dims[1], 3)
        act_shape = update_shape_from_arch(activation, shaped_nodes, sys_array_dims[0], 3)
        weight_shape = update_shape_from_arch(weight, shaped_nodes, sys_array_dims[1], 2)
        weight_shape = update_shape_from_arch(weight, shaped_nodes, sys_array_dims[0], 3, force_reshape=True)
        assert weight_shape[2] == out_shape[3]
        if weight_shape[3] != act_shape[3]:
            raise RuntimeError(f"Weight and activation shapes are incorrect:"
                               f"Weight {weight.name} shape: {weight_shape}/{weight.shape}\n"
                               f"Activation {activation.name} shape: {act_shape}/{activation.shape}")
        cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'IC'])
        cdlt.inputs[0].add_padding('IH', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.inputs[0].add_padding('IW', node.kwargs['pad'], symmetric=True, dynamic=True)
        cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'OC'])
        cdlt.inputs[1].set_dim_order(['KH', 'KW', 'OC', 'IC'])
        if len(node.inputs) == 3:
            bias = node.inputs[2]
            bias_shape = update_shape_from_arch(bias, shaped_nodes, sys_array_dims[1], 0)
            assert bias_shape[0] == out_shape[3]
    elif cdlt.op_name in ['batchnorm_grad', 'batch_norm']:
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions

        for idx, i in enumerate(node.inputs):
            if len(i.shape) == 4:
                shaped_output = update_shape_from_arch(i, shaped_nodes, simd_dims[0], 3)
                cdlt.inputs[idx].set_dim_order(['N', 'H', 'W', 'C'])
            elif program.program_mode == 'training':
                assert len(i.shape) == 1
                shaped_output = update_shape_from_arch(i, shaped_nodes, simd_dims[0], 0)
        for idx, i in enumerate(node.outputs):
            if len(i.shape) == 4:
                shaped_output = update_shape_from_arch(i, shaped_nodes, simd_dims[0], 3)
                cdlt.outputs[idx].set_dim_order(['N', 'H', 'W', 'C'])
            elif program.program_mode == 'training':
                assert len(i.shape) == 1
                shaped_output = update_shape_from_arch(i, shaped_nodes, simd_dims[0], 0)
    elif cdlt.op_name in ['gemm', 'gemm_no_bias']:
        sys_array_dims = program.hag.get_subgraph_node("pe_array").dimensions

        activation = node.inputs[0]
        weight = node.inputs[1]

        out = node.outputs[0]
        if 'transB' in node.kwargs:
            assert bool(node.kwargs['transB']) == False

        if 'transA' in node.kwargs:
            assert bool(node.kwargs['transA']) == False

        act_shape = update_shape_from_arch(activation, shaped_nodes, sys_array_dims[0], 1)
        weight_shape = update_shape_from_arch(weight, shaped_nodes, sys_array_dims[0], 0)
        weight_shape = update_shape_from_arch(weight, shaped_nodes, sys_array_dims[0], 1, force_reshape=True)
        out_shape = update_shape_from_arch(out, shaped_nodes, sys_array_dims[1], 1)

        if program.program_mode == 'training':
            inp_shape = update_shape_from_arch(activation, shaped_nodes, sys_array_dims[0], 0, force_reshape=True)
            out_shape = update_shape_from_arch(out, shaped_nodes, sys_array_dims[1], 0, force_reshape=True)
            if out_shape[0] != inp_shape[0]:
                raise RuntimeError(f"Input and output shapes are incorrect for {cdlt.op_name}:"
                                   f"Input {activation.name} shape: {inp_shape}/{activation.shape}\n"
                                   f"Output {out.name} shape: {out_shape}/{out.shape}")
            if out_shape[1] != weight_shape[1]:
                raise RuntimeError(f"WEight and output shapes are incorrect for {cdlt.op_name}:"
                                   f"Weight {weight.name} shape: {weight_shape}/{weight.shape}\n"
                                   f"Activation {out.name} shape: {out_shape}/{out.shape}")

        if weight_shape[0] != act_shape[1]:
            raise RuntimeError(f"Weight and activation shapes are incorrect:"
                               f"Weight {weight.name} shape: {weight_shape}/{weight.shape}\n"
                               f"Activation {activation.name} shape: {act_shape}/{activation.shape}")
        if len(node.inputs) == 3:
            bias = node.inputs[2]
            bias_shape = update_shape_from_arch(bias, shaped_nodes, sys_array_dims[1], 0)

            if bias_shape[0] != out_shape[1]:
                raise RuntimeError(f"Bias and output shapes are incorrect for {cdlt.op_name}"
                                   f"Bias {bias.name} shape: {bias_shape}/{bias.shape}\n"
                                   f"Output {out.name} shape: {out_shape}/{out.shape}")
    elif cdlt.op_name == 'reduce_sum':
        assert len(node.inputs[0].shape) == 2
        assert len(node.outputs[0].shape) == 1
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions
        data = node.inputs[0]
        out = node.outputs[0]
        data_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 0)
        data_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 1, force_reshape=True)
        out_shape = update_shape_from_arch(out, shaped_nodes, simd_dims[0], 0)
    elif cdlt.op_name in UNARY_SIMD:
        simd_constraint = program.hag.get_subgraph_node("SIMD").dimensions[0]
        data = node.inputs[0]
        out = node.outputs[0]
        data_shape = update_shape_from_arch(data, shaped_nodes, simd_constraint, 3)
        out_shape = update_shape_from_arch(out, shaped_nodes, simd_constraint, 3)

        if cdlt.op_name == 'max_pool':
            cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
            cdlt.inputs[0].add_padding('IH', node.kwargs['pad'], symmetric=True, dynamic=True)
            cdlt.inputs[0].add_padding('IW', node.kwargs['pad'], symmetric=True, dynamic=True)
            cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'C'])
            assert len(node.args) == 4 and isinstance(node.args[2], int)
            node.add_attribute('KH', node.kernel_size[0])

            assert len(node.args) == 4 and isinstance(node.args[3], int)
            node.add_attribute('KW', node.kernel_size[1])
            sy, sx = node.kwargs['stride'][0], node.kwargs['stride'][1]
            assert isinstance(sy, int)
            assert isinstance(sx, int)
            node.add_attribute('sy', sy)
            node.add_attribute('sx', sx)
        elif cdlt.op_name == 'global_avg_pool':
            cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'OH', 'OW', 'C'])

    elif cdlt.op_name in ['sgd1d', 'sgd2d']:
        simd_constraint = program.hag.get_subgraph_node("SIMD").dimensions[0]

        data = node.inputs[0]
        grad = node.inputs[1]
        out = node.outputs[0]
        data_shape = update_shape_from_arch(data, shaped_nodes, simd_constraint, len(data.shape) - 1)

        if grad.name not in shaped_nodes:
            raise RuntimeError(f"Gradient {grad.name} not found in shaped nodes for {node.op_name}")
        assert grad.name in shaped_nodes
        assert grad.shape == data.shape
        if program.program_mode == 'training':
            out_shape = update_shape_from_arch(out, shaped_nodes, simd_constraint, len(out.shape) - 1)
            assert data.shape == out_shape

    elif cdlt.op_name == 'cross_entropy_loss':
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions
        data = node.inputs[0]
        target = node.inputs[1]
        out = node.outputs[0]

        class_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 0, force_reshape=True)
        class_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 1, force_reshape=True)
        updated_shape = update_shape_from_arch(target, shaped_nodes, simd_dims[0], 0)
        last_shape = update_shape_from_arch(out, shaped_nodes, simd_dims[0], 0)

        if updated_shape[0] != class_shape[0]:
            raise RuntimeError(f"Shape update for {cdlt.op_name} was invalid:\n"
                               f"Data {data.name}: {data.shape}/{class_shape}\n"
                               f"Target {target.name}: {target.shape}/{updated_shape}\n"
                               f"Output {out.name}: {out.shape}/{last_shape}")
    elif cdlt.op_name == 'cross_entropy_loss_grad':
        simd_dims = program.hag.get_subgraph_node("SIMD").dimensions
        data = node.inputs[0]
        target = node.inputs[1]
        grad = node.inputs[2]
        out = node.outputs[0]

        class_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 0)
        class_shape = update_shape_from_arch(data, shaped_nodes, simd_dims[0], 1, force_reshape=True)
        updated_shape = update_shape_from_arch(target, shaped_nodes, simd_dims[0], 0)
        grad_shape = update_shape_from_arch(grad, shaped_nodes, simd_dims[0], 0)
        last_shape = update_shape_from_arch(out, shaped_nodes, simd_dims[0], 0)
        last_shape = update_shape_from_arch(out, shaped_nodes, simd_dims[0], 1, force_reshape=True)

        # TODO: Need to fix this at some point to be consistent
        assert last_shape[0] == updated_shape[0] and class_shape[0] == last_shape[0]

    elif cdlt.op_name in BINARY_SIMD:
        simd_constraint = program.hag.get_subgraph_node("SIMD").dimensions[0]

        op1 = node.inputs[0]
        op2 = node.inputs[1]
        out = node.outputs[0]

        op1_shape = update_shape_from_arch(op1, shaped_nodes, simd_constraint, 3)
        op2_shape = update_shape_from_arch(op2, shaped_nodes, simd_constraint, 3)
        out_shape = update_shape_from_arch(out, shaped_nodes, simd_constraint, 3)


        if cdlt.op_name == 'global_average_pool_grad':
            cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
            cdlt.inputs[1].set_dim_order(['N', 'OH', 'OW', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
        elif cdlt.op_name == 'elem_add_grad':
            op3 = node.inputs[2]
            op3_shape = update_shape_from_arch(op3, shaped_nodes, simd_constraint, 3)
            grad1 = node.outputs[1]
            grad1_shape = update_shape_from_arch(grad1, shaped_nodes, simd_constraint, 3)
            cdlt.inputs[0].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.inputs[1].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.inputs[2].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.outputs[1].set_dim_order(['N', 'H', 'W', 'C'])

        elif cdlt.op_name == 'max_pool_grad':

            node.add_attribute('KH', node.kernel_size[0])
            node.add_attribute('KW', node.kernel_size[1])
            sy, sx = node.kwargs['stride'][0], node.kwargs['stride'][1]
            assert isinstance(sy, int)
            assert isinstance(sx, int)
            node.add_attribute('sy', sy)
            node.add_attribute('sx', sx)
            cdlt.inputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
            cdlt.inputs[1].set_dim_order(['N', 'OH', 'OW', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'IH', 'IW', 'C'])
        else:
            cdlt.inputs[0].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.inputs[1].set_dim_order(['N', 'H', 'W', 'C'])
            cdlt.outputs[0].set_dim_order(['N', 'H', 'W', 'C'])
        if cdlt.op_name not in ['max_pool_grad', 'global_average_pool_grad']:
            if op1_shape != op2_shape:
                raise RuntimeError(f"Operand1 and Operand2 shapes are incorrect for {cdlt.op_name}:\n"
                                   f"Operand1 {op1.name} shape: {op1_shape}/{op1.shape}\n"
                                   f"Operand2 {op2.name} shape: {op2_shape}/{op2.shape}\n")

            if out_shape != op2_shape:
                raise RuntimeError(f"Operand1 and output shapes are incorrect for {cdlt.op_name}:\n"
                                   f"Operand1 {op1.name} shape: {op1_shape}/{op1.shape}\n"
                                   f"Output {out.name} shape: {out_shape}/{out.shape}\n")
    elif cdlt.op_name in NOOPS:
        pass
    else:
        raise RuntimeError(f"Node: {node.op_name}\n"
              f"Shapes: {node.inputs[0].shape}\n"
              f"{node.inputs[1].shape}\n"
              f"{node.outputs[0].shape}")

    return cdlt

def tile(program, node: pm.Node, cdlt: 'Codelet', factor_fn_name='default', heuristic_fn=None, checkpoint_file=None) -> 'Codelet':
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

    cdlt = set_codelet_tiling(cdlt, hag, factor_fn_name)

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

                if op.op_type == 'transfer':
                    if len(op.path) <= 2:
                        dep_mapping[op.op_str] = op.op_str
                        offset -= 1
                        outgoing = False
                        if cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                            outgoing = True
                            cdlt.insert_op(op, target_idx)
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

    if checkpoint_file is not None:
        store_tile_checkpoint(cdlt, checkpoint_file)

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


