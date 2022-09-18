from copy import deepcopy

from codelets.templates.operand_template import IndexOperandTemplate
from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from codelets.codelet_impl import Codelet
from codelets.codelet_impl.codelet import USE_LOOP_END
from codelets.compiler.program import CodeletProgram

from examples.genesys.compilation_stages.tiling_utils import set_codelet_tiling
from examples.genesys.compilation_stages.stage_utils import default_tile_heuristic, \
    store_tile_checkpoint, \
    find_node_key, insert_simd_typecast
from . import CUSTOM_TILE_OPS

CUSTOM_PAD_OPS = CUSTOM_TILE_OPS + ["conv_bias", "conv_bias_add", "conv_bias_clip"]

import polymath as pm

TRANSPOSED_SHAPES = [['N', 'C', 'H', 'W'], ['N', 'IC', 'IH', 'IW'],
                     ['N', 'C', 'IH', 'IW'], ['N', 'OC', 'OH', 'OW'],
                     ['N', 'OC', 'OH1', 'OW1'],
                     ['ON', 'OC', 'OH', 'OW'], ['N', 'C', 'OH', 'OW']]
TRANSPOSE_PERM = [0, 2, 3, 1]
TRANSPOSE_POS = [0, 3, 1, 2]
FLIP_SHAPE_PERM = [2, 3, 1, 0]
POST_TRANSPOSE_SHAPES = [
    [ts[TRANSPOSE_PERM[i]] for i in range(len(TRANSPOSE_PERM))]
    for ts in TRANSPOSED_SHAPES
]

# FLIP_SHAPE_PERM = [2, 3, 0, 1]
FLIP_SHAPES = [['OC', 'IC', 'KH', 'KW'],
               ["C", "ONE", "KH", "KW"],
               ["OC", "ONE", "KH1", "KW1"]]

LANGUAGE_MODELS = ['bert-base-cased-transpose-opt-trimmed-ort', 'gpt2-trimmed-opt']


def quantize_codelet(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet') -> 'Codelet':

    return cdlt

def update_operand_dtypes(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', dtype_map=None) -> 'Codelet':

    compute_ops = cdlt.get_ops_by_type('compute')
    if any([o.target == 'pe_array' for o in compute_ops]):
        for c in compute_ops:
            if c.target == "pe_array":
                c.sources[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
                c.sources[1].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
                if len(c.sources) >= 3:
                    c.sources[2].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
                c.dests[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
    if any([o.target == 'SIMD' for o in compute_ops]):
        for c in compute_ops:
            if c.target == "SIMD":
                for o in c.operands:
                    o.set_dtype(dtype_map['SIMD'])
    return cdlt


def template_pad_pass(program, template: 'CodeletTemplate') -> 'CodeletTemplate':
    updated_dims = []
    if 'TRAINING' in program.hag.meta_cfg.keys() and program.hag.meta_cfg['TRAINING']:
        train = True
    else:
        train = False

    if template.op_name in program.metadata['FUSION_OP_INFO'].keys():
        pad_attrs = [k for k in template.dummy_ops.keys() if 'pad' in k]
        if 'max_pool' in template.op_name:
            raise RuntimeError
        elif len(pad_attrs) > 0 and 'IH' in template.dummy_ops.keys() and 'IW' in template.dummy_ops.keys():

            template.update_dummy_op('IH', template.node.inputs[0].shape[2] + template.node.pad_int)
            template.update_dummy_op('IW', template.node.inputs[0].shape[3] + template.node.pad_int)
            updated_dims.append('IW')

            updated_dims.append('IH')
            updated_dims.append('IW')
            # template.update_dummy_op('OW', template.node.inputs[0].shape[2] + 2 * template.node.kwargs['pad'])
            # template.update_dummy_op('OW', template.node.inputs[0].shape[3] + 2 * template.node.kwargs['pad'])

    else:
        if 'pad' in template.dummy_ops.keys():
            if template.op_name == "max_pool":
                template.update_dummy_op('IH', template.node.inputs[0].shape[2] + 2 * template.node.kwargs['pad'][0])
                template.update_dummy_op('IW', template.node.inputs[0].shape[3] + 2 * template.node.kwargs['pad'][0])
            else:
                # template.update_dummy_op('IH', template.node.inputs[0].shape[2] + 2 * template.node.kwargs['pad'])
                # template.update_dummy_op('IW', template.node.inputs[0].shape[3] + 2 * template.node.kwargs['pad'])
                template.update_dummy_op('IH', template.node.inputs[0].shape[2] + template.node.pad_int)
                template.update_dummy_op('IW', template.node.inputs[0].shape[3] + template.node.pad_int)
            updated_dims.append('IH')
            updated_dims.append('IW')
        if 'dilation' in template.dummy_ops.keys():
            assert "conv" in template.op_name
            template.update_dummy_op('KH', (template.node.inputs[1].shape[2] - 1) * template.node.dilation_int + 1)
            template.update_dummy_op('KW', (template.node.inputs[1].shape[3] - 1) * template.node.dilation_int + 1)
            updated_dims.append('KH')
            updated_dims.append('KW')

    compute_ops = template.get_ops_by_type('compute')
    if any([o.param_map['target'] == 'pe_array' for o in compute_ops]):
        # THis requires conv/gemm to be first operation
        compute_op = None
        for c in compute_ops:
            if c.param_map['target'] == "pe_array":
                compute_op = c
                break
        assert compute_op.param_map['op_name'] == 'MVMUL'
        # Need to pad IC
        # Static padding
        sys_dims = program.hag.get_subgraph_node("pe_array").dimensions[0]*program.hag.meta_cfg['DATA_WIDTH'] // 8
        bandwidth = program.hag.edge_map[('DRAM', 'IBUF')].bandwidth_bytes
        size_constr = max(sys_dims, bandwidth)
        mod_constr = bandwidth
        pad_fn = lambda shape, sh_constr, mod_constr: (shape + (sh_constr - shape).max(0)) + (mod_constr - (shape + (sh_constr - shape).max(0))) % mod_constr
        inp_dim = compute_op.param_map['sources'][0].operand_shape_list[-1]
        dummy_inp_dim = template.node.inputs[0].shape[1]
        if template.op_name in CUSTOM_PAD_OPS:
            new_inp_dim = pad_fn(dummy_inp_dim, size_constr, size_constr)
        else:
            new_inp_dim = pad_fn(dummy_inp_dim, size_constr, mod_constr)
        # new_inp_dim = pad_fn(dummy_inp_dim, size_constr, mod_constr)
        template.update_dummy_op(inp_dim.name, new_inp_dim)
        updated_dims.append(inp_dim.name)

        out_dim = compute_op.param_map['dests'][0].operand_shape_list[-1]
        # TODO: Need to validate
        if inp_dim.name == "N":
            idx = 1
        else:
            assert inp_dim.name == "IC"
            idx = 0
        dummy_out_dim = template.node.inputs[1].shape[idx]
        if template.op_name in CUSTOM_PAD_OPS:
            new_out_dim = pad_fn(dummy_out_dim, size_constr, size_constr)
        else:
            new_out_dim = pad_fn(dummy_out_dim, size_constr, mod_constr)

        template.update_dummy_op(out_dim.name, new_out_dim)
        updated_dims.append(out_dim.name)

    # Need to figure out if this works for DW Conv
    if any([o.param_map['target'] == 'SIMD' for o in compute_ops]) and program.name not in LANGUAGE_MODELS:
        simd_pad_dims = ["IC", "OC", "C"]
        constr = template.hag.all_subgraph_nodes['SIMD'].dimensions[0]
        # updated_dims = []
        for c in compute_ops:
            if c.param_map['target'] == "SIMD":
                for idx, i in enumerate(c.param_map['sources']):
                    dim = None
                    for d in simd_pad_dims:
                        if d in i.operand_shape_list_names:
                            dim = d
                            break
                        elif train and 'cross_entropy_loss' in template.op_name and "N" in i.operand_shape_list_names:
                            dim = "N"
                            break
                    if dim is None:
                        continue
                    # if "IC" in i.operand_shape_list_names:
                    #     dim = "IC"
                    # elif "OC" in i.operand_shape_list_names:
                    #     dim = "OC"
                    # elif "C" in i.operand_shape_list_names:
                    #     dim = "C"
                    # else:
                    #     continue
                    dim_idx = i.operand_shape_list_names.index(dim)
                    prev_idx = dim_idx
                    if i.operand_shape_list_names in POST_TRANSPOSE_SHAPES:
                        prev_idx = TRANSPOSE_PERM[dim_idx]
                    if i.operand_shape_list_names[dim_idx] not in updated_dims:
                        dimname = i.operand_shape_list[dim_idx].name
                        # Change dummy dim
                        dummy_dim = template.node.inputs[idx].shape[prev_idx]
                        template.update_dummy_op(dimname, dummy_dim + (constr - dummy_dim) % constr)
                        updated_dims.append(dimname)
                for idx, o in enumerate(c.param_map['dests']):
                    if "IC" in o.operand_shape_list_names:
                        dim = "IC"
                    elif "OC" in o.operand_shape_list_names:
                        dim = "OC"
                    elif "C" in o.operand_shape_list_names:
                        dim = "C"
                    else:
                        continue
                    dim_idx = o.operand_shape_list_names.index(dim)

                    prev_idx = dim_idx
                    if o.operand_shape_list_names in POST_TRANSPOSE_SHAPES:
                        prev_idx = TRANSPOSE_PERM[dim_idx]
                    if o.operand_shape_list_names[dim_idx] not in updated_dims:
                        dimname = o.operand_shape_list[dim_idx].name
                        dummy_dim = template.node.inputs[idx].shape[prev_idx]
                        template.update_dummy_op(dimname, dummy_dim + (constr - dummy_dim) % constr)
                        updated_dims.append(dimname)

    return template


def template_layout_pass(program, template: 'CodeletTemplate') -> 'CodeletTemplate':

    if program.name == 'bert-base-cased-transpose-opt-trimmed-ort':
        return template

    reordered_operands = {}
    for idx, i in enumerate(template.inputs):
        if i.shape_list_names in TRANSPOSED_SHAPES:
            i.reorder_shapes(TRANSPOSE_PERM)
            reordered_operands[i.name] = TRANSPOSE_PERM
        elif i.shape_list_names in FLIP_SHAPES:
            i.reorder_shapes(FLIP_SHAPE_PERM)
            reordered_operands[i.name] = FLIP_SHAPE_PERM

    for idx, o in enumerate(template.outputs):
        if o.shape_list_names in TRANSPOSED_SHAPES:
            o.reorder_shapes(TRANSPOSE_PERM)
            reordered_operands[o.name] = TRANSPOSE_PERM
        elif o.shape_list_names in FLIP_SHAPES:
            o.reorder_shapes(FLIP_SHAPE_PERM)
            reordered_operands[o.name] = FLIP_SHAPE_PERM

    for idx, t in enumerate(template.temps):
        if t.shape_list_names in TRANSPOSED_SHAPES:
            t.reorder_shapes(TRANSPOSE_PERM)
            reordered_operands[t.name] = TRANSPOSE_PERM
        elif t.shape_list_names in FLIP_SHAPES:
            t.reorder_shapes(FLIP_SHAPE_PERM)
            reordered_operands[t.name] = FLIP_SHAPE_PERM

    for o in template.ops:
        if o.op_type == 'transfer':
            operand = o.param_map['operand']
            if isinstance(operand, IndexOperandTemplate) and operand.name in reordered_operands:
                operand.reorder_offsets(reordered_operands[operand.name])


        elif o.op_type == 'compute':
            for iop in o.param_map['sources']:
                if isinstance(iop, IndexOperandTemplate) and iop.name in reordered_operands:
                    iop.reorder_offsets(reordered_operands[iop.name])

            for oop in o.param_map['dests']:
                if isinstance(oop, IndexOperandTemplate) and oop.name in reordered_operands:
                    oop.reorder_offsets(reordered_operands[oop.name])

    return template


def add_simd_typecast(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', dtype_map=None,
                      codelet_output_map=None) -> 'Codelet':


    cdlt_idx = program.codelets.index(cdlt)

    if cdlt.is_noop():
        pass
        # output_key = node.outputs[0].name
        # input_key = node.inputs[0].name
        #
        # if input_key not in dtype_map:
        #     input_key = find_node_key(node.inputs[0], dtype_map)
        #
        # dtype_map[output_key] = dtype_map[input_key]
        # codelet_output_map[output_key] = (cdlt.op_name, cdlt.instance_id)
        # insert_simd_typecast(program, node, cdlt.inputs[0], cdlt, dtype_map, codelet_output_map, input_key)

    else:
        pass
        # for idx, operand in enumerate(cdlt.inputs):
        #     i = node.inputs[idx]
        #     if not isinstance(i, (pm.input, pm.state)):
        #         i_key = i.name
        #         if i_key not in dtype_map:
        #             i_key = find_node_key(i, dtype_map)
        #             dtype_map[i.name] = dtype_map[i_key]
        #             codelet_output_map[i.name] = codelet_output_map[i_key]
        #         insert_simd_typecast(program, node, operand, cdlt, dtype_map, codelet_output_map, i_key)
        #     else:
        #         dtype_map[i.name] = cdlt.get_operand_by_node_name(i.name).dtype
        #         codelet_output_map[i.name] = (cdlt.op_name, cdlt.instance_id)
        #
        # # for o in node.outputs:
        # for idx in range(len(cdlt.outputs)):
        #     o = node.outputs[idx]
        #     dtype_map[o.name] = cdlt.get_operand_by_node_name(o.name).dtype
        #     codelet_output_map[o.name] = (cdlt.op_name, cdlt.instance_id)

    return cdlt


def remove_unused_variables(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', shaped_nodes=None) -> 'Codelet':
    assert isinstance(shaped_nodes, dict)
    # inputs = cdlt.inputs
    # for i in inputs:
    #     if i not in cdlt.used_inputs:
    #         cdlt.remove_input(i)
    return cdlt

def update_dependencies(cdlt, loop_replacement_map):

    for op in cdlt.ops:
        if op.op_type == "config" and op.start_or_finish == "end":
            if any([cdlt.ops[i].op_type != "config" for i in range(cdlt.ops.index(op) + 1, len(cdlt.ops))]) and not cdlt.is_fusion():
                cdlt.ops.insert(len(cdlt.ops) - 1, cdlt.ops.pop(cdlt.ops.index(op)))
        new_deps = []
        for d in op.dependencies:
            if d in loop_replacement_map:
                new_deps.append(loop_replacement_map[d])
            else:
                new_deps.append(d)
        op._dependencies = new_deps
    return cdlt

def update_data_movements(cdlt):
    for o in cdlt.operands:
        if len(o.data_moves) > 0 and o.data_moves[-1].dst_node not in o.tiling:
            last_move = o.data_moves[-1]
            dest_name = last_move.dst_node
            level = cdlt.get_tile_level(dest_name)
            level_sizes = cdlt.domain_loop_map[level]
            o.tiling[dest_name] = last_move.get_size_from_loops(cdlt, level_sizes)

        if o in cdlt.outputs and not o.is_tiled():
            missing_tiles = [l for l in o.unique_data_locations() if l not in list(o.tiling.keys())]
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


def update_temporary_data_moves(cdlt):
    for t in cdlt.temps:
        if len(t.data_path) > 2:

            missing_tiles = [l for l in t.unique_data_locations() if l not in list(t.tiling.keys()) or
                             all(i == 0 for i in t.tiling[l].values())]
            supported_operands = [o for o in cdlt.operands if t.shape_list == o.shape_list]
            for d in t.data_moves:
                if d.unset_offsets:
                    assert "compute" in d.op_name
                    updated_offset = False
                    for o in supported_operands:
                        for dm in o.data_moves:
                            if cdlt.get_tile_level(d.src_node) == cdlt.get_tile_level(dm.src_node) and \
                                    cdlt.get_tile_level(d.dst_node) == cdlt.get_tile_level(dm.dst_node):
                                d.reinit_offset_map(dm.offset_map)
                                t.tiling[d.src_node] = o.tiling[dm.src_node].copy()
                                d.resolve_offsets(cdlt)
                                d.evaluated_domain_offsets = deepcopy(dm.evaluated_domain_offsets)
                                updated_offset = True
                                break
                        if updated_offset:
                            break
                    assert updated_offset
            missing_tiles = [l for l in t.unique_data_locations() if l not in list(t.tiling.keys())]
            for m in missing_tiles:
                level = cdlt.get_tile_level(m)
                level_sizes = cdlt.domain_loop_map[level]
                mmove = None
                for a in t.data_moves:

                    if a.src_node == m:
                        mmove = a
                        break
                    elif a.dst_node == m:
                        mmove = a
                        break

                if mmove is None:
                    raise RuntimeError(f"UNable to find movement for missing tile {m}\n"
                                       f"Operand: {t.name}\n"
                                       f"Moves: {t.movement_keys()}")
                t.tiling[m] = mmove.get_size_from_loops(cdlt, level_sizes)

    return cdlt

def propagate_offsets(cdlt: 'Codelet', hag: 'ArchitectureNode') -> 'Codelet':
    # For each operand, we need to find data movements which correspond to compute operations
    # Iterate over each operands data movements, and

    for o in cdlt.all_operands:
        if len(o.data_path) == 0:
            continue
        # if len(o.data_path) == 0 or (len(o.data_path) == 1 and o.data_path[0] != "IMM"):
        if (len(o.data_path) == 1 and o.data_path[0] != "IMM"):
            raise RuntimeError(f"Operand {o.name} has invalid data path: {o.data_path}")
        dm_idx = 0

        while dm_idx < len(o.data_moves):
            dm = o.data_moves[dm_idx]
            op = cdlt.op_map[dm.op_name]

            # We found a compute datamovement
            if op.op_type == "compute":

                # Check to see if it is an operand which is a constant, and doesnt have offsets
                if all([o == 0 for o in dm.offset_map.values()]):
                    snode = hag.get_subgraph_node(dm.src_node)
                    single_elem_size = snode.data_size
                    assert single_elem_size == o.data_size, "Single element size does not match data size:\n" \
                                                            f"Elem size: {single_elem_size}\n" \
                                                            f"Data size: {o.data_size}"
                    assert len(o.data_path) == 2
                    assert list(dm.offset_map.keys()) == list(o.shape_symbols.keys())
                    dm.reinit_offset_map(o.shape_symbols.copy())
                    break
                if dm.src_node is None:
                    raise RuntimeError(f"Unset source node for data movement in operand {o.name}.\n"
                                       f"Dest node: {dm.dst_node}\n"
                                       f"Operation: {dm.op_name}")
                if dm.dst_node is None:
                    raise RuntimeError(f"Unset dest node for data movement in operand {o.name}.\n"
                                       f"Source node: {dm.src_node}\n"
                                       f"Operation: {dm.op_name}")
                if o in op.sources and cdlt.get_tile_level(dm.src_node) < cdlt.get_tile_level(dm.dst_node):
                    # Case 1: It is an operand which is read from in the compute.
                    # For this case, we need to backtrack, starting with the compute, and updating the offsets
                    # for transfers which were used to send data to the compute unit.

                    # Need to validate the movement to see that it is indeed reading the data, not writing
                    if dm_idx == 0 and op.op_name not in ["MACC", "MAX", "MIN"]:
                        raise RuntimeError(f"Operand {o.name} included\n"
                                           f"as source operand, but data movement tile level does not match:\n"
                                           f"({cdlt.get_tile_level(dm.src_node)}){dm.src_node} --> ({cdlt.get_tile_level(dm.dst_node)}){dm.dst_node}\n"
                                           f"Operation: {dm.op_name}.")
                    # Create a copy of the offset map which will be propagated backward
                    offset_map = dm.offset_map
                    offset_deps = sum([v for v in dm.symbol_atoms_map.values()], [])
                    offset_deps = [str(v) for v in offset_deps]
                    curr_idx = dm_idx - 1
                    while curr_idx >= 0 and cdlt.get_tile_level(o.data_moves[curr_idx].src_node) < cdlt.get_tile_level(o.data_moves[curr_idx].dst_node):
                        o.data_moves[curr_idx].reinit_offset_map(offset_map.copy())
                        cdlt.op_map[o.data_moves[curr_idx].op_name].update_dependencies(offset_deps)
                        curr_idx -= 1
                elif o in op.dests and cdlt.get_tile_level(dm.src_node) > cdlt.get_tile_level(dm.dst_node):
                    # Case 2: It is an operand which is written to in the compute.
                    # For this case, we need to move forward, starting with the compute, and updating the offsets
                    # for transfers which were used to send data from the compute unit back off chip.
                    assert o in op.dests
                    if dm_idx == (len(o.data_moves) - 1) and op.op_name != "ADD":
                        raise RuntimeError(f"Operand {o.name} included\n"
                                           f"as ddest operand, but data movement tile level does not match:\n"
                                           f"({cdlt.get_tile_level(dm.src_node)}){dm.src_node} --> ({cdlt.get_tile_level(dm.dst_node)}){dm.dst_node}\n"
                                           f"Operation: {dm.op_name}\n"
                                           f"{op.op_name}")
                    offset_deps = sum([v for v in dm.symbol_atoms_map.values()], [])
                    offset_deps = [str(v) for v in offset_deps]
                    offset_map = dm.offset_map
                    curr_idx = dm_idx + 1

                    while curr_idx < len(o.data_moves) and cdlt.get_tile_level(o.data_moves[curr_idx].src_node) > cdlt.get_tile_level(o.data_moves[curr_idx].dst_node):
                        o.data_moves[curr_idx].reinit_offset_map(offset_map.copy())
                        cdlt.op_map[o.data_moves[curr_idx].op_name].update_dependencies(offset_deps)
                        curr_idx += 1
                    dm_idx = curr_idx - 1
                else:
                    raise RuntimeError(f"Operand {o.name} has datamovement for {dm.op_name}, but has invalid datammovement:\n"
                                       f"Is source operand: {o in op.sources}\n"
                                       f"Is dest operand: {o in op.dests}\n"
                                       f"({cdlt.get_tile_level(dm.src_node)}){dm.src_node} --> ({cdlt.get_tile_level(dm.dst_node)}){dm.dst_node}\n")
            dm_idx += 1

    return cdlt

def tile(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', factor_fn_name='default', heuristic_fn=None,
         checkpoint_file=None, stopping_condition=None, selection_metric=None) -> 'Codelet':
    hag = program.hag
    cdlt.set_tile_levels()
    heuristic_fn = heuristic_fn or default_tile_heuristic

    cdlt = propagate_offsets(cdlt, program.hag)

    # Find amount of splits for each loop by looking at dependencies
    loop_splits = {}
    for i, o in enumerate(cdlt.all_operands):
        if len(o.dependencies) == 0 and len(o.data_path) == 0:
            continue
        loops = [d for d in o.dependencies if "loop" in d]
        max_level = max(cdlt.get_tile_level(dp) for dp in o.data_path)
        for l in loops:
            if l in loop_splits and loop_splits[l] < max_level:
                loop_splits[l] = max_level
            else:
                loop_splits[l] = max_level
    bands = cdlt.extract_bands()
    cdlt = set_codelet_tiling(cdlt, hag, factor_fn_name, stopping_condition, selection_metric, heuristic_fn)

    loop_replacement_map = {}
    start_end_ops = [(cdlt.ops[s], cdlt.ops[e]) for s, e in bands]
    all_deps = {}

    for split_idx, (start_op, end_op) in enumerate(start_end_ops):
        outer_loop_map = {}

        start = cdlt.ops.index(start_op)
        end = cdlt.ops.index(end_op)

        idx = start
        splits = loop_splits[cdlt.ops[idx].op_str] - 1
        llevels = [o.loop_level for o in cdlt.ops[start: end + 1]]
        max_level = max(llevels)
        min_level = min(llevels)
        dep_mapping = {}
        for split in range(splits):
            op_band = cdlt.ops[start: end + 1]
            ### DEBUG
            start_count = len(cdlt.ops)
            ### DEBUG
            offset = (end - start)

            for op in op_band:

                i = cdlt.ops.index(op)
                target_idx = offset + i
                inner_loop_level = (max_level - min_level) + op.loop_level
                if inner_loop_level < op.loop_level:
                    raise RuntimeError

                inner_deps = []
                for d in op.dependencies:
                    dp = cdlt.op_map[d]
                    if cdlt.ops.index(dp) >= start:
                        inner_deps.append(dep_mapping[d])

                new_op_id, new_global_id = cdlt.get_new_op_ids(op)
                extra_kwargs = {}
                if op.op_type == "loop_end":
                    cdlt.insert_op(op, target_idx + ((len(cdlt.ops) - start_count))//2)
                    continue
                elif op.op_type == 'transfer':

                    if len(op.path) <= 2:

                        dep_mapping[op.op_str] = op.op_str
                        outgoing = False

                        offset -= 1
                        if cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                            offset += 1
                            outgoing = True

                            cdlt.insert_op(op, target_idx + ((len(cdlt.ops) - start_count))//2)

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


                elif op.op_type == 'loop':

                    extra_kwargs['start'] = 0

                    extra_kwargs['end'] = cdlt.domain_loop_map[split + 1][op.op_str]
                    extra_kwargs['stride'] = 1

                    inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                       op_id=new_op_id,
                                       loop_id=new_op_id,
                                       global_op_id=new_global_id,
                                       dependencies=inner_deps, **extra_kwargs)


                    cdlt._domain_loop_map[split + 1][inner_op.op_str] = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.start = 0

                    op.stride = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.end = cdlt.domain_loop_map[split][op.op_str]
                    cdlt._domain_loop_map[split + 1].pop(op.op_str)

                    dep_mapping[op.op_str] = inner_op.op_str

                    inner_idx = target_idx + 1
                    if op.op_str in cdlt.derived_params[1]:
                        inner_op._end = cdlt.derived_params[1][op.op_str]['size']
                        op._num_iters = cdlt.derived_params[1][op.op_str]['split']
                        op._end = op._num_iters
                    # We need to create the end loop for the new inner loop, then move the original end loop to
                    # the correct ending indx
                    if USE_LOOP_END:
                        loop_end = cdlt.get_loop_end(op.op_str)
                        new_loop_end_id, new_global_loop_end_id = cdlt.get_new_op_ids(loop_end)
                        inner_op_end = loop_end.copy(cdlt, loop_name=inner_op.op_str, loop_level=inner_loop_level,
                                                     op_id=new_loop_end_id, global_op_id=new_global_loop_end_id)
                        cdlt.insert_op(inner_op_end, inner_idx)

                    ## end updates to end loop block

                    cdlt.loop_param_map[inner_op.op_str] = cdlt.loop_param_map[op.op_str]

                    if cdlt.loop_param_map[op.op_str] not in outer_loop_map:
                        outer_loop_map[cdlt.loop_param_map[op.op_str]] = op.op_str


                else:
                    assert op.op_type == 'compute', f"Invalid op type: {op.op_type}"
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
                    op.update_operand_indices(dep_mapping)
                    inner_idx = target_idx

                cdlt.insert_op(inner_op, inner_idx)

                if op.op_type == "loop" and outer_loop_map[cdlt.loop_param_map[op.op_str]] != op.op_str:
                    old_op = cdlt.ops.pop(cdlt.ops.index(op))
                    loop_replacement_map[old_op.op_str] = outer_loop_map[cdlt.loop_param_map[op.op_str]]
        all_deps.update(dep_mapping)
    cdlt = update_dependencies(cdlt, loop_replacement_map)

    cdlt = update_data_movements(cdlt)

    cdlt = update_temporary_data_moves(cdlt)

    if checkpoint_file is not None:
        store_tile_checkpoint(cdlt, checkpoint_file)

    return cdlt

def separate_simd_sa_ops(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet') -> 'Codelet':

    return cdlt

def hoist(program, node: pm.Node, cdlt: 'Codelet') -> 'Codelet':
    for o in cdlt.ops:
        if o.op_type == "loop" or o.op_type == "loop_end":
            continue
        i = cdlt.ops.index(o)
        all_deps = o.dependencies

        loop_name = cdlt.get_max_loop_dep(o)

        if loop_name is None:
            idx = i
            loop_level = 0
        elif len(all_deps) == 0:
            idx = cdlt.ops.index(cdlt.op_map[loop_name]) + 1
            loop_level = cdlt.op_map[loop_name].loop_level + 1
        else:
            idx = max([cdlt.ops.index(cdlt.op_map[dep]) for dep in all_deps])
            loop_level = cdlt.op_map[loop_name].loop_level + 1

        if (idx + (loop_level - o.loop_level) + 1) < i and cdlt.ops[idx + 1].loop_level <= loop_level:
            idx += 1
            cdlt.ops.insert(idx, cdlt.ops.pop(i))
            cdlt.ops[idx].loop_level = loop_level

    return cdlt
