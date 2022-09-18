from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from codelets.templates.operation_template import OperationTemplate
from examples.genesys import OP_DTYPES, FXP_CONFIGS, DTYPE_MAP
from functools import partial

from . import add_simd_constraint, range_from_cfg


def reduce_mean(cdlt_name: str, ninput_dims, axis, hag: ArchitectureNode):
    DIM_NAMES = ["N", "C", "H", "W"]
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate(cdlt_name) as cdlt:
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[axis])

        input_dims = []
        out_dims = []
        all_dims = []
        for i in range(ninput_dims):
            dim = cdlt.dummy_op(DIM_NAMES[i], cdlt.node.inputs[0].shape[i])
            input_dims.append(dim)
            all_dims.append(dim)
            if i == axis:
                all_dims.pop()
                all_dims.insert(0, dim)
                out_dims.append(ONE)
            else:
                out_dims.append(dim)
        all_dims.append(ONE)
        data = cdlt.create_operand_template("data", OP_DTYPES, input_dims, default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, out_dims, default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[axis]), dtype=acc_dtype_name)
        axis_op = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero = cdlt.dummy_op('zero', 0)

        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero)
        cdlt.configure("start", "IMM", immediate_value=denom)

        loops = []
        input_indices = [None] * len(input_dims)
        out_indices = [None] * len(out_dims)
        for i, d in enumerate(reversed(all_dims)):
            l = cdlt.loop(d)
            loops.append(l)
            if d in input_dims:
                input_indices[input_dims.index(d)] = l
            if d in out_dims:
                out_indices[out_dims.index(d)] = l

            OperationTemplate.loop_ctxt_level += 1
            OperationTemplate.loop_stack.append(l.loop_id)
            OperationTemplate.loop_ctx_dependencies.append(l.op_str)
        assert all([i is not None for i in input_indices])
        assert all([i is not None for i in out_indices])
        cdlt.transfer(data, ["DRAM", "VMEM1"])
        # TODO: Zero out output data at compile time
        cdlt.transfer(out, ["DRAM", "VMEM2"])
        out.set_write_destination("VMEM2")
        cdlt.compute("ADD", [data[tuple(input_indices)], out[tuple(out_indices)]], [out[tuple(out_indices)]], target="SIMD")
        cdlt.compute("MUL", [out[tuple(out_indices)], denom_op], [out[tuple(out_indices)]], target="SIMD")
        cdlt.transfer(out, ["VMEM2", "DRAM"])

        for d in reversed(all_dims):
            OperationTemplate.loop_ctxt_level -= 1
            OperationTemplate.loop_stack.pop()
            OperationTemplate.loop_ctx_dependencies.pop()

        cdlt.configure("end", "SIMD")
        cdlt = add_simd_constraint(hag, cdlt, DIM_NAMES[ninput_dims - 1])
        # cdlt.add_compilation_param("LEVEL1_hint", f"splits['{DIM_NAMES[axis]}'] == 1")

    return cdlt


def reduce_min2d(hag: ArchitectureNode):
    #
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("reduce_min2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [ONE, C], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        _, max_val = range_from_cfg(FXP_CONFIGS[str(acc_dtype)])
        max_val_dummy = cdlt.dummy_op('max_val', max_val)
        max_val_temp = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="max_val")
        cdlt.configure("start", "IMM", immediate_value=max_val_dummy)

        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("MIN", [data[n, c], out[o, c]], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def reduce_mean2d(hag: ArchitectureNode):
    #
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("reduce_mean2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [ONE, C], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[0]), dtype=acc_dtype_name)
        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.dummy_op('zero', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        cdlt.configure("start", "IMM", immediate_value=denom)
        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("ADD", [data[n,c], out[o, c]], [out[o, c]], target="SIMD")
                    cdlt.compute("MUL", [out[o, c], denom_op], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

        ############ TESTING####################################
        denom = cdlt.dummy_op("denom1", 1/(cdlt.node.inputs[0].shape[0]), dtype=acc_dtype_name)

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        zero0 = cdlt.dummy_op('zero0', 0)
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='zero0')
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom1')

        cdlt.configure("start", "IMM", immediate_value=zero0)

        cdlt.configure("start", "IMM", immediate_value=denom)
        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("ADD", [data[n, c], out[o, c]], [out[o, c]], target="SIMD")
                    cdlt.compute("MUL", [out[o, c], denom_op], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
        ############ TESTING####################################

    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")

    return cdlt

def reduce_sum(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("reduce_sum") as cdlt:


        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [C], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("ADD", [data[n,c], data[n,c]], [out[c]], target="SIMD")
                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def reduce_mean3d(hag: ArchitectureNode):
    #
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("reduce_mean3d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [ONE, C], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[0]), dtype=acc_dtype_name)
        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "IMM", immediate_value=denom, index=1)
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("ADD", [data[n,c], out[o, c]], [out[o, c]], target="SIMD")
                    cdlt.compute("MUL", [out[o, c], denom_op], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")

    return cdlt

def load_reduce_cdlts(cfg):
    REDUCTION_CODELETS = {
        "reduce_sum": reduce_sum,
        "reduce_mean2d": reduce_mean2d,
        # "reduce_mean2d": partial(reduce_mean, 'reduce_mean2d', 2, 0),
        "reduce_mean3d": partial(reduce_mean, 'reduce_mean3d', 3, 1),
        # "reduce_mean3d": partial(reduce_mean, 'reduce_mean3d', 3, 2),
        "reduce_min2d": reduce_min2d,
    }
    return REDUCTION_CODELETS