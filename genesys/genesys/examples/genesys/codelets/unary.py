from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from codelets.templates.operation_template import OperationTemplate
from examples.genesys import OP_DTYPES, DTYPE_MAP
from . import range_from_cfg, add_simd_constraint, add_flex_simd_constraints
from functools import partial
#
def elem_unary_nd(cdlt_name, instr_name, num_dims, imm_val, hag):
    DIM_NAMES = ["N", "C", "H", "W"]
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate(cdlt_name) as cdlt:

        op1_dims = []
        assert num_dims <= len(DIM_NAMES)
        for i in range(num_dims):
            dim = cdlt.dummy_op(DIM_NAMES[i], cdlt.node.inputs[0].shape[i])
            op1_dims.append(dim)

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, op1_dims, default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, op1_dims, default_dtype=acc_dtype)

        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        if isinstance(imm_val, int):
            imm_dummy = cdlt.dummy_op('imm_dummy', imm_val)
            param = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='imm_dummy')
            cdlt.configure("start", "IMM", immediate_value=imm_dummy)
        elif isinstance(imm_val, str):
            dummy_imm = cdlt.dummy_op(imm_val, cdlt.node.kwargs[imm_val], dtype=acc_dtype_name)
            param = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name=imm_val)
            cdlt.configure("start", "IMM", immediate_value=dummy_imm)
        else:
            assert imm_val is None
            param = None
        loops = []
        op1_indices = []
        for i, d in enumerate(op1_dims):
            l = cdlt.loop(d)
            loops.append(l)
            op1_indices.append(l)

            OperationTemplate.loop_ctxt_level += 1
            OperationTemplate.loop_stack.append(l.loop_id)
            OperationTemplate.loop_ctx_dependencies.append(l.op_str)

        cdlt.transfer(op1, ["DRAM", "VMEM1"])
        out.set_write_destination("VMEM2")
        if param is not None:
            cdlt.compute(instr_name, [op1[tuple(op1_indices)], param], [out[tuple(op1_indices)]], target="SIMD")
        else:
            cdlt.compute(instr_name, [op1[tuple(op1_indices)]], [out[tuple(op1_indices)]], target="SIMD")

        cdlt.transfer(out, ["VMEM2", "DRAM"])

        for d in reversed(op1_dims):
            OperationTemplate.loop_ctxt_level -= 1
            OperationTemplate.loop_stack.pop()
            OperationTemplate.loop_ctx_dependencies.pop()

        cdlt.configure("end", "SIMD")

        cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def coarse_flatten(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("coarse_flatten") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def coarse_flatten2d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("coarse_flatten2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt

def coarse_flatten3d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("coarse_flatten3d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_transpose(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("tensor_transpose") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")


        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        pass
    return cdlt


def elem_sqrt(num_dims, hag):
    if num_dims < 4:
        name = f"elem_sqrt{num_dims}d"
    else:
        name = f"elem_sqrt"
    DIM_NAMES = ["N", "C", "H", "W"]
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate(name) as cdlt:
        op1_dims = []
        assert num_dims <= len(DIM_NAMES)
        for i in range(num_dims):
            dim = cdlt.dummy_op(DIM_NAMES[i], cdlt.node.inputs[0].shape[i])
            op1_dims.append(dim)

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, op1_dims, default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, op1_dims, default_dtype=acc_dtype)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        cdlt.configure("end", "SIMD")


    return cdlt

def elem_cast(hag):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_cast") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_cast", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_cast2d(hag):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_cast2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("RELU", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_exp(hag):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_exp") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_exp", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("EXP", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def inv_sqrt(hag):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("inv_sqrt") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_exp", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("INV_SQRT", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_tanh(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_tanh") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out_tanh", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "IMM", immediate_value=16, index=0)
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("TANH", [op1[n, c, h, w], param], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])

    return cdlt


def elem_tanh2d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_tanh2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_tanh", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_outputs([out])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param_op = cdlt.dummy_op("param", 16)
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='param')

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=param_op)
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("TANH", [op1[n, c], param], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_ceil2d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_ceil2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("CEIL", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
        cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C"])
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_pow2d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_pow2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        exp = cdlt.dummy_op("exp", cdlt.node.kwargs['exp'], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.dummy_op('zero', 0)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("MOVE", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.compute("POW", [op1[n, c], out[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_pow1d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_pow1d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N], default_dtype=acc_dtype)
        exp = cdlt.dummy_op("exp", cdlt.node.kwargs['exp'], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.dummy_op('zero', 0)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        with cdlt.loop(N) as n:
            cdlt.transfer(op1, ["DRAM", "VMEM1"])
            out.set_write_destination("VMEM2")
            cdlt.compute("MOVE", [op1[n]], [out[n]], target="SIMD")
            cdlt.compute("POW", [op1[n], out[n]], [out[n]], target="SIMD")
            cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt

def elem_pow3d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_pow3d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        exp = cdlt.dummy_op("exp", cdlt.node.kwargs['exp'], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        zero_op = cdlt.dummy_op('zero', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    cdlt.transfer(op1, ["DRAM", "VMEM1"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("MOVE", [op1[n, c, h]], [out[n, c,h]], target="SIMD")
                    cdlt.compute("POW", [op1[n, c, h], out[n, c,h ]], [out[n, c,h ]], target="SIMD")
                    cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt

def tensor_transpose2d(hag: ArchitectureNode):
    #
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("tensor_transpose2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [C, N], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        # axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")

        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("TRANSPOSE", [data[n, c]], [out[c, n]], target="SIMD")
            cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")


    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def tensor_transpose3d(hag: ArchitectureNode):
    #
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("tensor_transpose3d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, H, C], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        # axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(H) as h:
                with cdlt.loop(C) as c:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("TRANSPOSE", [data[n, c, h]], [out[n, h, c]], target="SIMD")
                    cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt.add_compilation_param("LEVEL1_hint", f"splits['H'] == 1 or splits['C'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "H")

    return cdlt

def tensor_transpose4d(hag: ArchitectureNode):
    #
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("tensor_transpose4d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, W, H], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        # axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")


        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(W) as w:
                    with cdlt.loop(H) as h:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("TRANSPOSE", [data[n, c, h, w]], [out[n, c, w, h]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    ## Changes for benchmarking
    # cdlt.add_compilation_param("LEVEL1_hint", f"splits['H'] == 1 or splits['W'] == 1")
    # cdlt = add_simd_constraint(hag, cdlt, "W")

    cdlt.add_compilation_param("LEVEL1_hint", f"splits['H'] == 1 or splits['W'] == 1 or splits['C'] == 1")
    cdlt = add_flex_simd_constraints(hag, cdlt, ["C", "W", "H"])

    #### End changes for benchmarking


    return cdlt

def clip(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_clip") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        minval = cdlt.dummy_op("min", cdlt.node.kwargs['minval'], dtype=acc_dtype_name)
        maxval = cdlt.dummy_op("max", cdlt.node.kwargs['maxval'], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        min_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='min')
        max_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='max')

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=minval)

        cdlt.configure("start", "IMM", immediate_value=maxval)
        # temp_res = cdlt.create_operand_template("partial", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp_res = cdlt.create_temp_operand([N, C, H, W], "VMEM1")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        temp_res.set_write_destination("VMEM1")
                        cdlt.compute("MAX", [op1[n, c, h, w], max_op], [temp_res[n, c, h, w]], target="SIMD")
                        out.set_write_destination("VMEM2")
                        cdlt.compute("MIN", [temp_res[n, c, h, w], min_op], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def load_unary_cdlts(cfg):

    UNARY_CODELETS = {
        "coarse_flatten": coarse_flatten,
        "coarse_flatten2d": coarse_flatten2d,
        "coarse_flatten3d": coarse_flatten3d,
        "elem_tanh": partial(elem_unary_nd, "elem_tanh", "TANH", 4, 16),
        "elem_tanh3d": partial(elem_unary_nd, "elem_tanh3d", "TANH", 3, 16),
        "elem_tanh2d": elem_tanh2d,
        # TODO: Check if this needs to be 'sigmoid'
        "elem_sigmoid": partial(elem_unary_nd, "elem_sigmoid", "SIGMOID", 4, 16),
        "leaky_relu": partial(elem_unary_nd, "leaky_relu", "LEAKY_RELU", 4, 'alpha'),
        "elem_clip": clip,
        "elem_ceil2d": elem_ceil2d,
        "elem_pow1d": elem_pow1d,
        "elem_pow2d": elem_pow2d,
        "elem_pow3d": elem_pow3d,
        "elem_exp": elem_exp,
        "relu": partial(elem_unary_nd, "relu", "RELU", 4, 16),
        "relu2d": partial(elem_unary_nd, "relu2d", "RELU", 2, 16),
        'tensor_transpose2d': tensor_transpose2d,
        'tensor_transpose3d': tensor_transpose3d,
        'tensor_transpose4d': tensor_transpose4d,
        'elem_cast': elem_cast,
        'elem_cast2d': elem_cast2d,
        "inv_sqrt": inv_sqrt,
        "elem_sqrt": partial(elem_unary_nd, "elem_sqrt", "SQRT", 4, None),
        "elem_sqrt2d": partial(elem_unary_nd, "elem_sqrt2d", "SQRT", 2, None),
        "elem_sqrt1d": partial(elem_unary_nd, "elem_sqrt1d", "SQRT", 1, None),
        # "elem_sqrt": partial(elem_sqrt, 4),
        # "elem_sqrt1d": partial(elem_sqrt, 1),
        # "elem_sqrt2d": partial(elem_sqrt, 2),
    }
    return UNARY_CODELETS