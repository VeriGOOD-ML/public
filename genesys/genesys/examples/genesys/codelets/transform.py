from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, DTYPE_MAP


def tensor_reshape4d2d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape4d2d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt

def tensor_reshape3d2d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape3d2d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt

def tensor_reshape2d3d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape2d3d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.outputs[0].shape[2])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_reshape4d3d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape4d3d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_reshape3d4d(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape3d4d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.outputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.outputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.outputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt

def tensor_squeeze(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_squeeze") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_resize(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("resize") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H1 = cdlt.dummy_op("H1", cdlt.node.inputs[0].shape[2])
        W1 = cdlt.dummy_op("W1", cdlt.node.inputs[0].shape[3])

        H2 = cdlt.dummy_op("H2", cdlt.node.outputs[0].shape[2])
        W2 = cdlt.dummy_op("W2", cdlt.node.outputs[0].shape[3])
        DIMS = cdlt.dummy_op('DIMS', cdlt.node.inputs[1].shape[0])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H1, W1], default_dtype=acc_dtype)
        scales = cdlt.create_operand_template("scale", OP_DTYPES, [DIMS], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H2, W2], default_dtype=acc_dtype)
        cdlt.set_inputs([op1, scales])
        cdlt.set_outputs([out])
    return cdlt

def tensor_pad(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("tensor_pad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt


def tensor_flip(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("tensor_flip") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

    return cdlt


def concat(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("concat") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        IC1 = cdlt.dummy_op("IC1", cdlt.node.inputs[0].shape[1])
        IC2 = cdlt.dummy_op("IC2", cdlt.node.inputs[1].shape[1])
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, IC1, H, W], default_dtype=acc_dtype)
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, IC2, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])

    return cdlt

def split(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("split") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])

        C1 = cdlt.dummy_op("C1", cdlt.node.outputs[0].shape[1])
        H1 = cdlt.dummy_op("H1", cdlt.node.outputs[0].shape[2])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H], default_dtype=acc_dtype)
        out1 = cdlt.create_operand_template("out1", OP_DTYPES, [N, C1, H1], default_dtype=acc_dtype)
        out2 = cdlt.create_operand_template("out2", OP_DTYPES, [N, C1, H1], default_dtype=acc_dtype)
        out3 = cdlt.create_operand_template("out3", OP_DTYPES, [N, C1, H1], default_dtype=acc_dtype)
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out1, out2, out3])

    return cdlt

def where(hag: ArchitectureNode):
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['ACC_WIDTH']}"]
    with CodeletTemplate("elem_where") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[1].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[1].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[1].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[1].shape[3])
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[0].shape[1])


        cond = cdlt.create_operand_template("cond", OP_DTYPES, [N, ONE, H, W], default_dtype=acc_dtype)
        x = cdlt.create_operand_template("x", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)

        cdlt.set_inputs([x, cond])
        cdlt.set_outputs([out])

    return cdlt

def load_transform_cdlts(cfg):

    TRANSFORM_CDLTS = {
        'tensor_reshape4d2d': tensor_reshape4d2d,
        'tensor_reshape4d3d': tensor_reshape4d3d,
        'tensor_reshape3d4d': tensor_reshape3d4d,
        'tensor_reshape3d2d': tensor_reshape3d2d,
        'tensor_reshape2d3d': tensor_reshape2d3d,
        'split': split,
        'tensor_squeeze': tensor_squeeze,
        'concat': concat,
        'resize': tensor_resize,
        'elem_where': where,
        # 'tensor_flip': tensor_flip,
        # 'tensor_pad': tensor_pad,
    }
    return TRANSFORM_CDLTS
