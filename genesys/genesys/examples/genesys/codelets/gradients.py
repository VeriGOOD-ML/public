from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, DTYPE_MAP, FXP_CONFIGS
from . import add_simd_constraint, range_from_cfg, add_simd_tile_constraint


def sgd1d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("sgd1d") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [C], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [C], default_dtype=acc_dtype)
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [C], default_dtype=acc_dtype)
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")

        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='lr')
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='momentum')

        cdlt.configure("start", "IMM", immediate_value=lr)
        cdlt.configure("start", "IMM", immediate_value=momentum)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [C], default_dtype=acc_dtype)
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        # lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        # momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(C) as c:
            cdlt.transfer(param, ["DRAM", "VMEM1"])
            cdlt.transfer(grad, ["DRAM", "VMEM2"])
            updated_param.set_write_destination("VMEM1")
            itemp1.set_write_destination("VMEM2")
            itemp2.set_write_destination("VMEM1")
            cdlt.compute("MUL", [param[c], momentum_op], [itemp1[c]], target="SIMD")
            cdlt.compute("MUL", [grad[c], lr_op], [itemp2[c]], target="SIMD")
            cdlt.compute("SUB", [itemp1[c], itemp2[c]], [updated_param[c]], target="SIMD")
            cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt



def sgd2d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='lr')
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='momentum')

        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        cdlt.configure("start", "IMM", immediate_value=lr)
        cdlt.configure("start", "IMM", immediate_value=momentum)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(param, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM2"])
                updated_param.set_write_destination("VMEM1")
                itemp1.set_write_destination("VMEM2")
                itemp2.set_write_destination("VMEM1")
                cdlt.compute("MUL", [param[n, c], momentum_op], [itemp1[n, c]], target="SIMD")
                cdlt.compute("MUL", [grad[n, c], lr_op], [itemp2[n, c]], target="SIMD")
                cdlt.compute("SUB", [itemp1[n, c], itemp2[n, c]], [updated_param[n, c]], target="SIMD")
                cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def sgd3d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd3d") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        param = cdlt.create_operand_template("param", OP_DTYPES, [C, H, W], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [C, H, W], default_dtype=acc_dtype)
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(H) as h:
                with cdlt.loop(W) as w:
                    cdlt.transfer(param, ["DRAM", "VMEM1"])
                    cdlt.transfer(grad, ["DRAM", "VMEM2"])
                    updated_param.set_write_destination("VMEM1")
                    cdlt.compute("ADD", [param[c, h, w], grad[c, h, w]], [updated_param[c, h, w]], target="SIMD")
                    cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def sgd4d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd4d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='lr')
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='momentum')

        cdlt.configure("start", "IMM", immediate_value=lr)
        cdlt.configure("start", "IMM", immediate_value=momentum)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(param, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        updated_param.set_write_destination("VMEM1")
                        itemp1.set_write_destination("VMEM2")
                        itemp2.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [param[n, c, h, w], momentum_op], [itemp1[n, c]], target="SIMD")
                        cdlt.compute("MUL", [grad[n, c, h, w], lr_op], [itemp2[n, c]], target="SIMD")
                        cdlt.compute("SUB", [itemp1[n, c], itemp2[n, c]], [updated_param[n, c, h, w]], target="SIMD")
                        cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

#
def batchnorm_grad_(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=acc_dtype)
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=acc_dtype)
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=acc_dtype)
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=acc_dtype)
        xhat = cdlt.create_operand_template("xhat", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)

        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        scale_grad = cdlt.create_operand_template("scale_grad", OP_DTYPES, [C], default_dtype=acc_dtype)
        offset_grad = cdlt.create_operand_template("offset_grad", OP_DTYPES, [C], default_dtype=acc_dtype)

        cdlt.set_inputs([data, scale, offset, mean, istd, grad])
        cdlt.set_outputs([data_grad, scale_grad, offset_grad])

        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=acc_dtype)
        temp1.start_location = "VMEM1"
        temp1.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp1)

        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp2.start_location = "VMEM1"
        temp2.set_write_destination("VMEM1")
        cdlt.add_temp_operand(temp2)

        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp3.start_location = "VMEM1"
        temp3.set_write_destination("VMEM1")

        temp4 = cdlt.create_operand_template("temp4", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp4.start_location = "VMEM1"
        temp4.set_write_destination("VMEM1")

        temp5 = cdlt.create_operand_template("temp5", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp5.start_location = "VMEM1"
        temp5.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp3)
        cdlt.add_temp_operand(temp4)
        cdlt.add_temp_operand(temp5)

        numer = cdlt.create_operand_template("numer", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(xhat)
        cdlt.add_temp_operand(numer)
        denom = cdlt.dummy_op("denom",
                              cdlt.node.inputs[0].shape[0] * cdlt.node.inputs[0].shape[2] * cdlt.node.inputs[0].shape[
                                  3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')
        cdlt.configure("start", "IMM", immediate_value=denom)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(offset_grad, ["DRAM", "VMEM2"])
                        cdlt.transfer(mean, ["DRAM", "VMEM2"])
                        cdlt.transfer(istd, ["DRAM", "VMEM2"])
                        cdlt.transfer(scale_grad, ["DRAM", "VMEM2"])
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        scale_grad.set_write_destination("VMEM1")
                        offset_grad.set_write_destination("VMEM1")
                        numer.set_write_destination("VMEM1")
                        xhat.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [data[n, c, h, w], mean[c]], [numer[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [numer[n, c, h, w], istd[c]], [xhat[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n, c, h, w], grad[n, c, h, w]], [numer[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [scale_grad[c], numer[n, c, h, w]], [scale_grad[c]], target="SIMD")
                        cdlt.compute("ADD", [grad[n, c, h, w], offset_grad[c]], [offset_grad[c]], target="SIMD")

            with cdlt.loop(N) as n1:
                with cdlt.loop(H) as h1:
                    with cdlt.loop(W) as w1:
                        cdlt.transfer(scale, ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [scale[c], istd[c]], [temp1[c]], target="SIMD")
                        cdlt.compute("DIV", [temp1[c], denom_op], [temp1[c]], target="SIMD")
                        cdlt.compute("MUL", [denom_op, grad[n, c, h, w]], [temp2[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n1, c, h1, w1], scale_grad[c]], [temp3[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp2[n1, c, h1, w1], temp3[n1, c, h1, w1]], [temp4[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp4[n1, c, h1, w1], offset_grad[c]], [temp5[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("MUL", [temp1[c], temp5[n1, c, h1, w1]], [data_grad[n1, c, h1, w1]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
                        cdlt.transfer(offset_grad, ["VMEM1", "DRAM"])
                        cdlt.transfer(scale_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def batchnorm_grad_x_mu(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_x_mu") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        x = cdlt.create_operand_template("x", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        x_mu = cdlt.create_operand_template("x_mu", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([x, mean])
        cdlt.set_outputs([x_mu])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(x, ["DRAM", "VMEM1"])
                        cdlt.transfer(mean, ["DRAM", "VMEM2"])
                        x_mu.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [x[n, c, h, w], mean[c]], [x_mu[n, c, h, w]], target="SIMD")
                        cdlt.transfer(x_mu, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_inv_std(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_inv_std") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[0])
        var = cdlt.create_operand_template("var", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        inv_std = cdlt.create_operand_template("inv_std", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([var])
        cdlt.set_outputs([inv_std])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        eps = cdlt.dummy_op('eps', 0.0001)
        eps_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="eps")
        cdlt.configure("start", "IMM", immediate_value=eps)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            cdlt.transfer(var, ["DRAM", "VMEM1"])
            inv_std.set_write_destination("VMEM1")
            cdlt.compute("INV_SQRT", [var[c,]], [inv_std[c]], target="SIMD")
            cdlt.transfer(inv_std, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_xhat(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_xhat") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        x_mu = cdlt.create_operand_template("x_mu", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        inv_std = cdlt.create_operand_template("inv_std", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        x_hat = cdlt.create_operand_template("x_hat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([x_mu, inv_std])
        cdlt.set_outputs([x_hat])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(x_mu, ["DRAM", "VMEM1"])
                        cdlt.transfer(inv_std, ["DRAM", "VMEM2"])
                        x_hat.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [x_mu[n, c, h, w], inv_std[c]], [x_hat[n, c, h, w]], target="SIMD")
                        cdlt.transfer(x_hat, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_dx_rhs(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_dx_rhs") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        gy = cdlt.create_operand_template("gy", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        scaled_gy = cdlt.create_operand_template("scaled_gy", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dx_rhs = cdlt.create_operand_template("dx_rhs", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([gy, scaled_gy])
        cdlt.set_outputs([dx_rhs])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(gy, ["DRAM", "VMEM1"])
                        cdlt.transfer(scaled_gy, ["DRAM", "VMEM2"])
                        dx_rhs.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [gy[n, c, h, w], scaled_gy[n, c, h, w]], [dx_rhs[n, c, h, w]], target="SIMD")
                        cdlt.transfer(dx_rhs, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_gamma_inv_std(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_gamma_inv_std") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[0])

        gamma = cdlt.create_operand_template("gamma", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        inv_std = cdlt.create_operand_template("inv_std", OP_DTYPES, [C],
                                                 default_dtype=DTYPE_MAP[acc_dtype])
        gam_mul_inv_std = cdlt.create_operand_template("gam_mul_inv_std", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([gamma, inv_std])
        cdlt.set_outputs([gam_mul_inv_std])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            cdlt.transfer(gamma, ["DRAM", "VMEM1"])
            cdlt.transfer(inv_std, ["DRAM", "VMEM2"])
            gam_mul_inv_std.set_write_destination("VMEM1")
            cdlt.compute("MUL", [gamma[c], inv_std[c]], [gam_mul_inv_std[c]],
                         target="SIMD")
            cdlt.transfer(gam_mul_inv_std, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_scaled_gy(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_scaled_gy") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        dg_mul_xhat = cdlt.create_operand_template("dg_mul_xhat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dbeta = cdlt.create_operand_template("dbeta", OP_DTYPES, [C],
                                                 default_dtype=DTYPE_MAP[acc_dtype])
        scaled_gy = cdlt.create_operand_template("scaled_gy", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([dg_mul_xhat, dbeta])
        cdlt.set_outputs([scaled_gy])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        inv_m = cdlt.dummy_op("inv_m", cdlt.node.inputs[0].shape[0]*cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3])
        inv_m_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="inv_m")

        cdlt.configure("start", "IMM", immediate_value=inv_m)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(dg_mul_xhat, ["DRAM", "VMEM1"])
                        cdlt.transfer(dbeta, ["DRAM", "VMEM2"])
                        scaled_gy.set_write_destination("VMEM1")
                        cdlt.compute("ADD", [dg_mul_xhat[n, c, h, w], dbeta[c]], [scaled_gy[n, c, h, w]],
                                     target="SIMD")
                        cdlt.compute("MUL", [scaled_gy[n, c, h, w], inv_m_op], [scaled_gy[n, c, h, w]], target="SIMD")
                        cdlt.transfer(scaled_gy, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_dbeta(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_dbeta") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dbeta = cdlt.create_operand_template("dbeta", OP_DTYPES, [C],
                                               default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([grad])
        cdlt.set_outputs([dbeta])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        zero_op = cdlt.dummy_op('zero', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "IMM", immediate_value=zero_op)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(grad, ["DRAM", "VMEM1"])
                        cdlt.transfer(dbeta, ["DRAM", "VMEM2"])
                        dbeta.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [grad[n,c,h,w], dbeta[c]], [dbeta[c]],
                                     target="SIMD")
                        cdlt.transfer(dbeta, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["N", "H", "W"])

    return cdlt

def batchnorm_grad_dgamma_xhat(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_dgamma_xhat") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        x_hat = cdlt.create_operand_template("x_hat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dgamma = cdlt.create_operand_template("dgamma", OP_DTYPES, [C],
                                               default_dtype=DTYPE_MAP[acc_dtype])
        dg_mul_xhat = cdlt.create_operand_template("dg_mul_xhat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])


        cdlt.set_inputs([grad, x_hat])
        cdlt.set_outputs([dgamma, dg_mul_xhat])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        zero_op = cdlt.dummy_op('zero', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "IMM", immediate_value=zero_op)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(grad, ["DRAM", "VMEM1"])
                        cdlt.transfer(x_hat, ["DRAM", "VMEM2"])
                        cdlt.transfer(dgamma, ["DRAM", "VMEM1"])

                        dgamma.set_write_destination("VMEM1")
                        dg_mul_xhat.set_write_destination("VMEM2")
                        cdlt.compute("MACC", [grad[n,c,h,w], x_hat[n,c,h,w], dgamma[c]], [dgamma[c]],
                                     target="SIMD")
                        cdlt.compute("MUL", [dgamma[c], x_hat[n, c, h, w]], [dg_mul_xhat[n, c, h, w]], target="SIMD")
                        cdlt.transfer(dgamma, ["VMEM1", "DRAM"])
                        cdlt.transfer(dg_mul_xhat, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["N", "H", "W"])
    return cdlt


def batchnorm_grad_dgamma(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad_dgamma") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        x_hat = cdlt.create_operand_template("x_hat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dgamma = cdlt.create_operand_template("dgamma", OP_DTYPES, [C],
                                               default_dtype=DTYPE_MAP[acc_dtype])


        cdlt.set_inputs([grad, x_hat])
        cdlt.set_outputs([dgamma])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        zero_op = cdlt.dummy_op('zero', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="zero")

        cdlt.configure("start", "IMM", immediate_value=zero_op)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(grad, ["DRAM", "VMEM1"])
                        cdlt.transfer(x_hat, ["DRAM", "VMEM2"])
                        cdlt.transfer(dgamma, ["DRAM", "VMEM1"])
                        dgamma.set_write_destination("VMEM1")
                        cdlt.compute("MACC", [grad[n,c,h,w], x_hat[n,c,h,w], dgamma[c]], [dgamma[c]],
                                     target="SIMD")
                        cdlt.transfer(dgamma, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["N", "H", "W"])
    return cdlt

def batchnorm_grad_dgamma_mul_xhat(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"

    with CodeletTemplate("batchnorm_grad_dgamma_mul_xhat") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        x_hat = cdlt.create_operand_template("x_hat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dgamma = cdlt.create_operand_template("dgamma", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])

        dg_mul_xhat = cdlt.create_operand_template("dg_mul_xhat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([x_hat, dgamma])
        cdlt.set_outputs([dg_mul_xhat])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(x_hat, ["DRAM", "VMEM2"])
                        cdlt.transfer(dgamma, ["DRAM", "VMEM1"])
                        dg_mul_xhat.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [x_hat[n, c, h, w], dgamma[c]], [dg_mul_xhat[n, c, h, w]], target="SIMD")
                        cdlt.transfer(dg_mul_xhat, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad_dx(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"

    with CodeletTemplate("batchnorm_grad_dx") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        g_mul_istd = cdlt.create_operand_template("inv_std", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        dx_rhs = cdlt.create_operand_template("dx_rhs", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        dx = cdlt.create_operand_template("dx", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([dx_rhs, g_mul_istd])
        cdlt.set_outputs([dx])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(dx_rhs, ["DRAM", "VMEM1"])
                        cdlt.transfer(g_mul_istd, ["DRAM", "VMEM2"])
                        dx.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [dx_rhs[n, c, h, w], g_mul_istd[c]], [dx[n, c, h, w]], target="SIMD")
                        cdlt.transfer(dx, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

def batchnorm_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=acc_dtype)
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=acc_dtype)
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=acc_dtype)
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=acc_dtype)
        xhat = cdlt.create_operand_template("xhat", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)

        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        scale_grad = cdlt.create_operand_template("scale_grad", OP_DTYPES, [C], default_dtype=acc_dtype)
        offset_grad = cdlt.create_operand_template("offset_grad", OP_DTYPES, [C], default_dtype=acc_dtype)

        cdlt.set_inputs([data, scale, offset, mean, istd, grad])
        cdlt.set_outputs([data_grad, scale_grad, offset_grad])

        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=acc_dtype)
        temp1.start_location = "VMEM1"
        temp1.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp1)

        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp2.start_location = "VMEM1"
        temp2.set_write_destination("VMEM1")
        cdlt.add_temp_operand(temp2)

        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp3.start_location = "VMEM1"
        temp3.set_write_destination("VMEM1")

        temp4 = cdlt.create_operand_template("temp4", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp4.start_location = "VMEM1"
        temp4.set_write_destination("VMEM1")

        temp5 = cdlt.create_operand_template("temp5", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        temp5.start_location = "VMEM1"
        temp5.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp3)
        cdlt.add_temp_operand(temp4)
        cdlt.add_temp_operand(temp5)

        numer = cdlt.create_operand_template("numer", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(xhat)
        cdlt.add_temp_operand(numer)
        denom = cdlt.dummy_op("denom",
                              cdlt.node.inputs[0].shape[0] * cdlt.node.inputs[0].shape[2] * cdlt.node.inputs[0].shape[
                                  3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')
        cdlt.configure("start", "IMM", immediate_value=denom)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(offset_grad, ["DRAM", "VMEM2"])
                        cdlt.transfer(mean, ["DRAM", "VMEM2"])
                        cdlt.transfer(istd, ["DRAM", "VMEM2"])
                        cdlt.transfer(scale_grad, ["DRAM", "VMEM2"])
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        scale_grad.set_write_destination("VMEM1")
                        offset_grad.set_write_destination("VMEM1")
                        data.set_write_destination("VMEM1")
                        xhat.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [data[n, c, h, w], mean[c]], [data[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [data[n, c, h, w], istd[c]], [xhat[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n, c, h, w], grad[n, c, h, w]], [data[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [scale_grad[c], data[n, c, h, w]], [scale_grad[c]], target="SIMD")
                        cdlt.compute("ADD", [grad[n, c, h, w], offset_grad[c]], [offset_grad[c]], target="SIMD")

            with cdlt.loop(N) as n1:
                with cdlt.loop(H) as h1:
                    with cdlt.loop(W) as w1:
                        cdlt.transfer(scale, ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [scale[c], istd[c]], [temp1[c]], target="SIMD")
                        cdlt.compute("DIV", [temp1[c], denom_op], [temp1[c]], target="SIMD")
                        cdlt.compute("MUL", [denom_op, grad[n, c, h, w]], [temp2[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n1, c, h1, w1], scale_grad[c]], [temp3[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp2[n1, c, h1, w1], temp3[n1, c, h1, w1]], [temp4[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp4[n1, c, h1, w1], offset_grad[c]], [temp5[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("MUL", [temp1[c], temp5[n1, c, h1, w1]], [data_grad[n1, c, h1, w1]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
                        cdlt.transfer(offset_grad, ["VMEM1", "DRAM"])
                        cdlt.transfer(scale_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


# def flatten_grad(hag: ArchitectureNode):
#     inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
#     acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
#     with CodeletTemplate("flatten_grad") as cdlt:
#
#         N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
#         C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
#         H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
#         W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
#
#         data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
#         grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
#         out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
#         cdlt.set_inputs([data, grad])
#         cdlt.set_outputs([out])
#         cdlt.configure("end", "SIMD")
#
#     return cdlt
def flatten_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("flatten_grad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

    return cdlt

def relu_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("relu_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [data[n, c, h, w], grad[n, c, h, w]], [data_grad[n, c, h, w]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def relu_grad2d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("relu_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("RELU", [data[n, c], grad[n, c]], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_tanh_grad(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_tanh_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        one_val = cdlt.dummy_op("one", 1, dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='one')
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=acc_dtype)
        temp1.start_location = "VMEM1"
        cdlt.add_temp_operand(temp1)
        # one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=one_val)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM1"])
                        data.set_write_destination("VMEM1")
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [data[n, c, h, w], data[n, c, h, w]], [data[n, c, h, w]], target="SIMD")
                        one_op.set_write_destination("VMEM1")
                        one_op.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [one_op, data[n, c, h, w]], [temp1], target="SIMD")
                        cdlt.compute("MUL", [grad[n, c, h, w], temp1], [data_grad[n, c, h, w]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_tanh_grad2d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("elem_tanh_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        one_val = cdlt.dummy_op("one", 1, dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='one')
        # one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=acc_dtype)
        temp1.start_location = "VMEM1"
        #
        cdlt.add_temp_operand(temp1)

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=one_val)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                data.set_write_destination("VMEM1")
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("MUL", [data[n, c], data[n, c]], [data[n, c]], target="SIMD")
                one_op.set_write_destination("VMEM1")
                temp1.set_write_destination("VMEM1")
                cdlt.compute("SUB", [one_op, data[n, c]], [temp1], target="SIMD")
                cdlt.compute("MUL", [grad, temp1], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def max_pool_grad(hag: ArchitectureNode):
    #
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    with CodeletTemplate("max_pool_grad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        data = cdlt.create_operand_template("max_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("max_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)

        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        min_val, _ = range_from_cfg(FXP_CONFIGS[acc_dtype_name])
        min_val_op = cdlt.dummy_op('min_val', min_val)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        min_val_temp = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="min_val")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=min_val_op)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw], grad[n,c,y,x]], [data_grad[n, c, y*sy + kh, x*sx + kw]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def average_pool_grad(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    with CodeletTemplate("average_pool_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("avg_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        #
        data_grad = cdlt.create_operand_template("avg_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])

        zero_val = cdlt.dummy_op("zero", cdlt.node.kwargs['minval'], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='zero')

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_val)
        sy = cdlt.dummy_op('sy', cdlt.node.stride[0])
        sx = cdlt.dummy_op('sx', cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw],
                                                     grad[n, c, y, x]],
                                             [data_grad[n, c, y*sy + kh, x*sx + kw]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def global_average_pool_grad(hag: ArchitectureNode):

    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    with CodeletTemplate("global_average_pool_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        #
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        zero_val = cdlt.dummy_op("zero", 0, dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='zero')
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_val)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(IH) as iy:
                    with cdlt.loop(IW) as ix:
                        with cdlt.loop(OH) as oy:
                            with cdlt.loop(OW) as ox:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MEAN", [data[n, c, iy, ix], grad[n, c, oy, ox]], [data_grad[n, c, iy, ix]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def cross_entropy_loss_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("cross_entropy_loss_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        target = cdlt.create_operand_template("target", OP_DTYPES, [N], default_dtype=acc_dtype)
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=acc_dtype)

        cdlt.set_inputs([data, target])
        cdlt.set_outputs([data_grad])

        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(target, ["DRAM", "VMEM2"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("SUB", [data[n, c], target[n]], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "N")

    return cdlt

def load_gradient_cdlts(cfg):

    GRADIENT_CDLTS = {
        'average_pool_grad': average_pool_grad,
        "cross_entropy_loss_grad": cross_entropy_loss_grad,
        'elem_tanh_grad': elem_tanh_grad,
        'elem_tanh_grad2d': elem_tanh_grad2d,
        "flatten_grad": flatten_grad,
        'global_average_pool_grad': global_average_pool_grad,
        'max_pool_grad': max_pool_grad,
        'relu_grad2d': relu_grad2d,
        'relu_grad': relu_grad,
        "sgd1d": sgd1d,
        "sgd2d": sgd2d,
        "sgd3d": sgd3d,
        "sgd4d": sgd4d,
        # "batchnorm_grad": batchnorm_grad,
        ### BATCH NORM SUB-CODELETS:
        "batchnorm_grad_x_mu": batchnorm_grad_x_mu,
        "batchnorm_grad_inv_std": batchnorm_grad_inv_std,
        "batchnorm_grad_xhat": batchnorm_grad_xhat,
        "batchnorm_grad_dx_rhs": batchnorm_grad_dx_rhs,
        "batchnorm_grad_gamma_inv_std": batchnorm_grad_gamma_inv_std,
        "batchnorm_grad_scaled_gy": batchnorm_grad_scaled_gy,
        "batchnorm_grad_dbeta": batchnorm_grad_dbeta,
        "batchnorm_grad_dgamma_xhat": batchnorm_grad_dgamma_xhat,
        "batchnorm_grad_dgamma": batchnorm_grad_dgamma,
        "batchnorm_grad_dgamma_mul_xhat": batchnorm_grad_dgamma_mul_xhat,
        "batchnorm_grad_dx": batchnorm_grad_dx,
        #
    }
    return GRADIENT_CDLTS