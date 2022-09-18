from codelets.adl.operation import Operand
from codelets.codelet_impl.codelet import Codelet
from codelets.templates.operand_template import OperandTemplate
from codelets.templates.operation_template import OperationTemplate
from codelets.templates.codelet_template import CodeletTemplate
from codelets.adl.flex_param import FlexParam
from codelets.adl.graph import ArchitectureNode
from . import OP_DTYPES

def gemm(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["M", "N"], dtype=OP_DTYPES[0])
    weight = Operand("weight", OP_DTYPES, ["N", "P"], dtype=OP_DTYPES[0])
    bias = Operand("bias", OP_DTYPES, ["P"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["M", "P"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("gemm", [data, weight, bias], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "OBUF")
        with cdlt.loop(0, "P") as p:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "M") as m:
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(bias[p], ["DRAM", "BBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("N_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("P_hint2", f"size == {sys_array_dims[1]}")
    return cdlt



def gemm_no_bias(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["M", "N"], dtype=OP_DTYPES[0])
    weight = Operand("weight", OP_DTYPES, ["N", "P"], dtype=OP_DTYPES[0])
    out = Operand("out", OP_DTYPES, ["M", "P"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("gemm_no_bias", [data, weight], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with cdlt.loop(0, "P") as p:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "M") as m:
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("N_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("P_hint2", f"size == {sys_array_dims[1]}")
    return cdlt


def conv2d(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = Operand("data", OP_DTYPES, ["N", "IC", "IH", "IW"], dtype=OP_DTYPES[0])
    weight = Operand("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"], dtype=OP_DTYPES[0])
    out = Operand("out", OP_DTYPES, ["N", "OC", "OH", "OW"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("conv", [data, weight], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with cdlt.loop(0, "OC") as oc:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "IC") as ic:
                    with cdlt.loop(0, "KH") as kh:
                        with cdlt.loop(0, "KW") as kw:
                            with cdlt.loop(0, "OH") as y:
                                with cdlt.loop(0, "OW") as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(data[n, ic, y*"stride" + kh, x*"stride" + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["OC", "IC", "KH", "KW", "N", "OH", "OW"])
    # cdlt.add_compilation_param("LOOP_TILE_ORDER", ["KH", "KW", "OC", "IC", "N", "OH", "OW"])
    wbuf_elements = hag.get_subgraph_node("WBUF").num_elements
    obuf_elements = hag.get_subgraph_node("OBUF").num_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OH']*sizes['OC']"
    cdlt.add_compilation_param("LEVEL1_hint", f"{wbuf_index_size} <= {wbuf_elements} and {obuf_index_size} <= {obuf_elements}")
    cdlt.add_compilation_param("N_hint1", f"((size & (size - 1)) == 0)")
    cdlt.add_compilation_param("N_hint2", f"size == 1")
    cdlt.add_compilation_param("OH_hint2", f"size == 1")
    cdlt.add_compilation_param("OW_hint2", f"size == 1")
    cdlt.add_compilation_param("KH_hint2", f"size == 1")
    cdlt.add_compilation_param("KW_hint2", f"size == 1")
    cdlt.add_compilation_param("IC_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("OC_hint2", f"size == {sys_array_dims[1]}")
    cdlt.add_compilation_param("IC_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint1", f"size % {sys_array_dims[1]} == 0")

    return cdlt

def conv2d_bias(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = Operand("data", OP_DTYPES, ["N", "IC", "IH", "IW"], dtype=OP_DTYPES[0])
    weight = Operand("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"], dtype=OP_DTYPES[0])
    bias = Operand("bias", OP_DTYPES, ["OC"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "OC", "OH", "OW"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("conv_bias", [data, weight, bias], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")

        with cdlt.loop(0, "OC") as oc:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "IC") as ic:
                    with cdlt.loop(0, "KH") as kh:
                        with cdlt.loop(0, "KW") as kw:
                            with cdlt.loop(0, "OH") as y:
                                with cdlt.loop(0, "OW") as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(bias[oc], ["DRAM", "BBUF"])
                                    cdlt.transfer(data[n, ic, y*"stride" + kh, x*"stride" + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                                    # cdlt.compute("MVMUL", [data[n, ic, y*"stride" + kh, x*"stride" + kw], weight[oc, ic, kh, kw], bias[oc]], [out[n, oc, y, x]], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["OC", "IC", "OH", "OW", "N", "KH", "KW"])
    wbuf_elements = hag.get_subgraph_node("WBUF").num_elements
    obuf_elements = hag.get_subgraph_node("OBUF").num_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    # cdlt.add_compilation_param("LEVEL1_hint", f"{wbuf_index_size} <= {wbuf_elements} and {obuf_index_size} <= {obuf_elements}")
    cdlt.add_compilation_param("N_hint1", f"((size & (size - 1)) == 0)")
    cdlt.add_compilation_param("N_hint2", f"size == 1")
    # cdlt.add_compilation_param("OH_hint2", f"size == 1")
    # cdlt.add_compilation_param("OW_hint2", f"size == 1")
    cdlt.add_compilation_param("KH_hint2", f"size == 1")
    cdlt.add_compilation_param("KW_hint2", f"size == 1")
    cdlt.add_compilation_param("IC_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("OC_hint2", f"size == {sys_array_dims[1]}")
    cdlt.add_compilation_param("IC_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint1", f"size % {sys_array_dims[1]} == 0")
    return cdlt

def elem_add(hag: ArchitectureNode):
    op1 = Operand("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    op2 = Operand("op2", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("add_out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("elem_add", [op1, op2], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        # out.set_write_destination("OBUF")
                        cdlt.compute("ADD", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
                        # cdlt.transfer(out[n, c, h, w], ["OBUF", "DRAM"])
    return cdlt

def elem_add_grad(hag: ArchitectureNode):
    op1 = Operand("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    op2 = Operand("op2", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    op1_grad = Operand("op1_grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    op2_grad = Operand("op2_grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("elem_add_grad", [op1, op2, grad], [op1_grad, op2_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM2"])
                        op1_grad.set_write_destination("VMEM1")
                        op2_grad.set_write_destination("VMEM1")
                        cdlt.compute("MULADD", [op1, op2, grad], [op1_grad, op2_grad], target="SIMD")
                        cdlt.transfer(op1_grad[n, c, h, w], ["VMEM1", "DRAM"])
                        cdlt.transfer(op2_grad[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt


def sgd1d(hag: ArchitectureNode):
    param = Operand("param", OP_DTYPES, ["N"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N"], dtype=OP_DTYPES[2])
    updated_param = Operand("updated", OP_DTYPES, ["N"], dtype=OP_DTYPES[2])
    with Codelet("sgd1d", [param, grad], [updated_param], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            cdlt.transfer(param[n], ["DRAM", "VMEM1"])
            cdlt.transfer(grad[n], ["DRAM", "VMEM2"])
            updated_param.set_write_destination("VMEM1")
            cdlt.compute("ADD", [param, grad], [updated_param], target="SIMD")
            cdlt.transfer(updated_param[n], ["VMEM1", "DRAM"])
    return cdlt

def sgd2d(hag: ArchitectureNode):
    param = Operand("param", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    updated_param = Operand("updated", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    with Codelet("sgd2d", [param, grad], [updated_param], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                cdlt.transfer(param[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(grad[n, c], ["DRAM", "VMEM2"])
                updated_param.set_write_destination("VMEM1")
                cdlt.compute("ADD", [param, grad], [updated_param], target="SIMD")
                cdlt.transfer(updated_param[n, c], ["VMEM1", "DRAM"])
    return cdlt

def sgd3d(hag: ArchitectureNode):
    param = Operand("param", OP_DTYPES, ["C", "H", "W"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["C", "H", "W"], dtype=OP_DTYPES[2])
    updated_param = Operand("updated", OP_DTYPES, ["C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("sgd3d", [param, grad], [updated_param], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "C") as c:
            with cdlt.loop(0, "H") as h:
                with cdlt.loop(0, "W") as w:
                    cdlt.transfer(param[c, h, w], ["DRAM", "VMEM1"])
                    cdlt.transfer(grad[c, h, w], ["DRAM", "VMEM2"])
                    updated_param.set_write_destination("VMEM1")
                    cdlt.compute("ADD", [param, grad], [updated_param], target="SIMD")
                    cdlt.transfer(updated_param[c, h, w], ["VMEM1", "DRAM"])
    return cdlt

def sgd4d(hag: ArchitectureNode):
    param = Operand("param", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    updated_param = Operand("updated", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("sgd4d", [param, grad], [updated_param], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(param[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM2"])
                        updated_param.set_write_destination("VMEM1")
                        cdlt.compute("ADD", [param, grad], [updated_param], target="SIMD")
                        cdlt.transfer(updated_param[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt

def batch_norm(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    scale = Operand("scale", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    offset = Operand("offset", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("batch_norm", [data, scale, offset], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(scale[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(offset[c], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("MULADD", [data, scale, offset], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt


def batchnorm_grad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    scale = Operand("scale", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    offset = Operand("offset", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    mean = Operand("mean", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    var = Operand("var", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    scale_grad = Operand("scale_grad", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    offset_grad = Operand("offset_grad", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    with Codelet("batchnorm_grad", [data, scale, offset, mean, var, grad], [data_grad, scale_grad, offset_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(scale[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(offset[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(mean[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(var[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        scale_grad.set_write_destination("VMEM1")
                        offset_grad.set_write_destination("VMEM1")
                        cdlt.compute("MULADD", [data, scale, offset, mean, var, grad], [data_grad, scale_grad, offset_grad], target="SIMD")
                        cdlt.transfer(data_grad[n, c, h, w], ["VMEM1", "DRAM"])
                        cdlt.transfer(scale_grad[c], ["VMEM1", "DRAM"])
                        cdlt.transfer(offset_grad[c], ["VMEM1", "DRAM"])
    return cdlt

def coarse_flatten(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    with Codelet("coarse_flatten", [data], [out], hag) as cdlt:
        pass
    return cdlt

def tensor_transpose(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("tensor_transpose", [data], [out], hag) as cdlt:
        pass

    return cdlt

def tensor_reshape(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("tensor_reshape", [data], [out], hag) as cdlt:
        pass

    return cdlt

def tensor_pad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("tensor_pad", [data], [out], hag) as cdlt:
        pass
    return cdlt

def tensor_flip(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("tensor_flip", [data], [out], hag) as cdlt:
        pass

    return cdlt

def flatten_grad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("flatten_grad", [data, grad], [out], hag) as cdlt:
        pass
    return cdlt

def reduce_sum(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    out = Operand("out", OP_DTYPES, ["C"], dtype=OP_DTYPES[2])
    with Codelet("reduce_sum", [data], [out], hag) as cdlt:
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("ADD", [data, data], [out], target="SIMD")
                cdlt.transfer(out[c], ["VMEM1", "DRAM"])

    return cdlt

def cross_entropy_loss(hag: ArchitectureNode):
    res = Operand("res", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    target = Operand("target", OP_DTYPES, ["N", ], dtype=OP_DTYPES[2])
    loss = Operand("loss", OP_DTYPES, ["D"], dtype=OP_DTYPES[2])
    with Codelet("cross_entropy_loss", [res, target], [loss], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "D") as d:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "C") as c:
                    cdlt.transfer(res[n, c], ["DRAM", "VMEM1"])
                    cdlt.transfer(target[n], ["DRAM", "VMEM2"])
                    loss.set_write_destination("VMEM1")
                    cdlt.compute("ADD", [res, target], [loss], target="SIMD")
                    cdlt.transfer(loss[d], ["VMEM1", "DRAM"])
    return cdlt


def relu(hag: ArchitectureNode):
    op1 = Operand("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out_relu", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("relu", [op1], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])

    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])

    return cdlt

def relu_template(_):

    with CodeletTemplate("relu") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])

    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])
    return cdlt

def elem_tanh(hag: ArchitectureNode):
    op1 = Operand("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = Operand("out_tanh", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("elem_tanh", [op1], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("TANH", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])

    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])

    return cdlt

def elem_tanh2d(hag: ArchitectureNode):
    op1 = Operand("op1", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    out = Operand("out_tanh", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    with Codelet("elem_tanh2d", [op1], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                cdlt.transfer(op1[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("TANH", [op1], [out], target="SIMD")
                cdlt.transfer(out[n, c], ["VMEM1", "DRAM"])

    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C"])

    return cdlt

def relu_grad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("relu_grad", [data, grad], [data_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM1"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [data, grad], [data_grad], target="SIMD")
                        cdlt.transfer(data_grad[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt

def elem_tanh_grad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("elem_tanh_grad", [data, grad], [data_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "H") as h:
                    with cdlt.loop(0, "W") as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM1"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("TANH", [data, grad], [data_grad], target="SIMD")
                        cdlt.transfer(data_grad[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt

def elem_tanh_grad2d(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    with Codelet("elem_tanh_grad2d", [data, grad], [data_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(grad[n, c], ["DRAM", "VMEM1"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("TANH", [data, grad], [data_grad], target="SIMD")
                cdlt.transfer(data_grad[n, c], ["VMEM1", "DRAM"])
    return cdlt

# TODO: Implement valid operation sequence
def maxpool2d(hag: ArchitectureNode):
    #
    data = Operand("data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    #
    out = Operand("out", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("max_pool", [data], [out], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "KH") as kh:
                    with cdlt.loop(0, "KW") as kw:
                        with cdlt.loop(0, "OH") as y:
                            with cdlt.loop(0, "OW") as x:
                                cdlt.transfer(data[n, c, y*"sy" + kh, x*"sx" + kw], ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data, out], [out], target="SIMD")
                                cdlt.transfer(out[n, c, y, x], ["VMEM1", "DRAM"])
    return cdlt

def averagepool2d(hag: ArchitectureNode):
    #
    data = Operand("data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    #
    out = Operand("out", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("avg_pool", [data], [out], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        denom = FlexParam("denom", ["IH", "IW"], "IH*IW")
        cdlt.configure("start", "IMM", immediate_value=denom, index=0)
        cdlt.configure("start", "IMM", immediate_value=0, index=1)
        denom_op = cdlt.create_temp_operand([hag.get_subgraph_node("SIMD").dimensions[0]], "IMM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "KH") as kh:
                    with cdlt.loop(0, "KW") as kw:
                        with cdlt.loop(0, "OH") as y:
                            with cdlt.loop(0, "OW") as x:
                                cdlt.transfer(data[n, c, y*"sy" + kh, x*"sx" + kw], ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data, out], [out], target="SIMD")
                cdlt.compute("DIV", [out, denom_op], [out], target="SIMD")
                cdlt.transfer(out[n, c, y, x], ["VMEM1", "DRAM"])
    return cdlt

def max_pool_grad(hag: ArchitectureNode):
    #
    data = Operand("max_pool_data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    #
    data_grad = Operand("max_pool_data_grad", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("max_pool_grad", [data, grad], [data_grad], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "KH") as kh:
                    with cdlt.loop(0, "KW") as kw:
                        with cdlt.loop(0, "OH") as y:
                            with cdlt.loop(0, "OW") as x:
                                cdlt.transfer(data[n, c, y*"sy" + kh, x*"sx" + kw], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, y, x], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, y*"sy" + kh, x*"sx" + kw], ["VMEM1", "DRAM"])
    return cdlt


def average_pool_grad(hag: ArchitectureNode):
    #
    data = Operand("avg_pool_data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    #
    data_grad = Operand("avg_pool_data_grad", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("average_pool_grad", [data, grad], [data_grad], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "KH") as kh:
                    with cdlt.loop(0, "KW") as kw:
                        with cdlt.loop(0, "OH") as y:
                            with cdlt.loop(0, "OW") as x:
                                cdlt.transfer(data[n, c, y*"sy" + kh, x*"sx" + kw], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, y, x], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, y*"sy" + kh, x*"sx" + kw], ["VMEM1", "DRAM"])
    return cdlt

def global_average_pool_grad(hag: ArchitectureNode):
    #
    data = Operand("data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    #
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("global_average_pool_grad", [data, grad], [data_grad], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "IH") as iy:
                    with cdlt.loop(0, "IW") as ix:
                        with cdlt.loop(0, "OH") as oy:
                            with cdlt.loop(0, "OW") as ox:
                                cdlt.transfer(data[n, c, iy, ix], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, oy, ox], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MEAN", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, iy, ix], ["VMEM1", "DRAM"])
    return cdlt

# TODO: Implement valid operation sequence
def global_avg_pool(hag: ArchitectureNode):
    #
    data = Operand("data", OP_DTYPES, ["N", "C", "IH", "IW"], dtype=OP_DTYPES[2])
    #
    out = Operand("out", OP_DTYPES, ["N", "C", "OH", "OW"], dtype=OP_DTYPES[2])
    # # TODO: Add option to create operand
    with Codelet("global_avg_pool", [data], [out], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        denom = FlexParam("denom", ["IH", "IW"], "IH*IW")
        cdlt.configure("start", "IMM", immediate_value=denom, index=0)
        cdlt.configure("start", "IMM", immediate_value=0, index=1)
        denom_op = cdlt.create_temp_operand([hag.get_subgraph_node("SIMD").dimensions[0]], "IMM")
        with cdlt.loop(0, "N") as n:
            with cdlt.loop(0, "C") as c:
                with cdlt.loop(0, "IH") as iy:
                    with cdlt.loop(0, "IW") as ix:
                        with cdlt.loop(0, "OH") as oy:
                            with cdlt.loop(0, "OW") as ox:
                                cdlt.transfer(data[n, c, iy + oy, ix + ox], ["DRAM", "VMEM1"])
                                # TODO: Zero out output data at compile time
                                cdlt.transfer(out[n, c, oy, ox], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data, out], [out], target="SIMD")
                cdlt.compute("DIV", [out, denom_op], [out], target="SIMD")
                cdlt.transfer(out[n, c, oy, ox], ["VMEM2", "DRAM"])
    return cdlt

def cross_entropy_loss_grad(hag: ArchitectureNode):
    data = Operand("data", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    grad = Operand("grad", OP_DTYPES, ["D"], dtype=OP_DTYPES[2])
    target = Operand("target", OP_DTYPES, ["N", ], dtype=OP_DTYPES[2])
    data_grad = Operand("data_grad", OP_DTYPES, ["N", "C"], dtype=OP_DTYPES[2])
    with Codelet("cross_entropy_loss_grad", [data, target, grad], [data_grad], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with cdlt.loop(0, "D") as d:
            with cdlt.loop(0, "N") as n:
                with cdlt.loop(0, "C") as c:
                    cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                    cdlt.transfer(target[n], ["DRAM", "VMEM2"])
                    cdlt.transfer(grad[d], ["DRAM", "VMEM2"])
                    data_grad.set_write_destination("VMEM1")

                    cdlt.compute("ADD", [data, target, grad], [data_grad], target="SIMD")
                    cdlt.transfer(data_grad[n, c], ["VMEM1", "DRAM"])
    return cdlt

GENESYS_CODELETS = {
    "max_pool": maxpool2d,
    "avg_pool": averagepool2d,
    "global_avg_pool": global_avg_pool,
    "coarse_flatten": coarse_flatten,
    "conv_bias": conv2d_bias,
    "conv": conv2d,
    "gemm": gemm,
    "elem_add": elem_add,
    "elem_tanh": elem_tanh,
    "elem_tanh2d": elem_tanh2d,
    "relu": relu,
    "batch_norm": batch_norm,
    "batchnorm_grad": batchnorm_grad,
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_grad": cross_entropy_loss_grad,
    "flatten_grad": flatten_grad,
    "reduce_sum": reduce_sum,
    "sgd1d": sgd1d,
    "sgd2d": sgd2d,
    "sgd3d": sgd3d,
    "sgd4d": sgd4d,
    'elem_tanh_grad': elem_tanh_grad,
    'elem_tanh_grad2d': elem_tanh_grad2d,
    'elem_add_grad': elem_add_grad,
    'average_pool_grad': average_pool_grad,
    'max_pool_grad': max_pool_grad,
    'gemm_no_bias': gemm_no_bias,
    'relu_grad': relu_grad,
    'global_average_pool_grad': global_average_pool_grad,
    'tensor_transpose': tensor_transpose,
    'tensor_reshape': tensor_reshape,
    'tensor_flip': tensor_flip,
    'tensor_pad': tensor_pad,
}
