from codelets.adl.operation import OperandTemplate, Loop
from codelets.codelet_impl.codelet import Codelet

from codelets.adl.graph import ArchitectureNode
from . import OP_DTYPES

def gemm(hag: ArchitectureNode):
    data = OperandTemplate("data", OP_DTYPES, ["M", "N"], dtype=OP_DTYPES[0])
    weight = OperandTemplate("weight", OP_DTYPES, ["N", "P"], dtype=OP_DTYPES[0])
    bias = OperandTemplate("bias", OP_DTYPES, ["P"], dtype=OP_DTYPES[2])
    out = OperandTemplate("out", OP_DTYPES, ["M", "P"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("gemm", [data, weight, bias], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "P") as p:
            with Loop(0, "N") as n:
                with Loop(0, "M") as m:
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(bias[p], ["DRAM", "BBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "systolic_array")
    return cdlt



def conv2d(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = OperandTemplate("data", OP_DTYPES, ["N", "IC", "IH", "IW"], dtype=OP_DTYPES[0])
    weight = OperandTemplate("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"], dtype=OP_DTYPES[0])
    out = OperandTemplate("out", OP_DTYPES, ["N", "OC", "OH", "OW"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("conv", [data, weight], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "OC") as oc:
            with Loop(0, "N") as n:
                with Loop(0, "IC") as ic:
                    with Loop(0, "KH") as kh:
                        with Loop(0, "KW") as kw:
                            with Loop(0, "OH") as y:
                                with Loop(0, "OW") as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(data[n, ic, y*"stride" + kh, x*"stride" + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    return cdlt

def conv2d_bias(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = OperandTemplate("data", OP_DTYPES, ["N", "IC", "IH", "IW"], dtype=OP_DTYPES[0])
    weight = OperandTemplate("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"], dtype=OP_DTYPES[0])
    bias = OperandTemplate("bias", OP_DTYPES, ["OC"], dtype=OP_DTYPES[2])
    out = OperandTemplate("out", OP_DTYPES, ["N", "OC", "OH", "OW"], dtype=OP_DTYPES[2])
    required_params = {}

    with Codelet("conv_bias", [data, weight, bias], [out], hag, required_params=required_params) as cdlt:

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "OC") as oc:
            with Loop(0, "N") as n:
                with Loop(0, "IC") as ic:
                    with Loop(0, "KH") as kh:
                        with Loop(0, "KW") as kw:
                            with Loop(0, "OH") as y:
                                with Loop(0, "OW") as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(bias[oc], ["DRAM", "BBUF"])
                                    cdlt.transfer(data[n, ic, y*"stride" + kh, x*"stride" + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                                    # cdlt.compute("MVMUL", [data[n, ic, y*"stride" + kh, x*"stride" + kw], weight[oc, ic, kh, kw], bias[oc]], [out[n, oc, y, x]], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    return cdlt

def elem_add(hag: ArchitectureNode):
    op1 = OperandTemplate("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    op2 = OperandTemplate("op2", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = OperandTemplate("add_out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("elem_add", [op1, op2], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        with Loop(0, "N") as n:
            with Loop(0, "C") as c:
                with Loop(0, "H") as h:
                    with Loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        cdlt.compute("ADD", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt

def relu(hag: ArchitectureNode):
    op1 = OperandTemplate("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    out = OperandTemplate("out", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])
    with Codelet("relu", [op1], [out], hag) as cdlt:
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with Loop(0, "N") as n:
            with Loop(0, "C") as c:
                with Loop(0, "H") as h:
                    with Loop(0, "W") as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
    return cdlt


def maxpool2d_nchw(hag: ArchitectureNode):
    # loop_order = ["n", "oc", "kh", "kh", "ow", "oh"]
    #
    data = OperandTemplate("data", OP_DTYPES, ["N", "C", "IH", "IW"])
    #
    out = OperandTemplate("out", OP_DTYPES, ["N", "C", "OH", "OW"])
    # # TODO: Add option to create operand
    # # 1. set IMM to negative infinity
    with Codelet("max_pool", [data], [out], hag) as cdlt:

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "VMEM")
        cdlt.configure("start", "IMM") # 1.
        with Loop(0, "N") as n:
            with Loop(0, "C") as c:
                with Loop(0, "OH") as y:
                    with Loop(0, "OW") as x:
                        cdlt.transfer(data[n, c, y * "stride", x * "stride"], ["DRAM", "VMEM"])

                        cdlt.compute("MAX", [data, "IMM"], [out], target="SIMD")
                        cdlt.transfer(out[n, c, y, x], ["SIMD", "VMEM"])

        with Loop(0, "N") as n:
            with Loop(0, "C") as c:
                with Loop(0, "KH") as kh:
                    with Loop(0, "KW") as kw:
                        with Loop(0, "OH") as y:
                            with Loop(0, "OW") as x:
                                cdlt.transfer(data[n, c, y*"stride" + kh, x*"stride" + kw], ["DRAM", "VMEM"])
                                cdlt.transfer(out[n, c, y, x], ["VMEM"])
                                cdlt.compute("MAX", [data], [out[n]], target="SIMD")
                                cdlt.transfer(out[n, c, y, x], ["pe_array", "OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
#
GENESYS_CODELETS = {
    # "conv": conv2d_nchw,
    "conv_bias": conv2d_bias,
    "conv": conv2d,
    "gemm": gemm,
    "elem_add": elem_add,
    "relu": relu
}
