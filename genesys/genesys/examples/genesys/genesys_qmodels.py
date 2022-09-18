
from pathlib import Path
from typing import Dict, List
from .genesys_model_utils import get_resnet18, get_resnet50
# from .codelets import FUSION_OP_INFO, BINARY_CODELETS, UNARY_CODELETS
from .datagen_functions import binary, unary, manual_conv_from_existing, \
    maxpool2d, manual_conv, manual_gemm, conv_forward_naive, pad_conv, \
    pad_gemm, global_avg_pool, depthwise_conv2d, OperandData
from examples.genesys.codelets.reference_impls.data_transformations import transform_data
from examples.genesys.codelets.reference_impls.ref_op import create_operand_data, numpy_datagen

import numpy as np
import json

WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)




def compute_existing_values(json_path):
    with open(f"{json_path}", "r") as f:
        params = json.load(f)

    inpt_shape = list(params['program'][0]['inputs'][0]['shape_symbols'].values())
    wgt_shape = list(params['program'][0]['inputs'][1]['shape_symbols'].values())
    out_shape = list(params['program'][0]['outputs'][0]['shape_symbols'].values())
    stride = params['program'][0]['operation_parameters']['stride']

    parent_dir = Path(f"{Path(json_path).parent}")
    inpt_data = np.loadtxt(f"{parent_dir}/input_raw.txt", dtype=np.int32).reshape(tuple(inpt_shape))
    wgt_data = np.loadtxt(f"{parent_dir}/weights_raw.txt", dtype=np.int32).reshape(tuple(wgt_shape))
    out_data = np.loadtxt(f"{parent_dir}/output.txt", dtype=np.int32).reshape(tuple(out_shape))
    res = manual_conv_from_existing(inpt_data, wgt_data, out_data, stride)
    np.testing.assert_allclose(res, out_data)

def retrieve_input_data(inouts: Dict[str, List[OperandData]], idx, cdlt,
                        scale=2, constant_val=None, print_range=False):
    for i in inouts['inputs']:
        if i.opname == cdlt.inputs[idx].name:
            return i

    data = numpy_datagen(cdlt.inputs[idx].shape, cdlt.inputs[idx].dtype.bits(),
                         fxp_dtype=f"{cdlt.inputs[idx].dtype}",
                         scale=scale, constant_val=constant_val, print_range=print_range)
    op = create_operand_data(data, cdlt.inputs[idx], idx)
    return op

def generate_random_values(cdlt, program, **kwargs) -> Dict[str, List[OperandData]]:
    if cdlt.op_name in program.metadata['FUSION_OP_INFO'].keys():
        inouts = generate_random_values_fused_layer(cdlt, program, **kwargs)
    elif "depthwise_conv" in cdlt.op_name:
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.outputs[0]]
        inouts = generate_random_values_dw_conv(cdlt, operands, **kwargs)
    elif "conv" in cdlt.op_name:
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2], cdlt.outputs[0]]
        inouts = generate_random_values_conv(cdlt, operands, **kwargs)
    elif "maxpool" in cdlt.op_name or "max_pool" in cdlt.op_name:
        operands = [cdlt.inputs[0], cdlt.outputs[0]]
        inouts = generate_random_values_maxpool(cdlt, operands, **kwargs)
    elif "global_avgpool" in cdlt.op_name or "global_avg_pool" in cdlt.op_name:
        operands = [cdlt.inputs[0], cdlt.outputs[0]]
        inouts = generate_random_values_global_avgpool(cdlt, operands, **kwargs)
    elif cdlt.op_name in program.metadata['BINARY_CODELETS']:
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.outputs[0]]
        inouts = generate_random_values_binary(cdlt, operands, **kwargs)
    elif cdlt.op_name in program.metadata['UNARY_CODELETS']:
        operands = [cdlt.inputs[0], cdlt.outputs[0]]
        inouts = generate_random_values_unary(cdlt, operands, **kwargs)
    else:
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2], cdlt.outputs[0]]
        assert "gemm" in cdlt.op_name, f"Could not find value number generator for operation {cdlt.op_name}"
        inouts = generate_random_values_gemm(cdlt, operands, **kwargs)

    assert inouts is not None
    return inouts





def generate_random_values_fused_layer(cdlt,
                                       program,
                                  inouts=None,
                                  base_path=".",
                                  generate_partial_values=False,
                                format="nhwc",
                                fixed_values=None):
    layers = program.metadata['FUSION_OP_INFO'][cdlt.op_name]['seq']


    assert "Conv" == layers[0]

    # This assumes the first operation is always conv
    compute_ops = cdlt.get_ops_by_type('compute')
    assert compute_ops[0].op_name == "MVMUL"
    conv_inpt = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2], compute_ops[0].dests[0]]
    out_operand = compute_ops[0].dests[0]
    inouts = generate_random_values_conv(cdlt, conv_inpt, generate_partial_values=generate_partial_values)
    res = inouts['outputs'].pop()
    inouts['inputs'].append(res)
    inpt_idx = 3
    compute_op_idx = 1
    assert len(compute_ops) > 1
    for l in layers[1:]:
        if l == "DepthwiseConv":
            op = compute_ops[compute_op_idx]
            assert op.op_name == "MACC"
            operands = [out_operand, cdlt.inputs[inpt_idx], op.dests[0]]
            inouts = generate_random_values_dw_conv(cdlt, operands, inouts=inouts, generate_partial_values=generate_partial_values)

            inpt_idx += 1
            compute_op_idx += 1
        elif l == "Relu":
            op = compute_ops[compute_op_idx]
            assert op.op_name == "RELU"
            operands = [out_operand, op.dests[0]]
            inouts = generate_random_values_unary(cdlt, operands, op_name="relu", inouts=inouts, generate_partial_values=generate_partial_values)
            compute_op_idx += 1
        elif l == "Add":
            op = compute_ops[compute_op_idx]
            assert op.op_name == "ADD"
            operands = [out_operand, cdlt.inputs[inpt_idx], op.dests[0]]
            inouts = generate_random_values_binary(cdlt, operands, op_name="elem_add", inouts=inouts, generate_partial_values=generate_partial_values)
            inpt_idx += 1
            compute_op_idx += 1
        if inouts['outputs'][-1].idx not in cdlt.outputs:
            res = inouts['outputs'].pop()
            inouts['inputs'].append(res)
    inouts['inputs'] = [i for i in inouts['inputs'] if i.idx not in cdlt.temps]

    return inouts

def generate_random_values_binary(cdlt, operands,
                                  op_name = None,
                                  inouts=None,
                                  base_path=".",
                                  generate_partial_values=False,
                                format="nhwc",
                                fixed_values=None):
    inouts = inouts or {"inputs": [], "outputs": []}

    op_name = op_name or cdlt.op_name
    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.

    scale = 2
    input1_op = retrieve_input_data(inouts, operands[0], cdlt, scale=scale)
    input2_op = retrieve_input_data(inouts, operands[1], cdlt, scale=scale)

    inouts["inputs"].append(input1_op)
    inouts["inputs"].append(input2_op)

    input1 = input1_op.data.copy()
    input2 = input2_op.data.copy()


    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))
        input2 = input2.transpose((0, 3, 1, 2))


    output = binary(input1, input2, op_name, f"{operands[0].dtype}")
    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    # Write outputs to file
    # save_array(f'{base_path}/output.txt', output)
    inouts["outputs"].append(create_operand_data(output, operands[-1]))


    return inouts



def generate_random_values_unary(cdlt,
                                 operands,
                                 op_name=None,
                                 base_path=".",
                                 generate_partial_values=False,
                                format="nhwc",
                                 inouts=None):
    inouts = inouts or {"inputs": [], "outputs": []}
    op_name = op_name or cdlt.op_name

    if "sigmoid" in op_name:
        scale = 1.5
    elif "tanh" in op_name:
        scale = 1.6
    elif "reduce_mean2d" in op_name:
        scale = 3.5
    elif "pow" in op_name:
        scale = 1.5
    else:
        scale = 1
    input1_op = retrieve_input_data(inouts, operands[0], cdlt, scale=scale)
    input1 = input1_op.data.copy()

    if "clip" in op_name:
        minval = cdlt.required_params['min'].value
        maxval = cdlt.required_params['max'].value

        params = (minval, maxval)
    elif "tensor_transpose2d" in op_name:
        axes = (1,0)
        params = (axes,)
    elif "pow" in op_name:
        exp = cdlt.required_params['exp'].value
        params = (exp,)
    elif "reduce_mean" in op_name or "reduce_min" in op_name:
        axis = cdlt.required_params['axis'].value
        params = (axis,)

    else:
        params = tuple([])

    inouts["inputs"].append(input1_op)



    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))



    output = unary(input1, op_name, f"{operands[0].dtype}", *params)

    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))


    inouts["outputs"].append(create_operand_data(output, operands[-1]))
    return inouts





def generate_random_values_maxpool(cdlt, operands,
                                format="nhwc",
                                   base_path=".",
                                   generate_partial_values=False,
                                   inouts=None):
    inouts = inouts or {"inputs": [], "outputs": []}


    input_dims = tuple(operands[0].tiling['DRAM'].values())
    KH = cdlt.required_params['KH'].value

    stride_x = cdlt.required_params['sx'].value

    input1_op = retrieve_input_data(inouts, operands[0], cdlt)

    inouts["inputs"].append(input1_op)
    input = input1_op.data.copy()

    if format.lower() == "nhwc":
        input = input.transpose((0, 3, 1, 2))


    output = maxpool2d(input.astype(np.int64), KH, stride_x, padding=0)


    output = output.transpose((0, 2, 3, 1))

    inouts["outputs"].append(create_operand_data(output, operands[-1]))
    return inouts


def generate_random_values_global_avgpool(cdlt, operands,
                                          inouts=None,
                                          base_path=".",
                                          generate_partial_values=False,
                                          format="nhwc"):
    inouts = inouts or {"inputs": [], "outputs": []}


    input1_op = retrieve_input_data(inouts, operands[0], cdlt)

    inouts["inputs"].append(input1_op)
    input = input1_op.data.copy()

    if format.lower() == "nhwc":
        input = input.transpose((0, 3, 1, 2))


    output = global_avg_pool(input.astype(np.int64), f"{operands[0].dtype}")

    output = output.transpose((0, 2, 3, 1))

    inouts["outputs"].append(create_operand_data(output, operands[-1]))
    return inouts



def generate_random_values_dw_conv(cdlt, operands,
                                   inouts=None,
                                   base_path=".",
                                   generate_partial_values=False,
                                format="nhwc"):
    inouts = inouts or {"inputs": [], "outputs": []}

    stride = cdlt.required_params['stride'].value


    input_op = retrieve_input_data(inouts, operands[0], cdlt, scale=2)
    inouts["inputs"].append(input_op)
    input = input_op.data.copy()

    weight_op = retrieve_input_data(inouts, operands[1], cdlt, scale=2)
    inouts["inputs"].append(weight_op)
    weights = weight_op.data.copy()


    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)
        # Need to flip from (KH, KW, IC, OC) to (OC, IC, KH, KW)
        weights = weights.transpose(*tuple(WEIGHTS_CL_TO_CF))


    output = depthwise_conv2d(input, weights, stride, 0, f"{operands[0].dtype}")


    output = output.transpose(0, 2, 3, 1)


    inouts["outputs"].append(create_operand_data( output, operands[-1]))
    return inouts


def generate_random_values_conv(cdlt, operands,
                                base_path=".",
                                inouts=None,
                                format="nhwc",
                                generate_partial_values=False):
    inouts = inouts or {"inputs": [], "outputs": []}

    stride = cdlt.required_params['stride'].value


    input_op = retrieve_input_data(inouts, operands[0], cdlt, scale=1)
    inouts["inputs"].append(input_op)
    input = input_op.data.copy()

    weight_op = retrieve_input_data(inouts, operands[1], cdlt, scale=1)
    inouts["inputs"].append(weight_op)
    weights = weight_op.data.copy()

    inouts["inputs"].append(create_operand_data(transform_data(input, "input", "shuffled", cdlt), operands[0], fmt='shuffled'))
    inouts["inputs"].append(create_operand_data(transform_data(input, "input", "raw", cdlt), operands[0],  fmt='raw'))
    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "shuffled", cdlt), operands[1], fmt='shuffled'))
    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "shuffled_raw", cdlt), operands[1],  fmt='shuffled_raw'))
    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "raw", cdlt), operands[1], fmt='raw'))




    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)

        # Need to flip from (KH, KW, IC, OC) to (OC, IC, KH, KW)
        weights = weights.transpose(*tuple(WEIGHTS_CL_TO_CF))


    conv_param = {'stride': stride, 'pad': 0}
    assert len(operands) >= 4
    bias_op = retrieve_input_data(inouts, operands[2], cdlt, scale=1, constant_val=0)
    inouts["inputs"].append(bias_op)
    b = bias_op.data.copy()

    output, _ = conv_forward_naive(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    output = output.transpose(0, 2, 3, 1)
    if generate_partial_values:
        tinput = input.transpose(*tuple(ACT_CF_TO_CL))
        tweights = weights.transpose(*tuple(WEIGHTS_CF_TO_CL))
        coords = np.unravel_index(0, output.shape)
        partial_values_conv(cdlt, base_path, tinput, tweights, output, coords)


    inouts["outputs"].append(create_operand_data(output, operands[-1]))

    return inouts


def generate_random_values_gemm(cdlt, operands,
                                inouts=None,
                                base_path=".",
                                generate_partial_values=False):
    inouts = inouts or {"inputs": [], "outputs": []}

    input_op = retrieve_input_data(inouts, operands[0], cdlt, scale=1)
    inouts["inputs"].append(input_op)
    input = input_op.data.copy()

    weight_op = retrieve_input_data(inouts, operands[1], cdlt, scale=1)
    inouts["inputs"].append(weight_op)
    weights = weight_op.data.copy()

    bias_op = retrieve_input_data(inouts, operands[2], cdlt, scale=1)
    inouts['inputs'].append(bias_op)
    bias = bias_op.data.copy()


    inouts["inputs"].append(create_operand_data(transform_data(input, "input", "shuffled", cdlt), operands[0], fmt='shuffled'))
    inouts["inputs"].append(create_operand_data(transform_data(input, "input", "raw", cdlt), operands[0],  fmt='raw'))

    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "shuffled", cdlt), operands[1], fmt='shuffled'))
    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "shuffled_raw", cdlt), operands[1],  fmt='shuffled_raw'))
    inouts["inputs"].append(create_operand_data(transform_data(weights, "weights", "raw", cdlt), operands[1],  fmt='raw'))

    # assert len(cdlt.inputs) == 3
    # bias_op = retrieve_input_data(inouts, 2, cdlt, scale=1, constant_val=0)
    # inouts["inputs"].append(bias_op)
    # b = bias_op.data.copy()

    output = np.dot(np.int32(input), np.int32(weights))

    if generate_partial_values:

        partial_values_gemm(cdlt, base_path, input, weights, output, (0, 0))
    inouts["outputs"].append(create_operand_data(output, operands[-1]))

    return inouts


def get_model_values(model_name, layer_name, layer_num, write_data=False):
    if model_name == "resnet18":
        layer_data, model = get_resnet18(True, layer_name, layer_num)
    elif model_name == "resnet50":
        layer_data, model = get_resnet50(True, layer_name, layer_num)
    else:
        raise RuntimeError

    if "conv" in layer_name.lower():
        x, wgt, b, out = pad_conv(layer_data)
    else:
        assert "linear" in layer_name.lower() or "gemm" in layer_name.lower()
        x, wgt, b, out = pad_gemm(layer_data)
    if write_data:
        base_filename = f'{model_name}_{layer_name.lower()}'

        with open(f'{base_filename}_input_i8.txt', 'w') as f:
            # f.write('\n'.join(dram_layout(x)))
            f.write('\n'.join([str(i) for i in x.flatten()]))

        with open(f'{base_filename}_weights_i8.txt', 'w') as f:
            # f.write('\n'.join(dram_layout(shuffle_weights(wgt))))
            f.write('\n'.join([str(i) for i in wgt.flatten()]))

        out = out.flatten().tolist()
        out = [str(x) for x in out]
        with open(f'{base_filename}_output_i32.txt', 'w') as f:
            f.write('\n'.join(out))

        b = b.flatten().tolist()
        b = [str(x) for x in b]
        with open(f'{base_filename}_bias_i32.txt', 'w') as f:
            f.write('\n'.join(b))
    else:
        return x, wgt, b, out


def partial_values_gemm(cdlt, base_path, x, w, ref_out, o_coord):
    other_test, ic_vals = manual_gemm(x, w, o_coord)
    with open(f'{base_path}/out_coords.csv', 'w') as f:
        for k, v in ic_vals.items():
            # f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')

            f.write(f'IC={k}, (m/n/p), O_idx, I_idx, W_idx, I_val, W_val, partial\n')
            for l in v:
                f.write(f"IC={k}, " + "," + l + "\n")
    np.testing.assert_allclose(other_test, ref_out)


def partial_values_conv(cdlt, base_path, x, w, ref_out, o_coord):

    other_test, ic_vals = manual_conv(x, w, cdlt, o_coord, layout="nhwc")
    with open(f'{base_path}/out_coords.csv', 'w') as f:
        f.write(f'IC, (oc/n/ic/kh/kw/y/x), O_idx, I_idx, W_idx, I_val, W_val, partial\n')

        for k, v in ic_vals.items():
            for l in v:
                f.write(f"IC={k}, " + "," + l + "\n")
    np.testing.assert_allclose(other_test, ref_out)