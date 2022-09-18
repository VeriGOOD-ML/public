from pathlib import Path
from typing import Iterable
from benchmarks.model_generator import create_custom_conv, create_custom_gemm, create_custom_matmul
from benchmarks.load_onnx_model import store_unique_model_layers
from examples.genesys import compile_genesys_layer, GENESYS_CFG
from examples.genesys.genesys_qmodels import compute_existing_values
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import json
import pprint
from collections import defaultdict
import onnx

from compile_layer import store_outputs
OUT_DIR = Path(f"{Path(__file__).parent}/compilation_output")
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks")
MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")
LAYER_DIR = Path(f"{Path(__file__).parent}/../benchmarks/layers")
BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}
import os
import shutil


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    print(f"H: {H}, Pad: {padding}, KH: {field_height}, Stride: {stride}")
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = np.int32((H + 2 * padding - field_height) / stride + 1)
    out_width = np.int32((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def compute_im2col_dims(params, oh, ow):
    if 'stride' in params:
        stride = params['stride']
    elif 'strides' in params:
        stride = params['strides']
    else:
        raise KeyError(f"Could not find stride {list(params.keys())}")
    if 'pads' in params:
        pad = params['pads']
    elif 'pad' in params:
        pad = params['pad']
    else:
        raise KeyError(f"Could not find stride {list(params.keys())}")
    input = np.random.randint(low=0, high=127, size=(params['N'], params['IC'], params['IH'], params['IW']), dtype=np.int32)
    weights = np.random.randint(low=0, high=127, size=(params['OC'], params['IC'], params['KH'], params['KW']), dtype=np.int32)
    bias = np.zeros(shape=params['OC'], dtype=np.int32)
    tout = F.conv2d(torch.from_numpy(input.astype(np.float64)), torch.from_numpy(weights.astype(np.float64)),
                    torch.from_numpy(bias.astype(np.float64)), stride=stride, padding=pad)
    M = oh*ow
    N = params['KH']*params['KW']*params['IC']
    P = params['OC']
    # x_cols = im2col_indices(input, weights.shape[2], weights.shape[3], 0, params['stride'])
    # im2col(X, HF, WF, stride, pad)
    x_cols = im2col(input, weights.shape[2], weights.shape[3], stride, pad)

    assert M == x_cols.shape[1] and N == x_cols.shape[0]
    torch_res = F.linear(torch.from_numpy(x_cols.transpose(1, 0).astype(np.float64)),
                         torch.from_numpy(weights.reshape((weights.shape[0], -1)).astype(np.float64)),
                    torch.from_numpy(bias.astype(np.float64)))\
        .resize(oh, ow, params['N'], params['OC']).permute(2, 3, 0, 1)
    res = weights.reshape((weights.shape[0], -1)).dot(x_cols) + bias.reshape(-1, 1)
    out = res.reshape(params['OC'], oh, ow, params['N'])
    out = out.transpose(3, 0, 1, 2)
    torch.testing.assert_allclose(torch.from_numpy(out.astype(np.float64)), tout)
    torch.testing.assert_allclose(torch_res, tout)
    return M, N, P

def check_conv_params(n, ic, oc, ih, iw, k, stride, pad):
    layer = nn.Conv2d(ic, oc, k, stride, pad)

def compile_custom_gemm_layer(m, n, p, model_name, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None):
    # model_name = f"resnet50_{name_postfix}"
    # create_custom_gemm(optimize_model, training_mode, convert_data_format, to_polymath, M, N, P, fname=None):
    # create_custom_conv(optimize_model, training_mode, convert_data_format, to_polymath, input_shape, oc, ksize, stride,
    #                    pad,
    #                    name=None)
    create_custom_gemm(True, True, False, False, m, n, p, fname=model_name)
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_unique_model_layers(model_name, store_as_polymath=True)

    batch_size = 1
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    layer_name = f"{model_name}_gemm"

    # This function returns
    program = compile_genesys_layer(layer_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )
    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, "gemm")
        dir_ext = dir_ext or ''
        program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", "gemm", False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials, program=program)
    return program

def compile_custom_conv_layer(n, ic, oc, ih, iw, k, stride, pad, model_name, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None):
    check_conv_params(n, ic, oc, ih, iw, k, stride, pad)
    input_shape = (n, ic, ih, iw)
    # model_name = f"resnet50_{name_postfix}"

    create_custom_conv(True, True, False, False, input_shape, oc, k, stride, pad, name=model_name)
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_unique_model_layers(model_name, store_as_polymath=True)

    batch_size = 1
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    layer_name = f"{model_name}_conv"

    # This function returns
    program = compile_genesys_layer(layer_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )

    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, 'conv_bias')
        dir_ext = dir_ext or ''
        program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", "conv", False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials, program=program)

            # else:
            # print(f"{o}")
    return program


def get_onnx_shape(tensor_dict, val_name):
    assert val_name in tensor_dict
    value = tensor_dict[val_name]
    shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
    return tuple(shape)

def get_all_unique_layer_params(model_name, layer_name, input_shape_params, out_shape_params, param_names):
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    print(model_path)
    model = onnx.load_model(model_path)
    tensor_dict = {i.name: i for i in model.graph.input}
    tensor_dict.update({o.name: o for o in model.graph.output})
    tensor_dict.update({v.name: v for v in model.graph.value_info})
    layer_params = []
    for n in model.graph.node:
        if n.op_type == layer_name:
            outputs = n.output
            inputs = n.input
            kv_map = {}

            for a in n.attribute:
                if a.name in param_names:
                    attr_val = onnx.helper.get_attribute_value(a)
                    assert a.name not in kv_map
                    if isinstance(attr_val, Iterable):
                        kv_map[a.name] = attr_val[0]
                    else:
                        kv_map[a.name] = attr_val
            for i, v in enumerate(inputs):
                shape = get_onnx_shape(tensor_dict, v)
                assert len(shape) == len(input_shape_params[i])
                for pidx, p in enumerate(input_shape_params[i]):
                    if p in kv_map:
                        assert kv_map[p] == shape[pidx], f"Mismatched values for input key {p}:\n" \
                                                         f"Input val: {shape[pidx]}, Previous: {kv_map[p]} "
                    else:
                        kv_map[p] = shape[pidx]
            for o, v in enumerate(outputs):
                shape = get_onnx_shape(tensor_dict, v)
                assert len(shape) == len(out_shape_params[o])
                for pidx, p in enumerate(out_shape_params[o]):
                    if p in kv_map:
                        assert kv_map[p] == shape[pidx], f"Mismatched values for output key {p}:\n" \
                                                         f"Output val: {shape[pidx]}, Previous: {kv_map[p]} "
                    else:
                        kv_map[p] = shape[pidx]
            if kv_map not in layer_params:
                layer_params.append(kv_map)
    return layer_params

def ceildiv(a, b):
    return -(-a // b)

def validate_base_addresses(conv_params, instr_len, layer_type="conv_bias"):
    BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}
    if instr_len * 32 // 8 > BENCH_BASE_ADDR['BBUF']:
        return False
    if layer_type == "conv_bias":

        buf_size = conv_params['OC']*32 // 8
        buf_end = buf_size + BENCH_BASE_ADDR['BBUF']
        if buf_end > BENCH_BASE_ADDR['WBUF']:
            return False
        wgt_size = conv_params['IC']*conv_params['OC']*conv_params['KW']*conv_params['KH']
        wgt_end = wgt_size + BENCH_BASE_ADDR['WBUF']
        if wgt_end > BENCH_BASE_ADDR['IBUF']:
            return False
        ipt_size = conv_params['IC']*conv_params['N']*conv_params['IH']*conv_params['IW']
    else:
        buf_size = conv_params['P'] * 32 // 8
        buf_end = buf_size + BENCH_BASE_ADDR['BBUF']
        if buf_end > BENCH_BASE_ADDR['WBUF']:
            return False
        wgt_size = conv_params['M'] * conv_params['N']
        wgt_end = wgt_size + BENCH_BASE_ADDR['WBUF']
        if wgt_end > BENCH_BASE_ADDR['IBUF']:
            return False
    return True

def splits_from_params(params):
    splits = {}
    for k, v in params.items():
        if "_tile" in k:
            idx = k.split("_")[0]
            splits[idx] = v
    return splits


def invalid_tiling(params):
    splits = splits_from_params(params)
    if splits['IC'] == 1 or any([splits['KH'] > 1, splits['KW'] > 1, splits['OH'] > 1, splits['OW'] > 1]):
        return False
    else:
        return True

def pixel_skip(params):
    return params['KH'] < params['stride'] or params['KW'] < params['stride']

def tiled_oc(params):
    splits = splits_from_params(params)
    return splits['OC'] > 1

def layer_key_from_params(params):
    p = {k: v for k,v in params.items() if 'tile' not in k}
    return tuple(sorted(list(p.items()), key=lambda x: x[0]))

def programs_with_params(base_test_name, check_params):
    valid_programs = []
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):
        if dir.name.startswith(base_test_name) and "case" in dir.name and dir.is_dir():
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            is_valid = True
            for k, v in check_params.items():
                assert k in params
                if isinstance(v, int):
                    if v != params[k]:
                        is_valid = False
                        break
                elif not v(params[k]):
                    is_valid = False
                    break
            if is_valid:
                valid_programs.append(dir.name)
    print(valid_programs)

def find_duplicates(dname, base_test_name):
    pixel_tests = []
    invalid_tiling_tests = []
    valid_tests = []
    tiled_oc_tests = []
    all_params = []
    duplicates = []
    for d in os.scandir(f"{OUT_DIR}/{dname}"):
        if d.name.startswith(base_test_name) and d.is_dir():
            params = collect_program_params(f"{d.path}/{base_test_name}_json.json")
            if check_dictionary_presence(params, all_params):
                duplicates.append(d.name)
            else:
                all_params.append(params)
    print(f"Duplicates: {len(duplicates)}")


def check_programs():
    with open("tgt_tests.txt", "r") as f:
        files = f.readlines()
    files = [f.strip() for f in files]
    base_test_name = "resnet50_1_conv"
    pixel_tests = []
    invalid_tiling_tests = []
    valid_tests = []
    tiled_oc_tests = []
    all_params = []
    duplicates = []
    constraints = defaultdict(list)
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):

        if dir.name.startswith(base_test_name) and "case" in dir.name and dir.is_dir() and dir.name not in files:
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            if check_dictionary_presence(params, all_params):
                duplicates.append(dir.name)
            else:
                all_params.append(params)
            if invalid_tiling(params):
                invalid_tiling_tests.append(dir.name)
            if pixel_skip(params):
                pixel_tests.append(dir.name)
            if not pixel_skip(params) and not invalid_tiling(params):
                valid_tests.append(dir.name)
            if tiled_oc(params):
                tiled_oc_tests.append(dir.name)

    print(f"Pixel tests: {pixel_tests}, {len(pixel_tests)}")
    print(f"Invalid tiling tests: {invalid_tiling_tests}, {len(invalid_tiling_tests)}")
    print(f"Valid tests: {valid_tests}, {len(valid_tests)}")
    print(f"Tiled OC tests: {tiled_oc_tests}, {len(tiled_oc_tests)}")
    print(f"Duplicates: {len(duplicates)}")
    # for i in invalid_tiling_tests:
    #     if i in pixel_tests:
            # print(i)
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):
        # if dir.name in invalid_tiling_tests or dir.name in pixel_tests:
        if dir.name in invalid_tiling_tests:
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            key = layer_key_from_params(params)
            added_constrs = []
            if key in constraints:
                added_constrs.append(unique_splits_constraints(constraints[key]))

            if dir.name in invalid_tiling_tests:
                added_constrs.append("splits['IC'] > 1")

            if len(added_constrs) == 0:
                cstrt = None
            else:
                cstrt = " and ".join(added_constrs)
            ext = dir.name.split(base_test_name)[-1][1:]
            program = compile_from_existing_program(base_test_name, f"{ext}", base_test_name, params=params, added_constraint=cstrt)
            nparams = get_program_params(program, "conv_bias")
            splits = splits_from_params(nparams)
            constraints[key].append(splits)

def collect_program_params(fpath):
    params = {}
    with open(fpath) as f:
        program = json.load(f)
        layer = program['program'][0]

        params['N'] = layer['iterable_dimensions']['N']
        params['IC'] = layer['iterable_dimensions']['IC']
        params['OC'] = layer['iterable_dimensions']['OC']
        params['KH'] = layer['iterable_dimensions']['KH']
        params['KW'] = layer['iterable_dimensions']['KW']
        params['OH'] = layer['iterable_dimensions']['OH']
        params['OW'] = layer['iterable_dimensions']['OW']
        params['IH'] = layer['inputs'][0]["tiling"]['DRAM']['IH']
        params['IW'] = layer['inputs'][0]["tiling"]['DRAM']['IW']
        params['stride'] = layer['operation_parameters']['stride']
        params['pad'] = layer['operation_parameters']['pad']
        params['N_tile'] = layer['iterable_dimensions']['N'] // layer['inputs'][0]["tiling"]['IBUF']['N']
        params['IC_tile'] = layer['iterable_dimensions']['IC'] // layer['inputs'][0]["tiling"]['IBUF']['IC']
        params['KH_tile'] = layer['iterable_dimensions']['KH'] // layer['inputs'][1]["tiling"]['WBUF']['KH']
        params['KW_tile'] = layer['iterable_dimensions']['KW'] // layer['inputs'][1]["tiling"]['WBUF']['KW']
        params['OH_tile'] = layer['iterable_dimensions']['OH'] // layer['outputs'][0]["tiling"]['OBUF']['OH']
        params['OW_tile'] = layer['iterable_dimensions']['OW'] // layer['outputs'][0]["tiling"]['OBUF']['OW']
        params['OC_tile'] = layer['iterable_dimensions']['OC'] // layer['outputs'][0]["tiling"]['OBUF']['OC']
    return params

def compile_from_existing_program(dirname, dir_ext, json_name, added_constraint=None, params=None, preserve_tiling=False):
    if not params:
        params = collect_program_params(f"{OUT_DIR}/{dirname}/{json_name}_json.json")
    program = compile_custom_conv_layer(params['N'], params['IC'], params['OC'], params['IH'],
                                        params['IW'], params['KH'],
                                        params['stride'], params['pad'],
                                        json_name)
    program._name = json_name
    if added_constraint:
        program = update_tile_constraints(program, added_constraint, "conv_bias")
    program.compile(verbose=False, finalize_instructions=True)

    store_outputs(dirname, None, False,
                  1,
                  False,
                  None,
                  use_random=True,
                  dir_ext=dir_ext,
                  actual_data=False,
                  store_partials=False, program=program)
    return program

def update_tile_constraints(program, constraint, layer_type, orig_constraint=None):
    # if not orig_constraint:
    #     program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint'] = constraint
    # else:
    #     program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint'] = f"{orig_constraint} and {constraint}"
    if 'LEVEL1_hint' not in program.hag.codelets[layer_type].compilation_params.keys():
        program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = constraint
    elif constraint not in program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']:
        orig = program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']
        new_constraint = f"{orig} and {constraint}"
        program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = new_constraint
    return program

def collect_existing_params(base_dir_name, base_test_name):
    all_compilation_configs = []
    duplicate_configs = []
    dir_count = 0
    for dir in os.scandir(OUT_DIR):

        if base_dir_name in dir.name and dir.is_dir():
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            dir_count += 1


            if check_dictionary_presence(params, all_compilation_configs):
                duplicate_configs.append(dir.name)
            else:
                all_compilation_configs.append(params)
    print(list(sorted(duplicate_configs)))

    print(len(list(sorted(duplicate_configs))))

    return all_compilation_configs, dir_count

def unique_splits_constraints(splits):
    assert len(splits) > 0
    constraints = []
    for s in splits:
        c = []
        for k, v in s.items():
            c.append(f"'{k}': {v}")
        c_str = "(splits != {" + ",".join(c) + "})"
        constraints.append(c_str)
    return " and ".join(constraints)

def check_dictionary_presence(d, dlist):
    for v in dlist:
        if v == d:
            return True
    return False

def get_program_params(program, cdlt_type):

    cparams = {}
    if cdlt_type == "gemm":
        cparams['M'], cparams['N'] = program.codelets[0].inputs[0].shape
        assert program.codelets[0].inputs[1].shape[0] == cparams['N']
        cparams['N'], cparams['P'] = program.codelets[0].inputs[1].shape
    else:
        cparams['N'], cparams['IH'], cparams['IW'], cparams['IC'] = program.codelets[0].inputs[0].shape

        cparams['N'], cparams['OH'], cparams['OW'], cparams['OC'] = program.codelets[0].outputs[0].shape

        cparams['KH'], cparams['KW'], cparams['IC'], cparams['OC'] = program.codelets[0].inputs[1].shape
        cparams['stride'] = program.codelets[0].required_params['stride'].value
        cparams['pad'] = program.codelets[0].required_params['pad'].value

    for k, v in program.codelets[0].param_splits[1].items():
        cparams[f'{k}_tile'] = v
    return cparams

def scale_and_compile_layers(model_name, dir_ext, layer_params, updated_layer_params, nunique,
                             idx_start=None,
                             verbose=False,
                             im2col_layers=None, added_constraint=None):
    im2col_layers = [] if not im2col_layers else im2col_layers
    layer_idx = idx_start if idx_start is not None else len(updated_layer_params)
    orig_conv_constraint = None
    orig_gemm_constraint = None
    all_shapes = []
    param_constraints = defaultdict(list)
    for idx in range(len(layer_params)):

        if idx < layer_idx:
            continue
        layer = layer_params[idx]
        nlayer_perm = 0
        scale_val = 1
        if verbose:
            layer_param_str = ", ".join([f"{k} = {v}" for k, v in layer.items()])
            print(f"Generating permutations for layer {idx}:\n"
                  f"{layer_param_str}")

        while nlayer_perm < nunique:
            print(f"Start for {nlayer_perm}")
            new_layer_params = {}
            if any([v <= scale_val for k, v in layer.items() if k not in ['N', 'strides', 'pads', 'IC', 'OC', 'KH', 'KW']]) and \
                    scale_val > 1:
                layer_param_str = "\n".join([f"{k} = {v}" for k, v in layer.items()])
                raise RuntimeError(f"Invalid scaling value for layer:\n"
                                   f"Scale value: {scale_val}\n"
                                   f"Layer sizes:\n{layer_param_str}")

            n = ceildiv(layer['N'], scale_val)
            new_layer_params['N'] = n
            if layer['IC'] > GENESYS_CFG['ARRAY_M']:
                ic = ceildiv(layer['IC'], scale_val)
            else:
                ic = layer['IC']
            new_layer_params['IC'] = ic

            if layer['OC'] > GENESYS_CFG['ARRAY_N']:
                oc = ceildiv(layer['OC'], scale_val)
            else:
                oc = layer['OC']
            ic += (GENESYS_CFG['ARRAY_M'] - ic) % GENESYS_CFG['ARRAY_M']
            oc += (GENESYS_CFG['ARRAY_M'] - oc) % GENESYS_CFG['ARRAY_M']
            new_layer_params['OC'] = oc
            h = ceildiv(layer['IH'], scale_val)
            new_layer_params['IH'] = h
            w = ceildiv(layer['IW'], scale_val)
            new_layer_params['IW'] = w
            ksize = layer['KH']
            stride = layer['strides']
            pad = layer['pads']
            new_layer_params['KH'], new_layer_params['KW'] = ksize, ksize

            # if check_dictionary_presence(new_layer_params, all_shapes):
            #     scale_val += 1
            #     continue
            # else:
            #     all_shapes.append(new_layer_params.copy())
            new_layer_params['pads'] = pad
            new_layer_params['strides'] = stride

            if idx in im2col_layers:

                oh = int((h + 2 * pad - ksize) / stride) + 1
                ow = int((w + 2 * pad - ksize) / stride) + 1
                M, N, P = compute_im2col_dims(new_layer_params, oh, ow)
                program = compile_custom_gemm_layer(M, N, P, f"{model_name}_custom")
            else:
                program = compile_custom_conv_layer(n, ic, oc, h, w, ksize, stride, pad, f"{model_name}_custom")

            if orig_conv_constraint is None:
                orig_conv_constraint = program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint']
                orig_gemm_constraint = program.hag.codelets['gemm'].compilation_params['LEVEL1_hint']
                if added_constraint is not None:
                    orig_conv_constraint = f"{orig_conv_constraint} and {added_constraint}"
                    orig_gemm_constraint = f"{orig_gemm_constraint} and {added_constraint}"

            if idx in im2col_layers:
                orig_constraint = orig_gemm_constraint
                cdlt_type = "gemm"
            else:
                orig_constraint = orig_conv_constraint
                cdlt_type = "conv_bias"
            key = layer_key_from_params(new_layer_params)

            if key in param_constraints:
                assert orig_conv_constraint is not None
                constraint = unique_splits_constraints(param_constraints[key])
                program.hag.codelets[cdlt_type].compilation_params[
                    'LEVEL1_hint'] = f"{orig_constraint} and {constraint}"

            if verbose:
                layer_param_str = ", ".join([f"{k} = {v}" for k, v in new_layer_params.items()])
                constraint_str = program.hag.codelets[cdlt_type].compilation_params['LEVEL1_hint']
                print(f"Generating permutation {nlayer_perm} for layer {idx}:\n"
                      f"layer params: {layer_param_str}\n"
                      f"Constraints: {constraint_str}")
            try:
                print(f"Compiling with split {scale_val}")
                program.compile(verbose=False, finalize_instructions=True)
            except Exception as e:
                print(f"Unable to compile layer {e}")
                scale_val += 1
                # layer_splits = []
                continue
            cparams = {}

            if idx in im2col_layers:
                cparams['M'], cparams['N'] = program.codelets[0].inputs[0].shape
                assert program.codelets[0].inputs[1].shape[0] == cparams['N']
                cparams['N'], cparams['P'] = program.codelets[0].inputs[1].shape
            else:
                cparams['N'], cparams['IH'], cparams['IW'], cparams['IC'] = program.codelets[0].inputs[0].shape

                cparams['N'], cparams['OH'], cparams['OW'], cparams['OC'] = program.codelets[0].outputs[0].shape

                cparams['KH'], cparams['KW'], cparams['IC'], cparams['OC'] = program.codelets[0].inputs[1].shape
                cparams['stride'] = stride
                cparams['pad'] = pad

            for k, v in program.codelets[0].param_splits[1].items():
                cparams[f'{k}_tile'] = v
            nparams = get_program_params(program, cdlt_type)
            splits = splits_from_params(nparams)
            param_constraints[key].append(splits)
            # splits = splits_from_params(cparams)
            # layer_splits.append(splits)

            if not check_dictionary_presence(cparams, updated_layer_params):
                updated_layer_params.append(cparams)
                if cdlt_type != "gemm" and pixel_skip(cparams):
                    print(f"Test case wit pixel")
                instr_len = program.codelets[0].num_instr

                if not validate_base_addresses(cparams, instr_len, layer_type=cdlt_type):
                    raise RuntimeError
                if verbose:
                    print(f"Storing outputs for layer")
                    print(program.emit("operations_idx"))

                store_outputs("cc_layer1", cdlt_type, False,
                              1,
                              False,
                              None,
                              use_random=True,
                              dir_ext=f"{dir_ext}{idx*nunique + nlayer_perm}",
                              actual_data=False,
                              store_partials=False, program=program)
                nlayer_perm += 1
            else:
                raise RuntimeError(f"Found duplicate layer somehow:\n"
                                   f"Splits: {splits}\n"
                                   f"Updated Layer params: {updated_layer_params}\n"
                                   f"Cparams: {cparams}\n"
                                   f"Layer: {idx}\n"
                                   f"Layer perm: {nlayer_perm}")

    return updated_layer_params

if __name__ == "__main__":
    inp_params = []
    inp_params.append(["N", "IC", "IH", "IW"])
    inp_params.append(["OC", "IC", "KH", "KW"])
    inp_params.append(["OC"])

    out_params = []
    out_params.append(["N", "OC", "OH", "OW"])

    attr_params = []
    attr_params.append("pads")
    attr_params.append("strides")
    # programs_with_params("resnet50_1_conv", {"IC": lambda x: x > 512, "OC": lambda x: x > 512})
    # check_programs()
    # find_duplicates("test_fpga", "resnet50_custom_conv")
    # params, dir_count = collect_existing_params()
    # collect_existing_params("resnet50_custom_conv_asic", "resnet50_custom_conv")

    # layer_params = get_all_unique_layer_params("resnet50", "Conv", inp_params, out_params, attr_params)
    #
    # params = scale_and_compile_layers("resnet50", "asic_1cases", layer_params, [], 1,
    #                                   verbose=True, im2col_layers=[0], added_constraint=f"np.prod(list(splits.values())) <= 20")
    # asic 6
    # n = 1
    # oc = 128
    # ic = 128
    # ih = 30
    # iw = 30
    # stride = 2
    # pad = 1
    # k = 3

    # n = 1
    # oc = 1024
    # ic = 256
    # ih = 14
    # iw = 14
    # stride = 1
    # pad = 0
    # k = 1
    # passing = [1,2,3,4,5,6,8,9, 10,11,14,15,16,17,18,19,20,21,22]
    # failing = []
    # for i in range(23):
    #     if i not in passing:
    #         failing.append(i)
    ### Fail case 7
    # n = 1
    # oc = 512
    # ic = 128
    # ih = 28
    # iw = 28
    # stride = 1
    # pad = 0
    # k = 1
    # tiling = {"N": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 7, "IC": 1, "OC": 1}
    # model_name = "fpga_case7_test"

    # ### Fail Case 12
    # n = 1
    # oc = 256
    # ic = 256
    # ih = 28
    # iw = 28
    # stride = 2
    # pad = 1
    # k = 3
    # tiling = {"N": 1, "KH": 1, "KW": 1, "OH": 2, "OW": 2, "IC": 1, "OC": 1}
    #
    # model_name = "fpga_case12_test"
    # ### Fail Case 13
    # n = 1
    # oc = 1024
    # ic = 256
    # ih = 14
    # iw = 14
    # stride = 1
    # pad = 0
    # k = 1
    # model_name = "fpga_case13_test"
    # tiling = {"N": 1, "KH": 1, "KW": 1, "OH": 2, "OW": 2, "IC": 1, "OC": 1}
    #
    # #

    # ### Test 18
    # n = 1
    # oc = 2048
    # ic = 512
    # ih = 7
    # iw = 7
    # stride = 1
    # pad = 0
    # k = 1
    # model_name = "fpga_case19_test"
    # tiling = {"N": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1, "IC": 1, "OC": 1}
    # tcstr = f"splits != {tiling}"
    # #
    # program = compile_custom_conv_layer(n, ic, oc, ih, iw, k, stride, pad, model_name,
    #                                     store_compile=True,
    #                                     partials=False,
    #                                     added_constr=tcstr
    #                                     )

    m = 196
    # n = 160//2
    n = 64
    p = 64
    # m = 196
    # n = 160//2
    # p = 64*4
    # m = 4
    # n = 160//2
    # p = 64*2
    # # # # # # print(m)
    model_name = "asic_fixed10"
    # model_name = "fpga_fixed11_loop_reorder"
    # model_name = "fpga_fixed11_constrained"
    program = compile_custom_gemm_layer(m, n, p, model_name, partials=True, store_compile=True,
                                        added_constr=f"splits['M'] > 1 and splits['P'] > 1 and splits['N'] > 1 and sizes['P'] == {GENESYS_CFG['ARRAY_N']} and "
                                                     f"sizes['N'] == {GENESYS_CFG['ARRAY_N']}",
                                        # added_constr=f"splits['M'] > 1 and splits['P'] > 1 and splits['N'] > 1"
                                        )
    # # t = np.random.randint(2048, )
    # print(program.emit("operations_idx"))
    # # print(f"\n\n")
    # # for o in program.codelets[0].ops:
    # #     if o.op_type == "transfer":
    # #         o.test_contig_strides()
    # #     else:
    # #         print(f"{o}")
    print(program.emit("operations_idx"))
    # print(program.emit("string_final"))
    # compute_existing_values(f"{OUT_DIR}/asic_tests_new/resnet50_custom_conv_asic_cases18/resnet50_custom_conv_json.json")


