import argparse
import os
import numpy as np

import sys
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from functools import partial
from pprint import pprint

import polymath as pm


from codelets.examples.genesys.codelets import load_fusion_op_info
from codelets.examples.genesys.config_loader import load_config
from codelets.examples.genesys.genesys_network_sim import compile_full_model
from codelets.examples.genesys.data_generator import DataGen


CWD = Path(f"{Path(__file__).parent}")
MODEL_DIR = Path(f"{Path(__file__).parent}/models")

BENCHMARK_NAMES = ["resnet50", "resnet18", "lenet"]


def compile_benchmark(model_name,
                      cfg_name,
                      identifier=0,
                      verbose=False,
                      filtered_layers=None,
                      stop_stage=None,
                      skip_layers=None,
                      skip_broken_layers=False,
                      only_systolic=False,
                      filter_op_types=None,
                      skip_op_types=None,
                      store_results=True,
                      count_compute=False,
                      check_layer_count=False,
                      conv_tile_constraints=None,
                      dir_ext=None
                      ):
    cfg_path = f"{CWD}/configs/{cfg_name}"
    arch_config = load_config(cfg_path)
    if dir_ext is None:
        dir_ext = ""
    else:
        assert isinstance(dir_ext, str)


    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    graph = pm.from_onnx(model_path)
    program, _ = compile_full_model(model_name,
                                 cfg_path,
                                 store_compile=False,
                                 dir_ext=None,
                                 added_constr=None,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=arch_config['FUSE_LAYERS'],
                                 generate_data=False,
                                    graph=graph,
                                    batch_size=arch_config['BATCH_SIZE'])

    if conv_tile_constraints is not None:
        conv_layers = ["conv_bias", "conv_bias_add_relu", "conv_bias_relu"]
        for l in conv_layers:
            if "LEVEL1_hint" not in program.hag.codelets[l].compilation_params.keys():
                program.hag.codelets[l].compilation_params[f'LEVEL1_hint'] = conv_tile_constraints
            else:
                orig = program.hag.codelets[l].compilation_params[f'LEVEL1_hint']
                new_constraint = f"{orig} and {conv_tile_constraints}"
                program.hag.codelets[l].compilation_params[f'LEVEL1_hint'] = new_constraint

    if only_systolic:
        if verbose:
            print(f"Compiling {model_name} without quantization, only systolic layers.")
        assert not arch_config['USE_QUANTIZATION']
        systolic_layers = ["conv_bias", "gemm", "gemm_no_bias", "conv"]
        program.filtered_compile(verbose=verbose, finalize=True, filter_op_types=systolic_layers)
    elif filtered_layers:
        assert skip_layers is None
        assert isinstance(filtered_layers, list)
        program.filtered_compile(filtered_layers, verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif filter_op_types:
        if verbose:
            print(f"Performing full compilation of {model_name} for layers {filter_op_types}.")
        program.filtered_compile(verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif skip_op_types:
        assert isinstance(skip_op_types, list)
        if verbose:
            print(f"Performing full compilation of {model_name}, skipping layers {skip_op_types}.")
        program.filtered_compile(verbose=verbose, finalize=True, skip_op_types=skip_op_types)
    else:
        if verbose:
            print(f"Performing full compilation of {model_name}.")
        program.compile(verbose=verbose, finalize=True, stop_stage=stop_stage)


    if stop_stage is None and store_results:
        sys_array_size = arch_config['ARRAY_M']
        dgen = DataGen(program,
                       single_codelets=not arch_config['SHARED_DATAGEN'],
                       shared_datagen=arch_config['SHARED_DATAGEN'],
                       dir_ext=f"{dir_ext}benchmark{sys_array_size}x{sys_array_size}",
                       identifier=identifier,
                       generate_data=arch_config['DATAGEN'],
                       verbose=verbose,
                       out_path=f"{CWD}/compilation_output",
                        store_whole_program=arch_config['SINGLE_PROGRAM_COMPILATION'])
        dgen.generate()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    argparser.add_argument('-m', '--model', required=True,
                            help='Name of the onnx model to create.')

    argparser.add_argument('-c', '--config', required=True,
                            help='Name of the architecture config file to use.')

    argparser.add_argument('-v', '--verbose', action='store_true', help='Use verbose compilation output')

    argparser.add_argument('-e', '--extension', type=str, default="0", help="Apply an extension to the compilation output directory name.")
    args = argparser.parse_args()

    fname = args.model
    if ".onnx" in fname:
        fname = fname.replace(".onnx", "")

    extension = args.extension
    verbose = args.verbose
    arch_config = args.config

    compile_benchmark(fname,
                        arch_config,
                        verbose=verbose,
                        identifier=extension)
