from typing import Dict, List
from pathlib import Path
from examples.genesys import compile_genesys, get_arch, FXP_CONFIGS
from examples.genesys.datagen_functions import OperandData, save_array
from examples.genesys.genesys_qmodels import generate_random_values
from examples.genesys.config_loader import load_config
from tools.compile_layer import store_compilation_output
from codelets.compiler.program import CodeletProgram
from pprint import pprint
from fxpmath import Fxp
import numpy as np
import os
import json
CWD = Path(f"{__file__}").parent

BENCH_DIR = Path(f"{Path(__file__).parent}/../../benchmarks")
MODEL_DIR = Path(f"{Path(__file__).parent}/../../benchmarks/models")


def create_dirs(fpath, dir_ext):
    cwd = Path(f"{__file__}").parent
    base_path = Path(f"{cwd}/compilation_output/{Path(fpath).stem}{dir_ext}")

    if not Path(f"{base_path}").exists():
        try:
            os.makedirs(base_path)
        except OSError as e:
            print(f"Creation of directory {base_path} failed:\n {e}")
        else:
            print(f"Successfully created of directory {base_path}")
    else:
        print(f"Directory {base_path} already exists.")
    return base_path

def quantize_initializer(data, cdlt_operand, pm_node, idx):
    if data.shape != pm_node.shape:
        # First, check for transpose
        assert set(data.shape) == set(pm_node.shape)
        swapped_idx = [data.shape.index(i) for i in pm_node.shape]
        data = data.transpose(*swapped_idx)
        assert pm_node.shape == data.shape
    if data.shape != cdlt_operand.shape:
        pad_width = [(0, j-i) for i, j in zip(data.shape, cdlt_operand.shape)]
        data = np.pad(data, pad_width)
        assert data.shape == cdlt_operand.shape

    quant_data = Fxp(data, **FXP_CONFIGS[str(cdlt_operand.dtype)]).val
    assert quant_data.shape == cdlt_operand.shape and isinstance(quant_data, np.ndarray)
    opdata = OperandData(data=quant_data,
                         node_name=cdlt_operand.node_name,
                         opname=cdlt_operand.name,
                         idx=idx)
    return opdata

def store_model_values(program: CodeletProgram, base_path, model_data=None):
    formatted: List[OperandData] = []
    value_dict: Dict[str, Dict[str,OperandData]] = {"inputs": {},
                  "intermediate": {},
                  "outputs": {}
                  }
    storage_info = {}
    data_path = f"{base_path}/data"
    if not Path(data_path).exists():
        os.makedirs(data_path)
    if model_data is None:
        model_data = {}
        # model_data = generate_inputs_from_program(program)
        # exit()
    for c in program.codelets:
        inouts = {"inputs": [], "outputs": []}
        for i, op in enumerate(c.inputs):
            assert op.node_name in program.operand_mapping
            if op.node_name in model_data:
                assert op.node_name not in value_dict['outputs']
                assert op.node_name not in value_dict['intermediate']
                assert op.node_name in program.graph.nodes
                pm_node = program.graph.nodes[op.node_name]
                operand = quantize_initializer(model_data[op.node_name], op, pm_node, i)
                inouts['inputs'].append(operand)

            elif op.node_name in value_dict['outputs']:
                operand = value_dict['outputs'].pop(op.node_name)
                assert operand.data.shape == op.shape, f"Invalid shape for {operand.node_name} in {c.op_name}:\n" \
                                                       f"Codelet shape: {op.shape}\n" \
                                                       f"Value dict shape: {operand.data.shape}"
                inouts['inputs'].append(operand)
                value_dict['intermediate'][op.node_name] = operand
            elif op.node_name in value_dict['intermediate']:
                operand = value_dict['intermediate'][op.node_name]
                assert operand.data.shape == op.shape
                inouts['inputs'].append(operand)

        inouts = generate_random_values(c, inouts=inouts)

        for i in inouts['inputs']:

            if i.fmt is None and i.node_name not in value_dict['inputs'] and i.node_name not in value_dict['intermediate']:
                value_dict['inputs'][i.node_name] = i
                storage_info[i.node_name] = {"cdlt": c.cdlt_uid,
                                             "path": None,
                                             'cdlt_name': i.idx.name,
                                             'operand_type': 'input'}

            elif i.fmt is not None:
                formatted.append(i)

        for o in inouts['outputs']:
            if o.fmt is None:
                value_dict['outputs'][o.node_name] = o
                storage_info[o.node_name] = {"cdlt": c.cdlt_uid,
                                             "path": None,
                                             'cdlt_name': o.idx.name,
                                             'operand_type': 'output'}

    for f in formatted:
        if f.node_name in value_dict['inputs']:
            assert f.fmt is not None
            if not Path(f"{data_path}/{f.node_name}").exists():
                os.makedirs(f"{data_path}/{f.node_name}")
            save_array(f'{data_path}/{f.node_name}/{f.node_name}_{f.fmt}.txt', f.data)

    for n, i in value_dict['inputs'].items():
        assert n in program.operand_mapping
        assert i.node_name in storage_info

        if Path(f"{data_path}/{i.node_name}").exists():
            save_array(f'{data_path}/{i.node_name}/{i.node_name}.txt', i.data)
            storage_info[i.node_name]['path'] = f'{data_path}/{i.node_name}/'
        else:
            save_array(f'{data_path}/{i.node_name}.txt', i.data)
            storage_info[i.node_name]['path'] = f'{data_path}/{i.node_name}.txt'


    for n, o in value_dict['outputs'].items():
        assert n in program.operand_mapping
        assert o.node_name in storage_info

        if Path(f"{data_path}/{o.node_name}").exists():
            save_array(f'{data_path}/{o.node_name}/{o.node_name}.txt', o.data)
            storage_info[o.node_name]['path'] = f'{data_path}/{o.node_name}/'
        else:
            save_array(f'{data_path}/{o.node_name}.txt', o.data)
            storage_info[o.node_name]['path'] = f'{data_path}/{o.node_name}.txt'

    with open(f"{base_path}/data_locations.json", "w") as outf:
        outf.write(json.dumps(storage_info, indent=2))



def store_model_outputs(model_name,
                        training_mode,
                        batch_size=1,
                        verbose=False,
                        emit_to_stdout=None,
                        load_path=None,
                        dir_ext=None,
                        program=None,
                        added_constr=None,
                        model_data=None,
                        generate_data=True
                        ):
    name = model_name
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    update_cfg_dtypes = False

    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    print(f"Creating compilation output for {name}\n")
    if program is None:

        model_files = [d.name.split(".")[0] for d in os.scandir(f"{BENCH_DIR}/models/srdfg")]
        if name in model_files and not training_mode:
            program = compile_genesys(name,
                                      update_cfg_dtypes=update_cfg_dtypes,
                                      tiling_path=tiling_path,
                                      store_tiling=store_tiling,
                                      store_checkpoint=False,
                                      store_json_output=store_json_output,
                                      json_output_filename=json_output_filename,
                                      verbose=verbose,
                                      benchmark_path=BENCH_DIR,
                                      factor_fn='default',
                                    batch_size=batch_size,
                                    do_hoist_stage=True,
                                    tiling_search_algorithm=tile_method,
                                    do_tile_stage=True,
                                    print_config=False,
                                            do_compile=False
                                      )
        elif name in model_files:
            name = name.split("_")[0]
            program = compile_genesys(name,
                                      train=True,
                                      update_cfg_dtypes=update_cfg_dtypes,
                                      tiling_path=tiling_path,
                                      store_tiling=store_tiling,
                                      store_checkpoint=False,
                                      store_json_output=store_json_output,
                                      json_output_filename=json_output_filename,
                                      verbose=verbose,
                                      benchmark_path=BENCH_DIR,
                                      factor_fn='default',
                                    batch_size=batch_size,
                                    do_hoist_stage=True,
                                    tiling_search_algorithm=tile_method,
                                    do_tile_stage=True,
                                    print_config=False,
                                            do_compile=False
                                      )
        else:
            raise RuntimeError(f"Invalid layer name for compilation : {name}")

    if added_constr:
        program = update_tile_constraints(program, added_constr, model_name)

    print(f"Compiling")
    program.compile(verbose=False, finalize=True)


    arch_cfg = get_arch(None, None, update_cfg_dtypes)
    print(f"Configuration for program:")
    # pprint.pprint(arch_cfg)
    if emit_to_stdout is not None:
        assert isinstance(emit_to_stdout, str)
        if "json" in emit_to_stdout:
            pprint.pprint(program.emit(emit_to_stdout))

    base_path = store_compilation_output(program, "arch_cfg", extension="json", arch_cfg=arch_cfg, dir_ext=dir_ext)
    store_compilation_output(program, "operations_idx", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "json", extension="json", dir_ext=dir_ext)

    store_compilation_output(program, "string_final", extension="txt", dir_ext=dir_ext)

    store_compilation_output(program, "decimal", extension="txt", dir_ext=dir_ext)

    store_compilation_output(program, "binary", extension="txt", dir_ext=dir_ext)
    if generate_data:
        store_model_values(program, base_path, model_data=model_data)
    return program

def update_tile_constraints(program, layer_constraints, orig_constraint=None):
    # TODO: Fix this to add constraints on a per-layer or per-network basis
    for layer_type, constr in layer_constraints.items():
        if 'LEVEL1_hint' not in program.hag.codelets[layer_type].compilation_params.keys():
            program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = constr
        elif constr not in program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']:
            orig = program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']
            new_constraint = f"{orig} and {constr}"
            program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = new_constraint

    return program

def generate_inputs_from_program(program):
    model_data = {}
    value_dict: Dict[str, Dict[str, OperandData]] = {"inputs": {},
                  "intermediate": {},
                  "outputs": {}}
    init_input = None
    for c in program.codelets:
        inouts = {"inputs": [], "outputs": []}
        for i, op in enumerate(c.inputs):
            assert op.node_name in program.operand_mapping
            if op.node_name in model_data:
                assert op.node_name not in value_dict['outputs']
                assert op.node_name not in value_dict['intermediate']
                assert op.node_name in program.graph.nodes
                pm_node = program.graph.nodes[op.node_name]
                operand = quantize_initializer(model_data[op.node_name], op, pm_node, i)
                inouts['inputs'].append(operand)

            elif op.node_name in value_dict['outputs']:
                operand = value_dict['outputs'].pop(op.node_name)
                assert operand.data.shape == op.shape
                inouts['inputs'].append(operand)
                value_dict['intermediate'][op.node_name] = operand
            elif op.node_name in value_dict['intermediate']:
                operand = value_dict['intermediate'][op.node_name]
                assert operand.data.shape == op.shape
                inouts['inputs'].append(operand)
        inouts = generate_random_values(c, program.metadata['FUSION_OP_INFO'], inouts=inouts)

        for i in inouts['inputs']:

            if i.fmt is None and i.node_name not in value_dict['inputs'] and i.node_name not in value_dict['intermediate']:
                value_dict['inputs'][i.node_name] = i

        for o in inouts['outputs']:
            if o.fmt is None:
                value_dict['outputs'][o.node_name] = o
    return model_data

def compile_full_model(model_name,
                       cfg_file,
                       store_compile=False,
                       dir_ext=None,
                       added_constr=None,
                       verbose=False,
                       model_data=None,
                       fuse_layers=False,
                       generate_data=True,
                       tile_method=None,
                       batch_size=1,
                       graph=None
                       ):

    model_path = f"{MODEL_DIR}/{model_name}.onnx"

    tile_method = tile_method or "min_tiles"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    arch_cfg = load_config(f"{CWD}/configs/{cfg_file}")
    # This function returns
    program = compile_genesys(model_name,
                              arch_cfg,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=verbose,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False,
                              fuse_layers=fuse_layers,
                              graph=graph
                              )
    if store_compile:

        if added_constr:
            program = update_tile_constraints(program, added_constr)

        dir_ext = dir_ext or ''

        store_model_outputs(model_name,
                            arch_cfg['TRAINING'],
                            batch_size=1,
                            verbose=verbose,
                            emit_to_stdout=None,
                            dir_ext=f"{dir_ext}",
                            program=program,
                            model_data=model_data,
                            generate_data=generate_data)

    return program, arch_cfg



