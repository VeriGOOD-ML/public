from codelets.adl.graph import ComputeNode, StorageNode
from codelets import initialize_program
from .compilation_stages.stages import tile, hoist, remove_unused_variables, update_operand_dtypes, \
    add_simd_typecast, template_layout_pass, template_pad_pass, separate_simd_sa_ops, quantize_codelet
from .genesys_instructions import GENESYS_INSTRUCTIONS
from examples.genesys.instruction_templates.genesys_templates import GENESYS_TEMPLATES

# from .genesys_inference_codelets import GENESYS_CODELETS
from .codelets import SPLIT_INFO, load_impls_cdlts, load_fusion_op_info

from . import DTYPE_MAP

import numpy as np
from pathlib import Path
import json
from pprint import pprint
from codelets.adl.serialization import deserialize_hag
import polymath as pm

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/../../../benchmarks"
OUT_DIR = f"{BENCH_DIR}/compiler_outputs"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
TILING_DIR = f"{BENCH_DIR}/tiling_info"

LOOPS_PER_LEVEL = 7
INCR_MAP = "{'LD': {'IBUF': 0, 'WBUF': 1, 'OBUF': 2, 'BBUF': 3}," \
           "'ST': {'IBUF': 4, 'WBUF': 5, 'OBUF': 6, 'BBUF': 7}}"
LD_ST_MAP = "{'LD': 0, 'ST': 1}"
LD_STORE_LOOPS = 14
LD_STORE_OFFSET_MAP = "{'LD': {'IBUF': 0, 'WBUF': 6, 'OBUF': 10, 'BBUF': 14}," \
           "'ST': {'OBUF': 15, 'WBUF': 19 , 'BBUF': 23, 'IBUF': 24}}"

VALID_MODELS = ['resnet50', 'resnet18', 'maskrcnn', 'lenet', 'lenetbn', "my_ddpg_model", "my_ppo_model", "my_sac_model"]

def define_genesys(cfg):
    # TODO: Add capabilties to PE array not systolic_array

    with ComputeNode("Genesys", meta_cfg=cfg) as hag:
        VMEM_PARTITIONS = [cfg['VMEM_DEPTH'], cfg['SIMD_WIDTH'], cfg['ACC_WIDTH']]
        IMM_PARTITIONS = [cfg['IMM_DEPTH'], cfg['ACC_WIDTH']]
        INSTR_PARTITIONS = [cfg['INSTR_DEPTH'], cfg['INSTR_WIDTH']]

        vmem1 = StorageNode("VMEM1", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                            buffering_scheme="single",
                            width=cfg['ACC_WIDTH'], depth=cfg['VMEM_DEPTH'],
                            partitions=VMEM_PARTITIONS, latency=1, input_ports=2, output_ports=2)

        vmem2 = StorageNode("VMEM2", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                            buffering_scheme="single", width=cfg['ACC_WIDTH'], depth=cfg['VMEM_DEPTH'],
                            partitions=VMEM_PARTITIONS, latency=1, input_ports=2, output_ports=2)

        imm = StorageNode("IMM", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                          width=cfg['ACC_WIDTH'], depth=cfg['IMM_DEPTH'], partitions=IMM_PARTITIONS,
                          latency=1, input_ports=2, output_ports=2)

        # TODO: Does this need to be added?
        instr_mem = StorageNode("INSTR_MEM", access_type='RAM', width=cfg['INSTR_WIDTH'],
                                banks=cfg['INSTR_BANKS'], depth=cfg['INSTR_DEPTH'], partitions=INSTR_PARTITIONS,
                                latency=1, input_ports=2, output_ports=2)
        DRAM_PARTITIONS = [cfg['DRAM_DEPTH'], cfg['DRAM_BANKS'], cfg['DRAM_WIDTH']]

        dram = StorageNode("DRAM", access_type='RAM', banks=cfg['DRAM_BANKS'], addressable_dim=2,
                           width=cfg['DRAM_WIDTH'], depth=cfg['DRAM_DEPTH'], partitions=DRAM_PARTITIONS,
                           latency=1, input_ports=2, output_ports=2,
                           on_chip=False)

        with ComputeNode("systolic_array") as systolic_array:
            pe_array = ComputeNode("pe_array", dimensions=[cfg['ARRAY_N'], cfg['ARRAY_M']])
            # TODO: Need to formalize the storage node sizes by # elements, width, and datatype
            IBUF_PARTITIONS = [cfg['IBUF_DEPTH'], cfg['ARRAY_N'], cfg['DATA_WIDTH']]
            WBUF_PARTITIONS = [cfg['WBUF_DEPTH'], cfg['ARRAY_N'], cfg['ARRAY_M'], cfg['DATA_WIDTH']]
            OBUF_PARTITIONS = [cfg['OBUF_DEPTH'], cfg['ARRAY_M'], cfg['ACC_WIDTH']]
            BBUF_PARTITIONS = [cfg['BBUF_DEPTH'], cfg['ARRAY_M'], cfg['ACC_WIDTH']]
            ibuf = StorageNode("IBUF", access_type='RAM', banks=cfg['ARRAY_N'], buffering_scheme="double",
                               width=cfg['DATA_WIDTH'], depth=cfg['IBUF_DEPTH'], partitions=IBUF_PARTITIONS,
                               latency=1, input_ports=2, output_ports=2)
            wbuf = StorageNode("WBUF", access_type='RAM', banks=cfg['ARRAY_N'] * cfg['ARRAY_M'], addressable_dim=1,
                               buffering_scheme="double", width=cfg['DATA_WIDTH'], depth=cfg['WBUF_DEPTH'],
                               partitions=WBUF_PARTITIONS, latency=1, input_ports=2, output_ports=2)
            bbuf = StorageNode("BBUF", access_type='RAM', banks=cfg['ARRAY_M'], buffering_scheme="double",
                               width=cfg['ACC_WIDTH'], depth=cfg['BBUF_DEPTH'], partitions=BBUF_PARTITIONS,
                               latency=1, input_ports=2, output_ports=2)
            obuf = StorageNode("OBUF", access_type='RAM', banks=cfg['ARRAY_M'], buffering_scheme="double",
                               width=cfg['ACC_WIDTH'], depth=cfg['OBUF_DEPTH'], partitions=OBUF_PARTITIONS,
                               latency=1, input_ports=2, output_ports=2)
            # TODO: BW for DRAM is 64bits/cycle
            # Channel bandwidth = axi bandwidth
            # Request chunk of data for each iteration
            # Request size * data width * banks
            # iters = tile_size / (Request size * data width * banks)

            systolic_array.add_subgraph_edge('DRAM', 'IBUF', bandwidth=cfg['IBUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'WBUF', bandwidth=cfg['PARAM_BUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'BBUF', bandwidth=cfg['PARAM_BUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'OBUF', bandwidth=cfg['OBUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'INSTR_MEM', bandwidth=cfg['INSTR_CHANNEL_BW'])

            systolic_array.add_subgraph_edge('IBUF', 'pe_array', bandwidth=pe_array.dimensions[0] * cfg['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('WBUF', 'pe_array',
                                             bandwidth=np.prod(pe_array.dimensions) * cfg['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('BBUF', 'pe_array', bandwidth=pe_array.dimensions[1] * cfg['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'pe_array', bandwidth=pe_array.dimensions[1] * cfg['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'DRAM', bandwidth=cfg['OBUF_CHANNEL_BW'])
            # TODO: Add OBUF TO DRAM EDGE
            systolic_array.add_subgraph_edge('pe_array', 'OBUF', bandwidth=pe_array.dimensions[1] * cfg['ACC_WIDTH'])
            for p in GENESYS_INSTRUCTIONS['systolic_array']:
                systolic_array.add_primitive(p)
        simd = ComputeNode("SIMD", dimensions=[cfg['SIMD_WIDTH']])
        hag.add_subgraph_edge('VMEM1', 'SIMD', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM1', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('VMEM2', 'SIMD', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM2', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'OBUF', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('IMM', 'SIMD', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'IMM', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('DRAM', 'VMEM1', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('VMEM1', 'DRAM', bandwidth=cfg['SIMD_CHANNEL_BW'])

        hag.add_subgraph_edge('DRAM', 'VMEM2', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('VMEM2', 'DRAM', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('OBUF', 'SIMD', bandwidth=cfg['SIMD_WIDTH'] * cfg['ACC_WIDTH'])
        for p in GENESYS_INSTRUCTIONS['SIMD']:
            simd.add_primitive(p)

        ## Set templates
        # Config
        for hag_node, templates in GENESYS_TEMPLATES['config'].items():
            hag.add_start_template(hag_node, templates['start'](hag))
            hag.add_end_template(hag_node, templates['end'](hag))

        # Transfer
        for hag_node, template in GENESYS_TEMPLATES['transfer'].items():
            hag.add_transfer_template(*hag_node, template(hag))

        # Compute
        for hag_node, template in GENESYS_TEMPLATES['compute'].items():
            hag.add_compute_template(*hag_node, template(hag))

        # Loop
        hag.add_loop_template("systolic_array", GENESYS_TEMPLATES['loop'](hag))
        hag.add_loop_end_template("systolic_array", GENESYS_TEMPLATES['loop_end'](hag))

        # Program start and end
        hag.add_program_start_template("Genesys", GENESYS_TEMPLATES['program']['start'](hag))
        hag.add_program_end_template("Genesys", GENESYS_TEMPLATES['program']['end'](hag))

        # Codelet start and end
        hag.add_codelet_start_template("Genesys", GENESYS_TEMPLATES['codelet']['start'](hag))
        hag.add_codelet_end_template("Genesys", GENESYS_TEMPLATES['codelet']['end'](hag))

    GENESYS_CODELETS, _ = load_impls_cdlts(cfg)

    for op_name, cdlt in GENESYS_CODELETS.items():
        cdlt_instance = cdlt(hag)
        hag.add_codelet(cdlt_instance)
    hag.add_util_fn("get_loop_level_id", ["buffer_name", "loop_id", "level", "ld_st"],
                    f"(loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL} * level + ({INCR_MAP})[ld_st][buffer_name]")

    hag.add_util_fn("get_ld_st_loop_id", ["buffer_name", "index", "ld_st"],
                    f"index + {LD_STORE_LOOPS} + ({LD_STORE_OFFSET_MAP})[ld_st][buffer_name]")
    return hag


def add_genesys_templates(hag: ComputeNode):
    # Config
    for hag_node, templates in GENESYS_TEMPLATES['config'].items():
        hag.add_start_template(hag_node, templates['start'](hag))
        hag.add_end_template(hag_node, templates['end'](hag))

    # Transfer
    for hag_node, template in GENESYS_TEMPLATES['transfer'].items():
        hag.add_transfer_template(*hag_node, template(hag))

    # Compute
    for hag_node, template in GENESYS_TEMPLATES['compute'].items():
        hag.add_compute_template(*hag_node, template(hag))
    # Loop
    hag.add_loop_template("systolic_array", GENESYS_TEMPLATES['loop'](hag))

    # Program start and end
    hag.add_program_start_template("Genesys", GENESYS_TEMPLATES['program']['start'](hag))
    hag.add_program_end_template("Genesys", GENESYS_TEMPLATES['program']['end'](hag))

    # Codelet start and end
    hag.add_codelet_start_template("Genesys", GENESYS_TEMPLATES['codelet']['start'](hag))
    hag.add_codelet_end_template("Genesys", GENESYS_TEMPLATES['codelet']['end'](hag))


def update_genesys_cfg_from_dtypes(inp_cfg, dtypes=None):
    assert dtypes is not None
    inp_cfg['DATA_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['inp_weight']].bits()
    inp_cfg['WGT_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['inp_weight']].bits()
    inp_cfg['BIAS_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['bias_out']].bits()
    inp_cfg['ACC_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['bias_out']].bits()

    return inp_cfg


def run_srdfg_passes(graph, cfg, batch_size=1, verbose=False, fuse_layers=False):
    FUSION_OP_INFO = load_fusion_op_info(cfg)
    if batch_size > 1:
        batch_size_pass = pm.UpdateBatchSize(batch_size, graph.op_name)
        graph = batch_size_pass(graph)

    if cfg['TRAINING']:
        if verbose:
            print(f"Generating training graph for {graph.name}")
        graph = pm.create_training_graph(graph)
    # Split dw_conv
    split_pass = pm.SplitOps(SPLIT_INFO)
    graph = split_pass(graph)
    if fuse_layers:
        fusions = []
        for opname, info in FUSION_OP_INFO.items():
            if opname != "single_layer_info":
                assert 'seq' in info
                fusions.append(info['seq'])
        fusion_pass = pm.FuseOps(fusions, pad_conv_constraint=True)
        graph = fusion_pass(graph)


    multi_dim_pass = pm.RenameMultiDimOps()
    graph = multi_dim_pass(graph)

    return graph


def get_transformed_srdfg(model_name,
                          cfg,
                          train=False,
                          batch_size=1,
                          verbose=False,
                          benchmark_path=None):
    MODEL_DIR = f"{benchmark_path}/models/srdfg"
    if model_name not in ['resnet50', 'resnet18', 'maskrcnn', 'lenet', 'lenetbn']:
        raise RuntimeError(f"Invalid model name for compilation")
    if train:
        model_name = f"{model_name}_train"
    graph = pm.pb_load(f"{MODEL_DIR}/{model_name}.srdfg")
    graph = run_srdfg_passes(graph, cfg, train=train, batch_size=batch_size, verbose=verbose)
    return graph


def get_arch(dtypes, genesys_cfg, update_cfg_dtypes):
    dtypes = dtypes or dtypes_from_cfg(genesys_cfg)
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(inp_cfg=genesys_cfg, dtypes=dtypes)
    else:
        def_cfg = genesys_cfg
    return def_cfg


def dtypes_from_cfg(cfg):
    dtypes = {}
    dtypes['SIMD'] = f"FXP{cfg['ACC_WIDTH']}"
    dtypes['SYSTOLIC_ARRAY'] = {}
    dtypes['SYSTOLIC_ARRAY']['inp_weight'] = f"FXP{cfg['DATA_WIDTH']}"
    dtypes['SYSTOLIC_ARRAY']['bias_out'] = f"FXP{cfg['BIAS_WIDTH']}"
    return dtypes

def compile_genesys(model_name,
                    genesys_cfg,
                    update_cfg_dtypes=False,
                    tiling_path=None,
                    store_tiling=False,
                    store_json_output=False,
                    json_output_filename=None,
                    verbose=False,
                    benchmark_path=None,
                    dtypes=None,
                    print_config=True,
                    store_ops=False,
                    factor_fn='default',
                    store_checkpoint=False,
                    do_tile_stage=True,
                    do_hoist_stage=True,
                    batch_size=1,
                    fuse_layers=False,
                    relocation_offsets=None,
                    tiling_search_algorithm='valid_split',
                    do_compile=True,
                    graph=None,
                    do_srdfg_passes=True
                    ):
    MODEL_DIR = f"{benchmark_path}/models/srdfg"
    OUT_DIR = f"{benchmark_path}/compiler_outputs"

    TILING_DIR = f"{benchmark_path}/tiling_info"
    dtypes = dtypes or dtypes_from_cfg(genesys_cfg)
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(genesys_cfg, dtypes=dtypes)
    else:
        def_cfg = genesys_cfg

    if def_cfg['TRAINING']:
        model_name = f"{model_name}_train"
    if graph is None:
        graph = pm.pb_load(f"{MODEL_DIR}/{model_name}.srdfg")
    if do_srdfg_passes:
        graph = run_srdfg_passes(graph, def_cfg, batch_size=batch_size, verbose=verbose, fuse_layers=fuse_layers)

    genesys = define_genesys(def_cfg)
    if print_config:
        print(f"Compiling model with the following config:\n")
        sizes_cfg = def_cfg.copy()
        sizes_cfg['IBUF_SIZE'] = genesys.get_subgraph_node("IBUF").total_capacity
        sizes_cfg['WBUF_SIZE'] = genesys.get_subgraph_node("WBUF").total_capacity
        sizes_cfg['OBUF_SIZE'] = genesys.get_subgraph_node("OBUF").total_capacity
        sizes_cfg['BBUF_SIZE'] = genesys.get_subgraph_node("BBUF").total_capacity
        pprint(sizes_cfg)

    mode = "training" if def_cfg['TRAINING'] else "inference"
    # Codelet compilation starts here
    cdlts, impls = load_impls_cdlts(def_cfg)

    metadata = {'GENESYS_IMPLS': impls, 'GENESYS_CODELETS': cdlts,
                'FUSION_OP_INFO': load_fusion_op_info(def_cfg)}
    program = initialize_program(graph, genesys, metadata=metadata, mode=mode)
    program.add_compilation_step("template_layout_pass", template_layout_pass, template=True)

    program.add_compilation_step("template_pad_pass", template_pad_pass, template=True,
                                 )

    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True,
                                 stage_kwargs={'dtype_map': dtypes})
    program.add_compilation_step("remove_unused_variables", remove_unused_variables, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("insert_quantization", quantize_codelet, preproc=True)
    if tiling_search_algorithm == 'min_tiles':
        tile_kwargs = {
                'factor_fn_name': factor_fn,
                       'stopping_condition': exhaustive_search_stopping_condition,
                        'selection_metric': min_tiles_selection_metric,
                       'heuristic_fn': n_tiles_heuristic
                       }
    else:
        tile_kwargs = {
            'factor_fn_name': factor_fn,
                       'stopping_condition': valid_split_stopping_condition,
                        'selection_metric': current_permutation_selection_metric,
                       'heuristic_fn': n_tiles_heuristic
                       }

    if store_tiling:
        tile_kwargs['checkpoint_file'] = str(Path(f"{TILING_DIR}/{graph.name}_tiling_info_checkpoint.json").absolute())
    finalize_instructions = True
    if do_tile_stage:
        program.add_compilation_step("tile", tile, stage_kwargs=tile_kwargs)
        program.add_compilation_step("separate_ops", separate_simd_sa_ops)
    else:
        finalize_instructions = False

    if do_hoist_stage:
        program.add_compilation_step("hoist", hoist, dependencies=["tile"])
        program.add_compilation_step("simd_typecast", add_simd_typecast, dependencies=["hoist"],
                                 stage_kwargs={"dtype_map": {}, "codelet_output_map": {}},
                                 skip_noops=False)
    else:
        finalize_instructions = False

    if relocation_offsets:
        program.set_relocation_ns_offsets(relocation_offsets)

    if do_compile:
        if tiling_path is not None:
            program.compile(tiling_path=f"{TILING_DIR}/{tiling_path}", verbose=verbose,
                            finalize_instructions=finalize_instructions)
        else:
            program.compile(verbose=verbose, finalize_instructions=finalize_instructions)

        if store_tiling:
            program.store_tiling(f"{TILING_DIR}")

        if store_json_output:
            out_type = "json" if store_ops else "json_no_ops"
            res = program.emit(out_type)

            if json_output_filename is not None:
                with open(json_output_filename, "w") as outfile:
                    json.dump(res, outfile, indent=4)
            else:
                store_dir = f"{OUT_DIR}/{model_name}_compiled"
                p = Path(f"{store_dir}.json")
                if p.exists():
                    count = 0
                    while Path(f"{store_dir}{count}.json").exists():
                        count += 1
                    with open(f"{store_dir}{count}.json", "w") as outfile:
                        json.dump(res, outfile, indent=4)
                else:
                    with open(f"{store_dir}.json", "w") as outfile:
                        json.dump(res, outfile, indent=4)
    return program

def valid_split_stopping_condition(search_space):
    return True

def exhaustive_search_stopping_condition(search_space):
    return False

def current_permutation_selection_metric(search_space, permutation):
    return permutation

def min_tiles_selection_metric(search_space, permutation):
    # Get valid permutation with minimum number of tiles
    min_tiles = 100000000
    min_tiles_permutation = None
    for perm, tiles in search_space.items():
        if tiles < min_tiles:
            min_tiles = tiles
            min_tiles_permutation = perm
    if min_tiles_permutation is not None:
        return min_tiles_permutation
    else:
        return None

# Number of tiles as tiling heuristc
def n_tiles_heuristic(permutation):
    n_tiles = 1
    for i in range(len(permutation)):
        n_tiles = n_tiles * permutation[i]
    return n_tiles


def compile_genesys_layer(layer_file,
                          genesys_cfg,
                          update_cfg_dtypes=False,
                          tiling_path=None,
                          store_tiling=False,
                          store_json_output=False,
                          json_output_filename=None,
                          verbose=False,
                          benchmark_path=None,
                          dtypes=None,
                          print_config=True,
                          store_ops=False,
                          factor_fn='default',
                          store_checkpoint=False,
                          do_tile_stage=True,
                          do_hoist_stage=True,
                          batch_size=1,
                          fuse_layers=False,
                          save_genesys_filename=None,
                          load_genesys_filename=None,
                          relocation_offsets=None,
                          tiling_search_algorithm='valid_split',
                          do_compile=True):
    LAYER_DIR = f"{benchmark_path}/layers/srdfg"
    OUT_DIR = f"{benchmark_path}/compiler_outputs"

    TILING_DIR = f"{benchmark_path}/tiling_info"
    dtypes = dtypes or dtypes_from_cfg(genesys_cfg)
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(genesys_cfg, dtypes=dtypes)
    else:
        def_cfg = genesys_cfg

    graph = pm.pb_load(f"{LAYER_DIR}/{layer_file}.srdfg")
    graph = run_srdfg_passes(graph, def_cfg, train=False, batch_size=batch_size, verbose=verbose, fuse_layers=fuse_layers)
    if load_genesys_filename is None:
        genesys = define_genesys(def_cfg)
    else:
        genesys = deserialize_hag(load_genesys_filename)

    if save_genesys_filename is not None:
        with open(save_genesys_filename, 'w') as f:
            json.dump(genesys.to_json(), f, indent=4)
    if print_config:
        print(f"Compiling model with the following config:\n")
        sizes_cfg = def_cfg.copy()
        sizes_cfg['IBUF_SIZE'] = genesys.get_subgraph_node("IBUF").total_capacity
        sizes_cfg['WBUF_SIZE'] = genesys.get_subgraph_node("WBUF").total_capacity
        sizes_cfg['OBUF_SIZE'] = genesys.get_subgraph_node("OBUF").total_capacity
        sizes_cfg['BBUF_SIZE'] = genesys.get_subgraph_node("BBUF").total_capacity
        pprint(sizes_cfg)
    mode = "inference"
    program = initialize_program(graph, genesys, mode=mode)
    program.add_compilation_step("template_layout_pass", template_layout_pass, template=True)

    program.add_compilation_step("template_pad_pass", template_pad_pass, template=True,
                                 )


    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True,
                                 stage_kwargs={'dtype_map': dtypes})
    program.add_compilation_step("remove_unused_variables", remove_unused_variables, preproc=True, stage_kwargs={'shaped_nodes': {}})
    if tiling_search_algorithm == 'min_tiles':
        tile_kwargs = {'factor_fn_name': factor_fn, 'stopping_condition': exhaustive_search_stopping_condition,
                        'selection_metric': min_tiles_selection_metric, 'heuristic_fn': n_tiles_heuristic}
    else:
        tile_kwargs = {'factor_fn_name': factor_fn, 'stopping_condition': valid_split_stopping_condition,
                        'selection_metric': current_permutation_selection_metric, 'heuristic_fn': n_tiles_heuristic}

    if store_tiling and store_checkpoint:
        tile_kwargs['checkpoint_file'] = str(Path(f"{TILING_DIR}/{graph.name}_tiling_info_checkpoint.json").absolute())

    finalize_instructions = True
    if do_tile_stage:
        program.add_compilation_step("tile", tile, stage_kwargs=tile_kwargs)
        program.add_compilation_step("separate_ops", separate_simd_sa_ops)
    else:
        finalize_instructions = False

    if do_hoist_stage:
        program.add_compilation_step("hoist", hoist, dependencies=["tile"])
        program.add_compilation_step("simd_typecast", add_simd_typecast, dependencies=["hoist"],
                                     stage_kwargs={"dtype_map": {}, "codelet_output_map": {}}, skip_noops=False)
    else:
        finalize_instructions = False

    if relocation_offsets:
        program.set_relocation_ns_offsets(relocation_offsets)

    if do_compile:
        if tiling_path is not None:
            program.compile(tiling_path=f"{TILING_DIR}/{tiling_path}", verbose=verbose,
                            finalize_instructions=finalize_instructions)
        else:
            program.compile(verbose=verbose, finalize=finalize_instructions)

        if store_tiling:
            program.store_tiling(f"{TILING_DIR}")

        if store_json_output:
            out_type = "json" if store_ops else "json_no_ops"
            res = program.emit(out_type)

            if json_output_filename is not None:
                with open(json_output_filename, "w") as outfile:
                    json.dump(res, outfile, indent=4)
            else:
                store_dir = f"{OUT_DIR}/{layer_file}_compiled"
                p = Path(f"{store_dir}.json")
                if p.exists():
                    count = 0
                    while Path(f"{store_dir}{count}.json").exists():
                        count += 1
                    with open(f"{store_dir}{count}.json", "w") as outfile:
                        json.dump(res, outfile, indent=4)
                else:
                    with open(f"{store_dir}.json", "w") as outfile:
                        json.dump(res, outfile, indent=4)
    return program


def compile_extracted_genesys_layer(model_name,
                                    layer_name,
                                    arch_config,
                                    train=False,
                                    update_cfg_dtypes=False,
                                    batch_size=1,
                                    verbose=False,
                                    benchmark_path=None,
                                    dtypes=None,
                                    print_config=True,
                                    factor_fn='default',
                                    specific_layer_name=None,
                                    relocation_offsets=None):
    MODEL_DIR = f"{benchmark_path}/models/srdfg"

    dtypes = dtypes or dtypes_from_cfg(arch_config)
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(arch_config, dtypes=dtypes)
    else:
        def_cfg = arch_config

    if model_name not in ['resnet50', 'resnet18', 'maskrcnn', 'lenet', 'lenetbn']:
        raise RuntimeError(f"Invalid model name for extracting layer for compilation")
    if train:
        model_name = f"{model_name}_train"
    graph = pm.pb_load(f"{MODEL_DIR}/{model_name}.srdfg")
    graph = run_srdfg_passes(graph, train=train, batch_size=batch_size, verbose=verbose)
    found_layer = False
    for node in list(graph.nodes.values()):
        if not isinstance(node, (pm.write, pm.placeholder)):
            if node.op_name == layer_name and not found_layer:
                found_layer = True
                break

    if not found_layer:
        raise RuntimeError(f"Invalid layer name {layer_name} for extracting layer from {model_name}")

    genesys = define_genesys(def_cfg)
    if print_config:
        print(f"Compiling model with the following config:\n")
        sizes_cfg = def_cfg.copy()
        sizes_cfg['IBUF_SIZE'] = genesys.get_subgraph_node("IBUF").total_capacity
        sizes_cfg['WBUF_SIZE'] = genesys.get_subgraph_node("WBUF").total_capacity
        sizes_cfg['OBUF_SIZE'] = genesys.get_subgraph_node("OBUF").total_capacity
        sizes_cfg['BBUF_SIZE'] = genesys.get_subgraph_node("BBUF").total_capacity
        pprint(sizes_cfg)
    mode = "training" if train else "inference"
    # Codelet compilation starts here
    cdlts, impls = load_impls_cdlts(def_cfg)
    metadata = {'GENESYS_IMPLS': impls, 'GENESYS_CODELETS': cdlts,
                'FUSION_OP_INFO': load_fusion_op_info(def_cfg)}
    program = initialize_program(graph, genesys, metadata=metadata, mode=mode)
    program.add_compilation_step("template_pad_pass", template_pad_pass, template=True,
                                 dependencies=["template_layout_pass"])
    program.add_compilation_step("template_layout_pass", template_layout_pass, template=True)

    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True,
                                 stage_kwargs={'dtype_map': dtypes})
    program.add_compilation_step("remove_unused_variables", remove_unused_variables, preproc=True, stage_kwargs={'shaped_nodes': {}})
    tile_kwargs = {'factor_fn_name': factor_fn, 'stopping_condition': valid_split_stopping_condition,
                   'selection_metric': current_permutation_selection_metric, 'heuristic_fn': n_tiles_heuristic}
    program.add_compilation_step("tile", tile, stage_kwargs=tile_kwargs)
    program.add_compilation_step("separate_ops", separate_simd_sa_ops)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    if relocation_offsets:
        program.set_relocation_ns_offsets(relocation_offsets)
    program.compile(verbose=verbose, sequence_algorithm='filtered', filtered_layers=[layer_name])

    return program

