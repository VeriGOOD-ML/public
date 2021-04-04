import json
import math
import os
from .backend.schedule import Schedule, topological_sort
from .backend.tabla_template import TablaTemplate
from .backend.component import Component
from .backend.instruction_gen import generate_pe_instructions
from .backend.memory_interface import generate_memory_instr, dump_output_weights_in_axi
from .backend.vlg_templates import cfg_template
from pathlib import Path
import pprint
import zipfile

import numpy as np
ALL_OPTS = {"reorder_instr": True, "unused_ni_opt": True, "apply_reuse": True}
def create_dirs(fpath):
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/../benchmarks/compilation_output/{Path(fpath).stem}"

    try:
        os.mkdir(base_path)
    except OSError as e:
        print(f"Creating of directory {base_path} failed:\n {e}")
    else:
        print(f"Successfully created of directory {base_path}")

    # Create mem-inst directory
    os.mkdir(f"{base_path}/mem-inst")

    # Create compute inst directory
    os.mkdir(f"{base_path}/compute-inst")
    os.mkdir(f"{base_path}/mem-inst/weights")
    os.mkdir(f"{base_path}/mem-inst/axi")


    return base_path


def create_config(json_config, ppath):
    cfg_dict = {}
    with open(json_config, "r") as read_file:
        data = json.load(read_file)
    cfg_dict['ns_size'] = data['namespace_size']
    cfg_dict['index_inst'] = 10
    cfg_dict['bus_read_depth'] = 512
    cfg_dict['bus_fifo_depth'] = 512
    cfg_dict['nb_fifo_depth'] = 256
    cfg_dict['num_pe'] = data['num_pes']
    cfg_dict['log_num_pe'] = int(math.log2(data['num_pes']))
    cfg_dict['log_num_pu'] = int(math.log2(data['num_pes']//data['pes_per_pu']))
    cfg_dict['log_mem_ns'] = 2
    cfg_dict['program_name'] = ppath.rsplit("/", 1)[-1]
    cfg_out = cfg_template.format(**cfg_dict)

    with open(f"{ppath}/config.vh", "w") as config_write:
        config_write.write(cfg_out)
    return data


def package_code(fpath, config_path):
    base_path = create_dirs(fpath)
    cfg_data = create_config(config_path, base_path)
    return base_path, cfg_data


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

            # returning all file paths
    return file_paths

def compress_folder(fpath):
    file_paths = get_all_file_paths(fpath)

    # printing the list of all files to be zipped
    print('Following files will be zipped:')
    for file_name in file_paths:
        print(file_name)

        # writing files to a zipfile
    with zipfile.ZipFile(f"{fpath}.zip", 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)

def store_dfg_data(sched, arch, package_path, sort_inputs=False):

    # inputs = []
    # weights = []
    # outputs = []
    # y = []
    package_name = sched.graph_name
    input_data = {}

    weight_data = {}
    outputs = {}
    for i, n in enumerate(sched._dfg_nodes):
        if n.dtype == "input":
            pe = arch.component_map[n.component_id]
            dindex = pe.get_namespace("ND").find_data_index(n.node_id)
            if sort_inputs:
                input_data[n.node_id] = (n.computed, pe.category_id, dindex, n.op_name)
            else:
                input_data[n.node_id] = (n.computed, i, i, n.op_name)

        elif n.dtype == "state":
            if n.parents == [0]:
                pe = arch.component_map[n.component_id]
                dindex = pe.get_namespace("NW").find_data_index(n.node_id)
                if sort_inputs:
                    weight_data[n.node_id] = (n.computed, pe.category_id, dindex, n.op_name)
                else:
                    weight_data[n.node_id] = (n.computed, i, i, n.op_name)
            else:
                assert n.children == [1]
                pe = arch.component_map[n.component_id]
                nid = n.parents[0] if sched.get_schedule_node(n.parents[0]).is_data_node else n.parents[1]
                dindex = pe.get_namespace("NW").find_data_index(nid)
                outputs[n.node_id] = (n.computed, pe.category_id, dindex, sched.get_schedule_node(nid).op_name)

    input_data_arr = sorted([k for k in input_data.keys()], key=lambda x: (input_data[x][2], input_data[x][1]))
    input_data_arr = np.asarray([input_data[k][0] for k in input_data_arr])

    weight_arr = sorted([k for k in weight_data.keys()], key=lambda x: (weight_data[x][2], weight_data[x][1]))
    weight_arr = np.asarray([weight_data[k][0] for k in weight_arr])


    output_arr = sorted([k for k in outputs.keys()], key=lambda x: (outputs[x][2], outputs[x][1]))

    output_arr = np.asarray([outputs[k][0] for k in output_arr])

    np.savetxt(f"{package_path}/{package_name}_input_data.txt", input_data_arr.astype(np.int), fmt="%d", delimiter="\n")
    np.savetxt(f"{package_path}/{package_name}_input_weights.txt", weight_arr.astype(np.int), fmt="%d", delimiter="\n")
    np.savetxt(f"{package_path}/{package_name}_output_weights.txt", output_arr.astype(np.int), fmt="%d", delimiter="\n")

def get_max_util(util_dict):
    maxes = {"NI": -1, "ND": -1, "NW": -1}
    for pe, ns_util in util_dict.items():
        for ns_name, util in ns_util.items():
            if util[1][0] > maxes[ns_name]:
                maxes[ns_name] = util[1][0]
    return maxes

def generate_summary(dir_path, arch, summary_info):
    util = arch.namespace_utilization(namespaces=["NI", "NW", "ND"])
    max_utils = get_max_util(util)
    instr_info = arch.ni_read_writes()
    with open(f"{dir_path}/summary.txt", "w") as summary_write:
        summary_write.write(f"Total compute Instructions:\t{summary_info['instr_count']}\n")
        summary_write.write(f"Total mem Instructions:\t{summary_info['mem_instr_count']}\n")
        summary_write.write(f"Total number of inputs:\t{summary_info['num_inputs']}\n")
        summary_write.write(f"Max NI Utilization:\t{max_utils['NI']}\n")
        summary_write.write(f"Max ND Utilization:\t{max_utils['ND']}\n")
        summary_write.write(f"Max NW Utilization:\t{max_utils['NW']}\n")
        summary_write.write(f"{summary_info['instr_summary']}\n")

    lines = ["PEID,NI,NW,ND,INSTR"]
    log_lines = ["PEID,NI,NW,ND,INSTR"]
    logsize = lambda x: int(np.ceil(np.log2(x))) if x > 0 else 0
    totals = ['Totals', 0, 0, 0, 0]
    max_instr = -1
    log_lines.append(f"MAX,{logsize(max_utils['NI'])},{logsize(max_utils['NW'])},{logsize(max_utils['ND'])}")
    for pe_id, pe in arch.category_component_dict['pe'].items():
        ni_util = util['PE' + str(pe_id)]['NI'][1][0]
        nw_util = util['PE' + str(pe_id)]['NW'][1][0]
        nd_util = util['PE' + str(pe_id)]['ND'][1][0]
        num_instr = len(pe.all_instructions())
        totals[1] += ni_util
        totals[2] += nw_util
        totals[3] += nd_util
        totals[4] += num_instr
        max_instr = num_instr if num_instr > max_instr else max_instr
        lines.append(f"{pe_id},{ni_util},{nw_util},{nd_util},{num_instr}")
        log_lines.append(f"{pe_id},{logsize(ni_util)},{logsize(nw_util)},{logsize(nd_util)}")
    lines.insert(1, f"MAX,{max_utils['NI']},{max_utils['NW']},{max_utils['ND']},{max_instr}")

    lines.append(f"{totals[0]},{totals[1]},{totals[2]},{totals[3]},{totals[4]}")
    with open(f"{dir_path}/utilization.txt", "w") as util_file:
        util_file.write("\n".join(lines))

    with open(f"{dir_path}/log_utilization.txt", "w") as util_file:
        util_file.write("\n".join(log_lines))

    instr_lines = []
    avg_unused = 0.0
    for pe_id, ns_dict in instr_info.items():
        if len(ns_dict.keys()) > 0:

            total_unused = 0
            iline = f"PE{pe_id}:\n"
            for index, val in ns_dict.items():
                if val.finish == -1:
                    total_unused += 1
                iline += f"\tNI{index} ({val.meta}): {val.start}  ---->  {val.finish}\n"
                avg_unused += 100.0*(total_unused/len(ns_dict.keys()))
            iline += f"At least {total_unused}/{len(ns_dict.keys())} ({100*(total_unused/len(ns_dict.keys()))}%), namespace slots\n"
            instr_lines.append(f"{iline}")
    if len(instr_info.keys()) > 0:
        instr_lines.append(f"\nAverage unused namespace: {avg_unused/len(instr_info.keys())}%")
    with open(f"{dir_path}/ns_read_write.txt", "w") as instr_info_file:
        instr_info_file.write("\n".join(instr_lines))


def compile(dfg_file, config_path, input_data_file, input_weights_file, meta_file,
            save_data=False,
            gen_sched_file=False,
            compress=False,
            sort_alg=None,
            debug=False,
            show_ns_utilization=None,
            gen_mem_instr=True,
            progress_bar=True,
            optimizations=None,
            is_training_algorithm=True):
    summary = {}
    Component.reset_ids()
    if Path(dfg_file).is_absolute():
        base_path = f"{Path(dfg_file).parent}/compilation_output"
        dfg_file_path = dfg_file
    else:
        base_path = Path(f"{__file__}").parent
        base_path = f"{base_path}/compilation_output"
        dfg_file_path = f"{base_path}/{dfg_file}"

    if not Path(f"{base_path}").exists():
        os.mkdir(base_path)

    package_path, config_data = package_code(dfg_file, config_path)
    new_arch = TablaTemplate(config_data)


    sched = Schedule(new_arch, debug=debug, progress_bar=progress_bar, is_training_algorithm=is_training_algorithm)

    sched.load_dfg(dfg_file_path, sort_type=sort_alg)

    if optimizations:
        opts = optimizations
    else:
        opts = ALL_OPTS
    new_arch, ninputs = sched.schedule_graph(new_arch, **opts)


    print(f"Finished scheduling\n")
    summary['num_inputs'] = ninputs
    summary['instr_count'], summary['instr_summary'] = generate_pe_instructions(sched, new_arch,
                                                                                package_path, use_instr_list=False)
    print(f"Finished generating instructions\n")

    data = [edge for edge in sched._dfg_edges
            if edge.is_src_edge and edge.dtype == "input"]

    if save_data:
        store_dfg_data(sched, new_arch, package_path, sort_inputs=False)
        input_data_file = f"{package_path}/{sched.graph_name}_input_data.txt"
        input_weights_file = f"{package_path}/{sched.graph_name}_input_weights.txt"
    print(f"Finished storing data instructions\n")
    summary['mem_instr_count'] = 0
    if gen_mem_instr:
        summary['mem_instr_count'] = generate_memory_instr(package_path, sched, new_arch, config_data, input_data_file,
                              input_weights_file,
                              meta_file,
                              debug=debug)
        print(f"Finished generating memory instructions\n")
        dump_output_weights_in_axi(package_path, sched, new_arch, config_data, debug=debug)

    if gen_sched_file:
        sched.print_schedule_graph(f"{package_path}/sched_{sched.graph_name}.json")
    if compress:
        compress_folder(package_path)

    if show_ns_utilization:
        print(f"Memory Utilization Across Namespaces:")
        util = new_arch.namespace_utilization(namespaces=show_ns_utilization)
        max_utils = get_max_util(util)
        pprint.pprint(max_utils)
    generate_summary(package_path, new_arch, summary)




