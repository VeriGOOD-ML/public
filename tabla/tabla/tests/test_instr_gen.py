from backend.schedule import Schedule
import os
import pytest
import json
from backend.tabla_template import TablaTemplate
from backend.component import Component
from backend.schedule_validation import validate_instructions, validate_graph
from backend.instruction_gen import generate_pe_instructions
from pathlib import Path
from .util import visualize
from compiler import compile
import shutil

import pprint

def test_bp():
    Component.reset_ids()
    base_path = "./test_dfgs"
    package_name = "bp_dfg"
    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_instructions(new_arch)
    generate_pe_instructions(test_sched, new_arch, package_name)


def test_class():
    Component.reset_ids()
    base_path = "./test_dfgs"
    package_name = "class_dfg"
    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"
    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_instructions(new_arch)

    generate_pe_instructions(test_sched, new_arch, package_name)

def test_classification():
    Component.reset_ids()
    base_path = "./test_dfgs"
    package_name = "classification_dfg"
    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_instructions(new_arch)

    generate_pe_instructions(test_sched, new_arch, package_name)


def test_linear_dfg():
    Component.reset_ids()
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/test_dfgs"
    dfg_name = "linear_dfg.json"
    file_path = f"{base_path}/{dfg_name}"
    with open(f'{cwd}/config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_instructions(new_arch)

    # generate_pe_instructions(test_sched, new_arch)


def test_reco():
    Component.reset_ids()
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/test_dfgs"
    package_name = "reco_dfg"
    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"

    with open(f'{cwd}/config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule(optimize=False)
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_instructions(new_arch)

    generate_pe_instructions(test_sched, new_arch, package_name)


def test_benchmark_logistic():
    Component.reset_ids()
    package_name = "pm_linear55"
    cwd = Path(f"{__file__}").parent
    # base_path = f"{cwd}/../benchmarks/dfgs/tabla_generated"
    base_path = f"{cwd}/test_dfgs"

    # dfg_name = "linear_784.json"
    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"

    with open(f'{cwd}/config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule(optimize=True)
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)

    test_sched.print_schedule_graph(f"{cwd}/test_outputs/schedule_{dfg_name}")
    # pprint.pprint(new_arch.namespace_utilization())

    generate_pe_instructions(test_sched, new_arch, package_name)

def test_visualize_instr():
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/test_outputs"
    dfg_name = "schedule_linear_55.json"
    file_path = f"{base_path}/{dfg_name}"
    visualize(file_path)

def test_cfg_template():
    from backend.vlg_templates import cfg_template
    import json
    import math

    cfg_dict = {}
    cwd = Path(f"{__file__}").parent

    with open(f"{cwd}/config.json", "r") as read_file:
        data = json.load(read_file)
    # print(data.keys())
    cfg_dict['ns_size'] = data['namespace_size']
    cfg_dict['index_inst'] = 10
    cfg_dict['bus_read_depth'] = 512
    cfg_dict['bus_fifo_depth'] = 512
    cfg_dict['nb_fifo_depth'] = 256
    cfg_dict['num_pe'] = data['num_pes']
    cfg_dict['log_num_pe'] = int(math.log2(data['num_pes']))
    cfg_dict['log_num_pu'] = int(math.log2(data['num_pes']//data['pes_per_pu']))
    cfg_dict['log_mem_ns'] = 2
    cfg_dict['program_name'] = "linear3"

    print(cfg_template.format(**cfg_dict))


def test_compile():
    cleanup = True
    Component.reset_ids()
    package_name = "pm_linear55"

    base_path = f"tests/test_dfgs"

    dfg_name = f"{package_name}.json"
    file_path = f"{base_path}/{dfg_name}"
    cfg_path = f'tests/config.json'
    compile(file_path, cfg_path, "../input_data.txt", "../input_weights.txt")
    if cleanup:
        shutil.rmtree(f"{base_path}/{package_name}")


