from backend import Schedule
import os
import pytest
import json
from backend import TablaTemplate
from backend import Component
from backend import validate_graph
import pprint

def test_load_schedule():
    base_path = "./test_dfgs"
    test_files = os.listdir(base_path)
    for file in test_files:
        if os.path.isfile(file) and file[:-4] == ".json":
            test_sched = Schedule()
            test_sched.load_dfg(f"{base_path}/{file}")

def test_get_node():
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    sched_node = test_sched.get_schedule_node(0)
    assert sched_node.op_name == 'source'
    with pytest.raises(KeyError):
        test_invalid_node = test_sched.get_schedule_node(1000)

def test_data_node():
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    sched_node = test_sched.get_schedule_node(2)
    non_data_node = test_sched.get_schedule_node(11)
    assert sched_node.is_data_node()
    assert not non_data_node.is_data_node()

def test_src_child_edge():
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    sched_edge_id = test_sched.get_parent_edge_id(10, 3)
    sched_edge = test_sched.get_schedule_edge(sched_edge_id)
    assert sched_edge.is_src_edge

def test_create_schedule():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    validate_graph(test_sched, new_arch)

def test_data_insertion():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    pe_id = new_arch.cat_component_map['pe'][0].component_id
    sched_edge = test_sched.get_schedule_edge(14)
    sched_edge.set_ready_cycle(0)
    sched_edge.add_source_component(pe_id)
    sched_edge.add_dest_component(pe_id)
    _ = new_arch.add_namespace_data(1, pe_id, sched_edge.namespace_name, sched_edge)

def test_initialized_namespaces():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/linear_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    init_nd = new_arch.cat_component_map['pe'][1].get_namespace('ND')
    nd_storage = init_nd.get_cycle_storage()
    assert nd_storage[0].src_id == 11

    init_nw = new_arch.cat_component_map['pe'][1].get_namespace('NW')
    nw_storage = init_nw.get_cycle_storage()
    assert nw_storage[0].src_id == 18


    init_nd = new_arch.cat_component_map['pe'][2].get_namespace('ND')
    nd_storage = init_nd.get_cycle_storage()
    assert nd_storage[0].src_id == 13

    init_nw = new_arch.cat_component_map['pe'][2].get_namespace('NW')
    nw_storage = init_nw.get_cycle_storage()
    assert nw_storage[0].src_id == 20

def test_node_depth():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/logistic_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    sched_node = test_sched.get_schedule_node(15) # This is the sigmoid operation node

    assert sched_node.depth == 5

def test_graph_width():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/logistic_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)

    assert test_sched.get_max_width() == 3

def test_path_creation():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/logistic_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)


def test_schedule_printing():
    Component.reset_ids()
    base_path = "./test_dfgs"
    file_path = f"{base_path}/logistic_dfg.json"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)

    test_sched.print_schedule_graph("./test_outputs/schedule_logistic_dfg.json")
#

def test_bp():
    Component.reset_ids()
    base_path = "./test_dfgs"
    dfg_name = "bp_dfg.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
    validate_graph(test_sched, new_arch)
    print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
    pprint.pprint(new_arch.pu_utilization())



def test_class():
    Component.reset_ids()
    base_path = "./test_dfgs"
    dfg_name = "class_dfg.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
    validate_graph(test_sched, new_arch)
    print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
    pprint.pprint(new_arch.pu_utilization())

def test_classification():
    Component.reset_ids()
    base_path = "./test_dfgs"
    dfg_name = "classification_dfg.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
    validate_graph(test_sched, new_arch)
    print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
    pprint.pprint(new_arch.pu_utilization())


def test_linear_dfg():
    Component.reset_ids()
    base_path = "./test_dfgs"
    dfg_name = "linear_dfg.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
    validate_graph(test_sched, new_arch)
    print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
    pprint.pprint(new_arch.pu_utilization())


def test_reco():
    Component.reset_ids()
    base_path = "./test_dfgs"
    dfg_name = "reco_dfg.json"
    file_path = f"{base_path}/{dfg_name}"

    with open('config.json') as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    test_sched.schedule_graph(new_arch)
    test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
    validate_graph(test_sched, new_arch)
    print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
    pprint.pprint(new_arch.pu_utilization())


# def test_benchmark_logistic():
#     Component.reset_ids()
#     base_path = "../benchmarks/dfgs/tabla_generated"
#     dfg_name = "logistic_2000.json"
#     file_path = f"{base_path}/{dfg_name}"
#
#     with open('config.json') as config_file:
#         data = json.load(config_file)
#
#     new_arch = TablaTemplate(data)
#     test_sched = Schedule()
#     test_sched.load_dfg(file_path)
#     test_sched.schedule_graph(new_arch)
#     validate_graph(test_sched, new_arch)
#     # test_sched.print_schedule_graph(f"./test_outputs/schedule_{dfg_name}")
#     print(f"Graph width: {test_sched.get_max_width()}\nAverage pe utilization: {new_arch.pe_utilization()}")
#     pprint.pprint(new_arch.pu_utilization())

