import json

from backend.schedule import Schedule
from backend.tabla_template import TablaTemplate
from backend.component import Component
from backend.memory_interface import MemoryInstructionGenerator
from pathlib import Path

def test_memory_instruction_generator():
    Component.reset_ids()
    cwd = Path(f"{__file__}").parent
    # base_path = f"{cwd}/../benchmarks/dfgs/tabla_generated"
    base_path = f"{cwd}/test_dfgs"
    # dfg_name = "linear_784.json"
    dfg_name = "pm_linear3.json"
    file_path = f"{base_path}/{dfg_name}"

    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    test_sched = Schedule()
    test_sched.load_dfg(file_path)
    new_arch = test_sched.schedule_graph(new_arch)

    data = [edge for edge in test_sched._dfg_edges
            if edge.is_src_edge and edge.dtype == "input"]

    n_axi = 4
    n_lanes = 16
    pes_per_lane = 4

    meminst_gen = MemoryInstructionGenerator(data, n_axi, n_lanes, pes_per_lane, new_arch)
    meminst_gen.gen_inst(base_path)
    meminst_gen.gen_binary(base_path)