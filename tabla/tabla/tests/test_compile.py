from pathlib import Path
from backend.schedule import Schedule
from backend.tabla_template import TablaTemplate
from backend.component import Component
from tests.util import check_graph_order
from compiler import compile
import shutil
import json


def test_embedded_weights():
    Component.reset_ids()
    optimizations = {'reorder_instr': False,
                     'unused_ni_opt': True,
                     'apply_reuse': True}
    package_name = "backprop_20_20_5"
    # package_name = "reco_138_130_10"
    dfg_name = f"{package_name}.json"
    package_path = f"{Path(f'{__file__}').parent}/../{package_name}"

    file_path = f"tests/dfg_json_files/{dfg_name}"
    cfg_path = f'config.json'
    if Path(package_path).exists():
        shutil.rmtree(package_path)
    compile(file_path, cfg_path,
            f"{package_name}_input_data.txt",
            f"{package_name}_input_weights.txt",
            "meta.txt", sort_alg="custom",
            gen_sched_file=False,
            save_data=True,
            gen_mem_instr=True,
            debug=False,
            show_ns_utilization=["ND", "NI", "NW"],
            optimizations=optimizations)

def test_depth_order():
    Component.reset_ids()

    base_path = Path(f"{__file__}").parent
    package_name = "linear54"
    dfg_file = f"{package_name}.json"
    dfg_file_path = f"{base_path}/../{dfg_file}"
    sched = Schedule()
    sched.load_dfg(dfg_file_path, sort_type="custom")
    check_graph_order(sched)
    # new_arch = sched.schedule_graph(new_arch)
