import sys, os
import pprint
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
from backend import Schedule, TablaTemplate, Component, generate_pe_instructions
import json
from pathlib import Path
import pprint
from tests.util import check_graph_order
from compiler import compile
import shutil
import json

def schedule_embedded_weights():
    cleanup = False
    Component.reset_ids()

    package_name = "linear5"
    # package_name = "logistic2000"
    dfg_name = f"{package_name}.json"
    package_path = f"{Path(f'{__file__}').parent}/{package_name}"
    dfg_file_path = f"{Path(f'{__file__}').parent}/../{dfg_name}"
    base_path = f""

    file_path = f"{dfg_name}"
    cfg_path = f'config.json'
    if Path(package_path).exists():
        shutil.rmtree(package_path)
    compile(file_path, cfg_path,
            f"{package_name}_input_data.txt",
            f"{package_name}_input_weights.txt",
            "meta.txt", sort_alg="custom",
            gen_sched_file=True,
            save_data=True,
            debug=False,
            show_ns_utilization=["NI"])#




if __name__ == "__main__":
    schedule_embedded_weights()
