from pathlib import Path
import polymath as pm

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

def get_single_conv2d_node():
    resnet18 = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")
    for name, node in resnet18.nodes.items():
        if node.name == "conv":
            return node
