import polymath as pm
from pathlib import Path
from codelets import compile
import json
from codelets.examples.genesys import generate_genesys

CWD = Path(f"{__file__}").parent
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

BENCH_DIR = f"{CWD}/input_files"

def parse_cfg():
    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def get_genesys():
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)
    return genesys

def test_resnet18():
    graph = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")
    # hag = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)
    hag = get_genesys()
    compile(graph, hag, f"{BENCH_DIR}")


def test_lenet():
    graph = pm.pb_load(f"{BENCH_DIR}/lenet.srdfg")
    # hag = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)
    hag = get_genesys()
    compile(graph, hag, f"{BENCH_DIR}")

def test_tiling_parameterization():
    pass