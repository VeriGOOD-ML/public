from codelets.examples.genesys import define_genesys
from collections import namedtuple
from pathlib import Path

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

if __name__ == "__main__":
    genesys = define_genesys("transformation")
    pe_array = genesys.get_subgraph_node("pe_array")
    print(pe_array.dimensions)