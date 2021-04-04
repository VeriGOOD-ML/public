from graphviz import Digraph
from pathlib import Path, PurePosixPath
import json
from collections import defaultdict

def check_graph_order(schedule):

    depth_map = defaultdict(list)
    for n in schedule._dfg_nodes:
        if len(n.parents) == 0:
            continue
        min_parent = min(n.parents)
        depth_map[n.depth].append(min_parent)
        if max(depth_map[n.depth]) != min_parent:
            print(f"Node id {n.node_id} is not minimum for depth {n.depth}\n\t"
                  f"Min: {max(depth_map[n.depth])}\n\t"
                  f"Node min: {min_parent}")



def visualize(filepath):
    outpath = Path(filepath).parent
    graph_name = PurePosixPath(filepath).stem
    with open(filepath, 'r') as f:
        graph = json.loads(f.read())
    out_graph = Digraph(graph_name)
    out_graph.attr(rankdir='LR')
    edges = graph["edges"]

    def edge_by_pair(src, dst):
        for e in edges:
            if e["src_id"] == src and e["dest_id"] == dst:
                return e
        raise KeyError(f"Could not find edge")
    for node in graph["nodes"]:
        out_graph.node(str(node["node_id"]), label=node_label(node))

        for pid in node["parents"]:
            edge = edge_by_pair(pid, node["node_id"])
            out_graph.edge(str(pid), str(node["node_id"]), label=edge_label(edge))

    name = f"{outpath}/viz_{graph_name}"
    out_graph.render(name, view=False)


def node_label(node):
    # if node['cat_component_id'] in [8,9,10]:
    instr = "" if node['instr'] is None else node['instr']
    label = f"Op: {node['op_name']}\n" \
        f"Instr {node['node_id']}: {instr}\n" \
        f"PE ID: {node['cat_component_id']}\n"
    return label
    # else:
    #     return f"PEID: {node['cat_component_id']}"

def edge_label(edge):
    label = f"Name:{edge['edge_id']}\n" \
        f"\tText Path:{edge['text_path']}\n" \
        f"\tComponent Path: {edge['path']}"
    return ""