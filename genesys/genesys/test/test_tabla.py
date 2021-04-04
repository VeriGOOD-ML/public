from codelets.examples.tabla import generate_tabla
import pytest
from collections import defaultdict
import networkx as nx
@pytest.mark.parametrize('num_pus, pes_per_pu, ni_size, nw_size, nd_size',[
    # (8, 8, 128, 128, 128),
    (1, 4, 128, 128, 128),
])
def test_tabla(num_pus, pes_per_pu, ni_size, nw_size, nd_size):
    adl_graph = generate_tabla(num_pus, pes_per_pu, ni_size, nw_size, nd_size)
    adl_graph._networkx_visualize("tabla")

def test_nx_clusters():
    adl = nx.MultiDiGraph(compound=True)

    adl.add_node("a", outer_graph="A")
    adl.add_node("b", outer_graph="A")
    adl.add_node("c", outer_graph="A")
    adl.add_edge("a", "b")
    adl.add_edge("a", "c")

    adl.add_node("d", outer_graph="B")
    adl.add_node("e", outer_graph="B")
    adl.add_edge("d", "e")
    adl.add_edge("c", "d")
    subgraphs = defaultdict(list)

    for n in adl.nodes:
        if 'outer_graph' in adl.nodes[n]:
            subgraphs[adl.nodes[n]['outer_graph']].append(n)

    gviz = nx.nx_agraph.to_agraph(adl)
    for k, v in subgraphs.items():
        _ = gviz.subgraph(v, name=f"cluster_{k}", label=f"{k}")
    gviz.layout(prog='dot')
    gviz.draw(f"test.pdf", format="pdf")