import pytest
import numpy as np
from backend import VectorMul
import pprint

@pytest.mark.parametrize('n, npes, npus',[
    (18, 16, 4)
])
def test_vmul(n, npes, npus):
    a_in = []
    b_in = []
    for i in list(range(0, 2*n)):
        if i % 2 == 0:
            a_in.append(i)
        else:
            b_in.append(i)

    inp_node = {"id": 3,
                "operation": "vmul",
                "parents": [a_in, b_in],
                "dataType": None,
                "children": [1]
                }
    vmul_node = VectorMul(inp_node, npes, npus, n)
    vmul_node.draw_graph()


