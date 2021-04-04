import pytest

from codelets.adl.operation.base_op import Operation
from codelets.adl.operation.loop_op import Loop


def test_loop_level():
    with Loop("i", 0, 5) as l1:
        assert l1.loop_level == 0
        with Loop("j", 1, 5) as l2:
            assert l2.loop_level == 1

        with Loop("i", 1, 5) as l3:
            assert l3.loop_level == 1