from typing import Dict, List

from codelets.codelet_impl import Codelet
from codelets.adl.operation.operand import Operand
from collections import namedtuple
import numpy as np
from .util import numpy_datagen

OperandData = namedtuple('OperandData', ['data', 'node_name', 'opname', 'idx', 'fmt'], defaults=[None])


def create_operand_data(data, operand, fmt=None):
    assert not isinstance(data, OperandData)
    assert not isinstance(data, Operand)
    return OperandData(data=data, opname=operand.name, node_name=operand.node_name, idx=operand, fmt=fmt)


class ReferenceOp(object):

    def __init__(self, cdlt, operands, outputs, program, scale=2):
        self._program = program
        self._cdlt = cdlt
        self._operands = operands
        self._outputs = outputs
        self._scale = scale

    @property
    def hag(self):
        return self.program.hag

    @property
    def program(self):
        return self._program

    @property
    def cdlt(self) -> Codelet:
        return self._cdlt

    @property
    def op_name(self) -> str:
        return self.cdlt.op_name

    @property
    def scale(self):
        return self._scale

    @property
    def operands(self) -> List[Operand]:
        return self._operands

    @property
    def outputs(self) -> List[Operand]:
        return self._outputs

    def fn_impl(self, inouts):
        raise NotImplemented

    def compute_outputs(self, inouts):


        inouts = inouts or {"inputs": [], "params": [], "outputs": []}
        inouts = self.set_operands(inouts)


        inouts = self.fn_impl(inouts)
        assert isinstance(inouts, dict)
        assert len(inouts['outputs']) == len(self.outputs)
        new_inouts = []
        for i, o in enumerate(inouts['outputs']):
            new_inouts.append(create_operand_data(o, self.outputs[i]))
        inouts['outputs'] = new_inouts
        assert all([isinstance(i, OperandData) for i in inouts['inputs']])
        assert all([isinstance(o, OperandData) for o in inouts['outputs']])

        assert all([isinstance(i.data, np.ndarray) for i in inouts['inputs']])
        assert all([isinstance(o.data, np.ndarray) for o in inouts['outputs']])
        return inouts

    def set_operands(self, inouts, constant_val=None, print_range=False):
        new_inputs = []
        for idx, op in enumerate(self.operands):
            found_inp = False
            for i in inouts['inputs']:
                if i.node_name == op.node_name:
                    found_inp = True
                    new_inputs.append(i)
                    break

            if found_inp:
                continue

            data = numpy_datagen(op.shape, op.dtype.bits(),
                                 fxp_dtype=f"{op.dtype}",
                                 scale=self.scale, constant_val=constant_val, print_range=print_range)
            new_op = create_operand_data(data, op)
            new_inputs.append(new_op)
        inouts['inputs'] = new_inputs
        return inouts

