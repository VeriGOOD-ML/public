from dataclasses import dataclass, field
from collections import defaultdict, deque
from codelets.adl.flex_param import FlexParam
from codelets.adl.flex_template import Instruction, FlexTemplate
from typing import List, Dict, Union, Optional
from sympy import Basic, Idx, IndexedBase, Expr
from copy import copy

class Operation(object):

    BASE_ATTRS = ['loop_id', 'loop_level', 'op_id', 'global_op_id', 'target', 'operation_type',
                  'dependencies', 'param_symbols', 'target', 'operation_type', 'instructions',
                  'required_params', 'resolved_params', 'dependencies', 'split_map']
    BASE_KWARG_NAMES = ['codelet', 'target', 'instructions', 'resolved_params', 'param_symbols',
                        'add_codelet', 'dependencies']
    id_counter = 0
    op_id_counters = defaultdict(int)
    loop_ctx_dependencies = deque()
    loop_ctxt_level = 0
    loop_stack = deque()
    current_codelet = None


    def __init__(self, operation_type: str,
                 required_params: List[str],
                 codelet=None,
                 target: str = None,
                 instructions: List[FlexTemplate] = None,
                 resolved_params: Dict[str, FlexParam] = None,
                 param_symbols: Dict[str, Basic] = None,
                 add_codelet=True,
                 dependencies=None):

        self._loop_id = Operation.current_loop_id()
        self._loop_level = copy(Operation.loop_ctxt_level)
        self._global_op_id = Operation.id_counter
        self._op_id = copy(Operation.op_id_counters[operation_type])
        if operation_type == "loop":
            self._loop_id = self._op_id

        Operation.op_id_counters[operation_type] += 1
        Operation.id_counter += 1
        self._dependencies = dependencies or []

        self._param_symbols = param_symbols or {}
        self._target = target
        self._operation_type = operation_type
        self._instructions = instructions or []

        self._required_params = required_params
        self._resolved_params = resolved_params or {}
        self._split_map = {}
        assert self._loop_id >= 0 or self._operation_type == "config"
        codelet = codelet or Operation.current_codelet
        if add_codelet:
            codelet.add_op(self)

        for r in self.required_params:
            if r not in codelet.required_params:
                codelet.add_required_param(r)

    @property
    def loop_id(self) -> int:
        return self._loop_id

    @loop_id.setter
    def loop_id(self, loop_id: int):
        self._loop_id = loop_id

    @property
    def required_params(self) -> List[str]:
        return self._required_params

    @property
    def target(self) -> str:
        return self._target

    @property
    def op_id(self) -> int:
        return self._op_id

    @property
    def split_map(self):
        return self._split_map

    @op_id.setter
    def op_id(self, op_id: int):
        self._op_id = op_id

    @property
    def resolved_params(self):
        return self._resolved_params

    @property
    def global_op_id(self) -> int:
        return self._global_op_id

    @global_op_id.setter
    def global_op_id(self, global_op_id: int):
        self._global_op_id = global_op_id

    @property
    def instructions(self) -> List[FlexTemplate]:
        return self._instructions

    @property
    def loop_level(self):
        return self._loop_level

    @loop_level.setter
    def loop_level(self, loop_level: int):
        self._loop_level = loop_level

    @property
    def is_template(self) -> bool:
        return self.loop_id is not None

    @property
    def op_type(self) -> str:
        return self._operation_type

    @staticmethod
    def current_loop_id():
        return Operation.loop_stack[-1]

    @property
    def op_str(self):
        return f"{self.op_type}{self.op_id}"

    @property
    def param_symbols(self):
        return self._param_symbols

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        self._dependencies = dependencies


    def __str__(self):
        op_str = f"{self.op_type}{self.op_id} -> {self.target}, PARAMS: {list(self.required_params)}, " \
                 f"{self.op_type.upper()}PARAMS: {self.op_type_params()}"
        return op_str

    def set_required_param(self, key, param):
        assert isinstance(param, FlexParam)
        value = param.value
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params}\n"
                           f"Updated value: {value}")
        if key in self.resolved_params and self.resolved_params[key].value != value:
            raise RuntimeError(f"Param {key} has already been set:\n"
                               f"Previous value: {self.resolved_params[key]}\n"
                               f"New value: {value}")

        if value is None:
            raise ValueError(f"Cannot set required parameter to None value:\n"
                             f"Value: {value}\n"
                             f"Key: {key}")
        self.resolved_params[key] = param

    def set_split_mapping(self, level: int, loop_id: str):
        self._split_map[level] = loop_id

    def unset_params(self):
        unset_params = []
        for k in self.required_params:
            if k not in self.resolved_params:
                unset_params.append(k)
        return unset_params

    def set_template(self, template: List[FlexTemplate]):
        self._instructions = template

    def op_type_params(self):
        raise NotImplementedError

    def emit(self, output_type: str):
        raise NotImplementedError

    def evaluate_parameters(self, node, hag, cdlt):
        raise NotImplementedError

    def copy(self, cdlt, **kwargs):
        obj = type(self).__new__(self.__class__)
        for a_key in Operation.BASE_ATTRS:
            parg_key = f"_{a_key}"
            if a_key in kwargs:
                obj.__dict__[parg_key] = kwargs[a_key]
            else:
                a = self.__dict__[parg_key]
                if isinstance(a, (str, int)) or a is None:
                    obj.__dict__[parg_key] = a
                elif isinstance(a, dict):
                    obj.__dict__[parg_key] = {}
                    for k, v in a.items():
                        obj.__dict__[parg_key][k] = copy(v)
                else:
                    assert isinstance(a, list)
                    obj.__dict__[parg_key] = []
                    for v in a:
                        obj.__dict__[parg_key].append(copy(v))
        return obj




