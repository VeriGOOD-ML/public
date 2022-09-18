from dataclasses import dataclass, field
from collections import defaultdict, deque
from codelets.adl.operation import Loop, Compute, Configure, Transfer
from codelets.adl.flex_param import FlexParam
from functools import singledispatch
from codelets.adl.flex_template import Instruction, FlexTemplate
from typing import List, Dict, Union, Optional, Any
from copy import copy

from .dummy_op import DummyOp, DummyParam
from .operand_template import OperandTemplate, IndexOperandTemplate

INITIALIZER_FN_MAP = {'loop': Loop,
                      'config': Configure,
                      'transfer': Transfer,
                      'compute': Compute}

class OperationTemplate(object):

    BASE_ATTRS = ['loop_id', 'loop_level', 'op_id', 'global_op_id', 'target', 'operation_type',
                  'dependencies', 'param_symbols', 'target', 'operation_type', 'instructions',
                  'required_params', 'resolved_params', 'dependencies']
    BASE_KWARG_NAMES = ['codelet', 'target', 'instructions', 'resolved_params', 'param_symbols',
                        'add_codelet', 'dependencies']
    id_counter = 0
    op_id_counters = defaultdict(int)
    loop_ctx_dependencies = deque()
    loop_ctxt_level = 0
    loop_stack = deque()
    current_codelet = None


    def __init__(self, operation_type: str,
                 param_map: Dict[str, Union[FlexParam, int, None]],
                 codelet=None,
                 instructions: List[FlexTemplate] = None,
                 resolved_params: Dict[str, FlexParam] = None,
                 add_codelet=True,
                 dependencies=None):

        self._loop_id = OperationTemplate.current_loop_id()
        self._loop_level = copy(OperationTemplate.loop_ctxt_level)
        self._global_op_id = OperationTemplate.id_counter
        self._op_id = copy(OperationTemplate.op_id_counters[operation_type])
        if operation_type == "loop":
            self._loop_id = self._op_id

        OperationTemplate.op_id_counters[operation_type] += 1
        OperationTemplate.id_counter += 1
        self._dependencies = dependencies or []
        self._operation_type = operation_type
        self._instructions = instructions or []

        self._param_map = param_map
        self._resolved_params = resolved_params or {}
        assert self._loop_id >= 0 or self._operation_type == "config"
        codelet = codelet or OperationTemplate.current_codelet
        if add_codelet:
            codelet.add_op(self)

    @property
    def loop_id(self) -> int:
        return self._loop_id

    @loop_id.setter
    def loop_id(self, loop_id: int):
        self._loop_id = loop_id

    @property
    def param_map(self) -> Dict[str, Any]:
        return self._param_map

    @property
    def op_id(self) -> int:
        return self._op_id

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
        return OperationTemplate.loop_stack[-1]

    @property
    def op_str(self):
        return f"{self.op_type}{self.op_id}"

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        self._dependencies = dependencies

    @staticmethod
    def reset():
        OperationTemplate.id_counter = 0
        OperationTemplate.op_id_counters = defaultdict(int)
        OperationTemplate.loop_ctx_dependencies = deque()
        OperationTemplate.loop_ctxt_level = 0
        OperationTemplate.loop_stack = deque()
        OperationTemplate.current_codelet = None

    def evaluate(self, instance_args):
        return instance_args['CodeletTemplate'].op_map[self.op_str]

    @property
    def positional_args(self):
        raise NotImplementedError

    def instantiate(self, instance_args):
        args = []
        for i, param_name in enumerate(self.positional_args):
            param = self.param_map[param_name]
            use_dummy_str = self.arg_dummy_strings[i]
            args.append(self.evaluate_args(param, instance_args, use_dummy_str))
        args = tuple(args)
        kwargs = {}
        for key, value in self.param_map.items():
            if key in self.positional_args:
                continue
            else:
                kwargs[key] = self.evaluate_args(value, instance_args, False)

        kwargs['add_codelet'] = False

        instance = INITIALIZER_FN_MAP[self.op_type](*args, **kwargs)
        return instance

    def evaluate_args(self, args, instance_args, use_dummy_string):
        if use_dummy_string and isinstance(args, (DummyParam, DummyOp)):
            return args.name

        if isinstance(args, list):
            eval_arg = []
            for a in args:
                eval_arg.append(self.evaluate_args(a, instance_args, use_dummy_string))
        elif isinstance(args, tuple):
            eval_arg = []
            for a in args:
                eval_arg.append(self.evaluate_args(a, instance_args, use_dummy_string))
            eval_arg = tuple(eval_arg)
        elif isinstance(args, (DummyParam, DummyOp, IndexOperandTemplate)):
            eval_arg = args.evaluate(instance_args)
        elif isinstance(args, OperationTemplate):
            cdlt = instance_args['CodeletTemplate']
            eval_arg = cdlt.op_map[args.op_str]
        elif isinstance(args, OperandTemplate):
            cdlt = instance_args['CodeletTemplate']
            eval_arg = cdlt.get_operand(args.name)
        else:
            eval_arg = args

        return eval_arg

    def __str__(self):
        return f"{self.op_str}"

    def __repr__(self):
        return f"<op={self.op_type}, id={self.op_id}>"

    @property
    def arg_dummy_strings(self):
        raise NotImplementedError

class ConfigureTemplate(OperationTemplate):
    PARAM_KEYS = ['start_or_finish', 'target']
    USE_DUMMY_STRING = [False, False]

    def __init__(self, start_or_finish,
                 target,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['start_or_finish'] = start_or_finish
        param_map['target'] = target
        super(ConfigureTemplate, self).__init__("config", {**param_map, **kwargs}, add_codelet=add_codelet)

    @property
    def positional_args(self):
        return ConfigureTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ConfigureTemplate.USE_DUMMY_STRING

class ComputeTemplate(OperationTemplate):
    PARAM_KEYS = ['op_name', 'sources', 'dests', 'target']
    USE_DUMMY_STRING = [True, False, False, False]
    def __init__(self, op_name,
                 sources,
                 dests,
                 target,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['op_name'] = op_name
        param_map['target'] = target
        assert all([isinstance(s, (OperandTemplate, IndexOperandTemplate)) for s in sources])
        assert all([isinstance(d, (OperandTemplate, IndexOperandTemplate)) for d in dests])
        param_map['sources'] = sources
        param_map['dests'] = dests
        param_map['target'] = target
        super(ComputeTemplate, self).__init__("compute", {**param_map, **kwargs}, add_codelet=add_codelet)

    @property
    def operands(self):
        operands = []
        for o in (self.param_map['sources'] + self.param_map['dests']):
            if isinstance(o, IndexOperandTemplate):
                operands.append(o.operand)
            else:
                assert isinstance(o, OperandTemplate)
                operands.append(o)
        return operands

    @property
    def positional_args(self):
        return ComputeTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ComputeTemplate.USE_DUMMY_STRING

class TransferTemplate(OperationTemplate):
    PARAM_KEYS = ['operand', 'path']
    USE_DUMMY_STRING = [False, True]

    def __init__(self, operand: Union[OperandTemplate, IndexOperandTemplate],
                 path,
                 add_codelet=True,
                 **kwargs
                 ):
        assert isinstance(operand, (OperandTemplate)), f"Invalid type for operand {operand.name}.\n" \
                                                       f"Transferred operands cannot use index offsets.\n" \
                                                       f"Operand type: {type(operand)}"
        assert isinstance(path, (tuple, list)) and len(path) >= 2

        param_map = {}
        param_map['path'] = path
        param_map['operand'] = operand

        super(TransferTemplate, self).__init__("transfer", {**param_map, **kwargs}, add_codelet=add_codelet)

    @property
    def operand(self):
        return self.param_map['operand']

    @property
    def positional_args(self):
        return TransferTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return TransferTemplate.USE_DUMMY_STRING


class LoopTemplate(OperationTemplate):
    PARAM_KEYS = ['start', 'end', 'stride', 'offset']
    USE_DUMMY_STRING = [True, True, True, True]
    loop_ids = 0

    def __init__(self, start: Union[int, DummyOp, DummyParam],
                 end=None,
                 stride=1,
                 offset=0,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        if end is not None:
            param_map['start'] = start
            param_map['end'] = end
        else:
            param_map['start'] = 0
            param_map['end'] = start

        param_map['stride'] = stride
        param_map['offset'] = offset

        super(LoopTemplate, self).__init__("loop", {**param_map, **kwargs}, add_codelet=add_codelet)

    def __enter__(self):
        OperationTemplate.loop_ctxt_level += 1
        OperationTemplate.loop_stack.append(self.loop_id)
        OperationTemplate.loop_ctx_dependencies.append(self.op_str)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        OperationTemplate.loop_ctxt_level -= 1
        OperationTemplate.loop_stack.pop()
        OperationTemplate.loop_ctx_dependencies.pop()

    @property
    def positional_args(self):
        return LoopTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return LoopTemplate.USE_DUMMY_STRING

    def __mul__(self, other):
        return loop_op(other, self, '*')

    def __rmul__(self, other):
        return loop_op(other, self, '*', reflected=True)

    def __add__(self, other):
        return loop_op(other, self, '+')

    def __radd__(self, other):
        return loop_op(other, self, '+', reflected=True)

@singledispatch
def loop_op(op1, op2, op_str, reflected=False):
    raise NotImplementedError(f"No implementation for loop {op_str} op {type(op1)}.")

@loop_op.register(DummyOp)
def _(op1: DummyOp, op2: LoopTemplate, op_str: str, reflected=False):
    fp_name = f"loop_dummy{DummyOp.op_count}"
    if reflected:
        fn_body_str = f"{op1.name}{op_str}'{op2.op_str}'"
        fp = FlexParam(fp_name, fn_args=[f"{op2.op_str}"], fn_body_str=fn_body_str)
        dparam = DummyParam(fp, (op2,))
    else:
        fn_body_str = f"{op2.op_str}{op_str}'{op1.name}'"
        fp = FlexParam(fp_name, fn_args=[f"{op2.op_str}"], fn_body_str=fn_body_str)
        dparam = DummyParam(fp, (op2,))
    return dparam

@loop_op.register(DummyParam)
def _(op1: DummyParam, op2: LoopTemplate, op_str: str, reflected=False):
    fp_name = f"loop_dummy{DummyParam.op_count}"
    if reflected:
        fn_body_str = f"{op1.name}{op_str}{op2.op_str}"
        fp = FlexParam(fp_name, fn_args=[f"{op2.op_str}", f"{op1.name}"], fn_body_str=fn_body_str)
        dparam = DummyParam(fp, (op2, op1))
    else:
        fn_body_str = f"{op2.op_str}{op_str}{op1.name}"
        fp = FlexParam(fp_name, fn_args=[f"{op1.name}", f"{op2.op_str}"], fn_body_str=fn_body_str)
        dparam = DummyParam(fp, (op1, op2))
    return dparam

@loop_op.register(LoopTemplate)
def _(op1: LoopTemplate, op2: LoopTemplate, op_str: str, reflected=False):
    fp_name = f"loop_loop{DummyParam.op_count}"
    fn_body_str = f"{op1.op_str}{op_str}{op2.op_str}"
    fp = FlexParam(fp_name, fn_args=[f"{op1.op_str}", f"{op2.op_str}"], fn_body_str=fn_body_str)
    dparam = DummyParam(fp, (op1, op2))
    return dparam