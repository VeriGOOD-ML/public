from typing import Dict, Any, Tuple, List, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from functools import singledispatch
from . import TEMPLATE_CLASS_ARG_MAP
from itertools import count
from typing import ClassVar
import inspect
from codelets import FXP_CONFIGS
from fxpmath import Fxp


from codelets.adl.flex_param import FlexParam

if TYPE_CHECKING:
    from .operation_template import LoopTemplate

flex_param_cnt = count()

@dataclass
class DummyOp:
    # This is a placeholder for templates. Each time a node method is called on this,
    # a function is generated to preserve the function call for when the value can be computed during
    # instantiation.
    template_types: List[str]
    flex_param: FlexParam
    obj_instance: Any = field(default=None)
    dtype: str = field(default=None)
    op_count: ClassVar[int] = 0

    def __post_init__(self):
        DummyOp.op_count += 1

    @property
    def name(self):
        return self.flex_param.name

    @property
    def value(self):
        return self.flex_param.value

    def update(self, dummy_op: 'DummyOp'):
        self.template_types = dummy_op.template_types
        self.flex_param.update_fn_code_args(dummy_op.flex_param.fn_args,
                                            dummy_op.flex_param.fn_body_str)
        self.obj_instance = dummy_op.obj_instance
        return dummy_op

    def __getattr__(self, name):
        # TODO: Add assertion here to make sure its a valid attribute
        new_code = f"{self.flex_param.fn_body_str}.{name}"
        self.flex_param.update_fn_code(new_code)
        return self

    def __getstate__(self):
        state = {'flexparam': self.flex_param.to_json(), 'template_types': self.template_types}
        return state

    def __setstate__(self, state):
        self.flex_param = FlexParam(name=state['flexparam']['name'], fn_args=state['flexparam']['args'], fn_body_str=state['flexparam']['body'])
        self.template_types = state['template_types']

    def __getitem__(self, key):
        if isinstance(key, str):
            key = f"'{key}'"
        new_code = f"{self.flex_param.fn_body_str}[{key}]"
        self.flex_param.update_fn_code(new_code)
        return self

    def __len__(self):
        new_code = f"len({self.flex_param.fn_body_str})"
        self.flex_param.update_fn_code(new_code)
        return self

    def evaluate(self, instance_args):
        obj_instances = []
        for t in self.template_types:
            if instance_args[t] not in obj_instances:
                obj_instances.append(instance_args[t])
        obj_instances = tuple(obj_instances)

        if self.dtype is not None:
            assert isinstance(self.dtype, str)
            self.flex_param.dtype_cast_func = lambda x: Fxp(x, **FXP_CONFIGS[self.dtype]).val.item()
        res = self.flex_param.evaluate_fn(*obj_instances, force_evaluate=True)

        return res

    def __mul__(self, other):
        return dummy_op(other, self, '*')

    def __rmul__(self, other):
        return dummy_op(other, self, '*', reflected=True)

    def __add__(self, other):
        return dummy_op(other, self, '+')

    def __radd__(self, other):
        return dummy_op(other, self, '+', reflected=True)

    def __sub__(self, other):
        return dummy_op(other, self, '-')

    def __rsub__(self, other):
        return dummy_op(other, self, '-', reflected=True)

    def __truediv__(self, other):
        return dummy_op(other, self, '/')

    def __rdiv__(self, other):
        return dummy_op(other, self, '/', reflected=True)

    def __rtruediv__(self, other):
        return dummy_op(other, self, '/', reflected=True)

    def __floordiv__(self, other):
        return dummy_op(other, self, '//')

    def __rfloordiv__(self, other):
        return dummy_op(other, self, '//', reflected=True)

    def __mod__(self, other):
        return dummy_op(other, self, '%')

    def __rmod__(self, other):
        return dummy_op(other, self, '%', reflected=True)

    def __lt__(self, other):
        return dummy_op(other, self, '<')

    def __le__(self, other):
        return dummy_op(other, self, '<=')

    def __gt__(self, other):
        return dummy_op(other, self, '>')

    def __ge__(self, other):
        return dummy_op(other, self, '>=')

    def max(self, other):
        if isinstance(other, int):
            template_type_str = self.template_types
            arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]
            lhs = self.flex_param.fn_body_str
            rhs = other
            fp_name = f"max({lhs}, {rhs})"
            fn_str = f"max({lhs}, {rhs})"
            fp = FlexParam(fp_name, arg_str, fn_str)
            return DummyOp(template_type_str, fp)
        else:
            assert isinstance(other, DummyOp)
            template_type_str = list(set(self.template_types + other.template_types))
            arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]

            fp_name = f"max({self.name},{other.name})"
            fn_str = f"max({self.flex_param.fn_body_str}, {other.flex_param.fn_body_str})"
            fp = FlexParam(fp_name, arg_str, fn_str)
            return DummyOp(template_type_str, fp)


@dataclass
class DummyParam:
    # This is a placeholder for templates. Each time a node method is called on this,
    # a function is generated to preserve the function call for when the value can be computed during
    # instantiation.
    flex_param: FlexParam
    dummy_args: Tuple[Union[DummyOp, 'LoopTemplate', 'DummyParam']]
    obj_instance: Any = field(default=None)
    op_count: ClassVar[int] = 0

    def __post_init__(self):
        DummyParam.op_count += 1

    @property
    def name(self):
        return self.flex_param.name

    @property
    def value(self):
        return self.flex_param.value

    def evaluate(self, instance_args):
        args = []
        for d in self.dummy_args:
            res = d.evaluate(instance_args)
            args.append(res)
        return self.flex_param.evaluate_fn(*tuple(args), force_evaluate=True)

    def get_full_obj_type(self, obj):
        obj_mro = inspect.getmro(obj.__class__)
        assert len(obj_mro) >= 2
        base = obj_mro[-2]
        name = f"{base.__module__}.{base.__name__}"
        return name

# TODO: Need to fix the argument names here, this is extremely confusing
@singledispatch
def dummy_op(op1, op2, op_str, reflected=False):
    raise NotImplementedError(f"No implementation for loop {op_str} op {type(op1)}.")

@dummy_op.register(DummyOp)
def _(op1: DummyOp, op2: DummyOp, op_str: str, reflected=False):
    template_type_str = list(set(op1.template_types + op2.template_types))
    arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]
    if reflected:
        lhs = op1
        rhs = op2
    else:
        lhs = op2
        rhs = op1
    fp_name = f"({lhs.name}{op_str}{rhs.name})"
    fn_str = f"({lhs.flex_param.fn_body_str}{op_str}{rhs.flex_param.fn_body_str})"
    fp = FlexParam(fp_name, arg_str, fn_str)
    return DummyOp(template_type_str, fp)


@dummy_op.register(int)
def _(op1: int, op2: DummyOp, op_str: str, reflected=False):
    template_type_str = op2.template_types
    arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]
    if reflected:
        lhs = op1
        rhs = op2.flex_param.fn_body_str
    else:
        lhs = op2.flex_param.fn_body_str
        rhs = op1
    fp_name = f"({lhs}{op_str}{rhs})"
    fn_str = f"({lhs}{op_str}{rhs})"
    fp = FlexParam(fp_name, arg_str, fn_str)
    return DummyOp(template_type_str, fp)

# @dummy_op.register(int)
# def _(op1: int, op2: DummyOp, op_str: str, reflected=False):
#     if reflected:
#         lhs = op1
#         rhs = op2.flex_param.fn_body_str
#     else:
#         lhs = op2.flex_param.fn_body_str
#         rhs = op1
#     new_code = f"({lhs}{op_str}{rhs})"
#     op2.flex_param.update_fn_code(new_code)
#     return op2

