from typing import List, Dict, Union, Tuple
from dataclasses import dataclass, field

from codelets import Datatype
from codelets.adl.operation import Operand
from .dummy_op import DummyOp, DummyParam
OPERATION_TEMPLATE_CLASSES = ['OperationTemplate', 'LoopTemplate', 'ComputeTemplate', 'ConfigureTemplate',
                              'TransferTemplate']


@dataclass
class OperandTemplate:
    name: str
    supported_dtypes: Union[List[Datatype]]
    shape_list: List[Union[int, DummyOp, DummyParam]]
    default_dtype: Union[None, Datatype] = field(default=None)
    write_destination: Union[str, DummyParam, DummyParam] = field(default=None)
    start_location: Union[str, DummyParam, DummyParam] = field(default=None)

    def __getitem__(self, item: List[DummyParam]):
        if isinstance(item, (list, tuple)) and len(item) == 0:
            return self
        if isinstance(item, list):
            assert not isinstance(item[0], (tuple, list))
            item = tuple(item)
        elif not isinstance(item, tuple):
            item = tuple([item])
        offset = OffsetTemplate(item)
        return IndexOperandTemplate(self, offset)

    def reorder_shapes(self, permutation: List[int]):
        assert len(permutation) == len(self.shape_list)
        self.shape_list = [self.shape_list[i] for i in permutation]

    @property
    def shape_list_names(self):
        slist = []
        for s in self.shape_list:
            if isinstance(s, (DummyParam, DummyOp)):
                slist.append(s.name)
            else:
                slist.append(s)
        return slist

    @property
    def operand_shape_list_names(self):
        return self.shape_list_names

    @property
    def operand_shape_list(self):
        return self.shape_list

    @property
    def shape_symbols(self):
        return self.shape_list_names

    def set_write_destination(self, write_destination: Union[str, DummyParam, DummyParam]):
        self.write_destination = write_destination

    def instantiate(self, instance_args):
        # TODO: Need to fix initialization so that evaluated dummy args are placed here
        evaluated_shape_list = evaluate_args(self.shape_list, instance_args, (DummyOp,))
        shape_list = []
        for s in evaluated_shape_list:
            if isinstance(s, int):
                shape_list.append(s)
            else:
                assert isinstance(s, (DummyParam, DummyOp))
                shape_list.append(s.name)

        if self.default_dtype is None:
            dtype = self.supported_dtypes[0]
        else:
            dtype = self.default_dtype
        operand = Operand(self.name, self.supported_dtypes, shape_list, dtype=dtype,
                          write_destination=self.write_destination)

        for s in evaluated_shape_list:
            assert isinstance(s, (DummyParam, DummyOp)), f"Non dummy param shape: {type(s)}"
            operand.update_shape_symbols(s.name, s.value)

        if self.start_location is not None:
            operand.data_path = [evaluate_args(self.start_location, instance_args, tuple([]))]

        return operand

@dataclass
class OffsetTemplate:
    # OffsetTemplates should be evaluated as a list of either LoopTemplate Operations or integers
    offsets: Tuple


@dataclass
class IndexOperandTemplate:
    operand: OperandTemplate
    offset: OffsetTemplate

    def evaluate(self, instance_args):
        operand = instance_args['CodeletTemplate'].get_operand(self.operand.name)
        # TODO: Need to fix initialization so that evaluated dummy args are placed here
        evaluated_offset = evaluate_args(self.offset.offsets, instance_args, tuple([]))
        assert isinstance(evaluated_offset, tuple)
        operand_idx = operand[evaluated_offset]
        return operand_idx

    def reorder_offsets(self, permutation: List[int]):
        assert len(permutation) == len(self.offset.offsets)
        self.offset.offsets = tuple([self.offset.offsets[i] for i in permutation])

    @property
    def offset_names(self):
        names = []
        for o in self.offset.offsets:
            if isinstance(o, (DummyParam, DummyOp)):
                names.append(o.name)
            else:
                names.append(o.op_str)
        return names

    @property
    def name(self):
        return self.operand.name

    @property
    def operand_shape_list_names(self):
        return self.operand.shape_list_names

    @property
    def operand_shape_list(self):
        return self.operand.shape_list

def evaluate_args(args, instance_args, preserve_types):


    if isinstance(args, list):
        eval_arg = []
        for a in args:
            eval_arg.append(evaluate_args(a, instance_args, preserve_types))
    elif isinstance(args, tuple):
        eval_arg = []
        for a in args:
            eval_arg.append(evaluate_args(a, instance_args, preserve_types))
        eval_arg = tuple(eval_arg)
    elif isinstance(args, (DummyParam, DummyOp)):
        eval_arg = args.evaluate(instance_args)
    elif args.__class__.__name__ in OPERATION_TEMPLATE_CLASSES:
        cdlt = instance_args['CodeletTemplate']
        eval_arg = cdlt.op_map[args.op_str]
    elif isinstance(args, OperandTemplate):
        cdlt = instance_args['CodeletTemplate']
        eval_arg = cdlt.get_operand(args.name)
    else:
        eval_arg = args

    if isinstance(args, preserve_types):
        return args

    return eval_arg