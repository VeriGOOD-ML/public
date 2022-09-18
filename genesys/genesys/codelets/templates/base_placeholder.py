from typing import Dict
from dataclasses import dataclass, field
from .dummy_op import DummyOp
from codelets.adl.flex_param import FlexParam
import inspect
import polymath as pm
from . import TEMPLATE_CLASS_ARG_MAP, CLASS_ATTR


@dataclass
class PlaceholderStructure:
    name: str
    template_type: str = field(default=None)

    def __post_init__(self):
        self.template_type = self.__class__.__name__

    def __getattr__(self, name):
        assert name in CLASS_ATTR[self.template_type] or self.template_type == "NodePlaceholder"
        args = TEMPLATE_CLASS_ARG_MAP[self.template_type]
        fp = FlexParam(f"{self.template_type}_{self.name}_dummy", args, f"{args[0]}.{name}")
        return DummyOp([self.template_type], fp)

    def __getstate__(self):
        return {'name': self.name, 'template_type': self.template_type}

    def __setstate__(self, state):
        self.name = state['name']
        self.template_type = state['template_type']