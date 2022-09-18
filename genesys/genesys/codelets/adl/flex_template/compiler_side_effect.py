from typing import Union, Tuple
from dataclasses import dataclass
from codelets.adl.flex_param import FlexParam

@dataclass
class SideEffect:
    name: str
    scope: str
    value: Union[str, int]
    side_effect_fp: FlexParam

    def run_side_effect(self, args: Tuple, add_self=False):
        assert isinstance(args, tuple)
        if add_self:
            args = args + (self.value,)

        self.value = self.side_effect_fp.evaluate_fn(*args, force_evaluate=True)
        return self.value

    def copy(self):
        return SideEffect(self.name, self.scope, self.value, self.side_effect_fp.copy())