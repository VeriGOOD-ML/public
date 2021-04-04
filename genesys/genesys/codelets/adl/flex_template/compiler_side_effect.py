from typing import Union
from dataclasses import dataclass
from codelets.adl.flex_param import FlexParam

@dataclass
class SideEffect:
    name: str
    scope: str
    value: Union[str, int]
    side_effect_fp: FlexParam

    def run_side_effect(self):
        pass

    def copy(self):
        return SideEffect(self.name, self.scope, self.value, self.side_effect_fp.copy())