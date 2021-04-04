from codelets.adl.graph.architecture_node import ArchitectureNode

from codelets.adl.flex_template.instruction import Instruction
from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from codelets.codelet_impl.codelet import Codelet


# NOTE originally, there were implementations for occupancy. This has been 
#      removed because this can be managed by the compiler during the scheduling
#      phase as a scoreboard or something instead of instance of architecture graph.

class ComputeNode(ArchitectureNode):

    def __init__(self, name, dimensions=None, codelets=None, primitives=None, index=None):
        super(ComputeNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)
        self._primitives = {}
        self._codelets = {}
        self.dimensions = dimensions or [1]
        if primitives:
            for p in primitives:
                if isinstance(p, dict):
                    # prim = self.parse_capability_json(p)
                    raise TypeError
                else:
                    prim = p
                self.add_primitive(prim)

        # TODO: Check codelet primitives for support
        if codelets:
            for p in codelets:
                if isinstance(p, dict):
                    # cdlt = self.parse_codelet_json(p)
                    raise TypeError
                else:
                    cdlt = p
                self.add_codelet(cdlt)
    @property
    def attribute_names(self):
        return ["dimensions", "codelets", "primitives"]

    @property
    def node_type(self):
        return 'compute'

    @property
    def viz_color(self):
        return "#BFBFFF"

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def primitives(self) -> Dict[str, Instruction]:
        return self._primitives

    @property
    def codelets(self) -> Dict[str, 'Codelet']:
        return self._codelets

    @dimensions.setter
    def dimensions(self, dimensions):
        assert isinstance(dimensions, list)
        self._dimensions = dimensions

    # def parse_capability_json(self, cap_dict):
    #     name = cap_dict.pop("field_name")
    #     cap = Codelet(name, **cap_dict)
    #     return cap
    #
    # def parse_codelet_json(self, cdlt_dict):
    #     name = cdlt_dict.pop("field_name")
    #     cap = Codelet(name, **cdlt_dict)
    #     return cap

    def get_viz_attr(self):
        caps = list(self.get_primitives())
        if len(caps) > 5:
            return f"Capabilities: {caps[:5]}"
        else:
            return f"Capabilities: {caps}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['dimensions'] = self.dimensions
        blob['attributes']['primitives'] = [c.to_json() for k, c in self.primitives.items()]
        blob['attributes']['codelets'] = [c.to_json() for k, c in self.codelets.items()]
        blob = self.finalize_json(blob)
        return blob