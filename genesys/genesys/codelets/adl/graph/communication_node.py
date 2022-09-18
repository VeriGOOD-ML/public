from codelets.adl.graph.architecture_node import ArchitectureNode
from typing import Dict

class CommunicationNode(ArchitectureNode):

    def __init__(self, name, comm_type=None, latency=None, bw=None, index=None):
        super(CommunicationNode, self).__init__(name, index=index)
        # Configuration Attributes
        self._comm_type = comm_type
        self._latency = latency
        self._bandwidth = bw
        # Visualization Attributes
        self._node_color = "#BFFFBF"

    @property
    def attribute_names(self):
        return ["comm_type", "latency", "bandwidth"]

    # Configuration Attributes
    @property
    def comm_type(self):
        return self._comm_type

    @comm_type.setter
    def comm_type(self, comm_type):
        self._comm_type = comm_type

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth

    # Visualization Attributes
    @property
    def viz_color(self):
        return self._node_color

    @property
    def node_color(self):
        return self._node_color

    # Derived / Other Attributes
    @property
    def node_type(self):
        return 'communication'

    # Class methods
    def get_viz_attr(self):
        return f"CommType: {self.comm_type}\\n" \
               f"Latency: {self.latency}\\n" \
               f"BW: {self.bandwidth}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['communication_type'] = self.comm_type
        blob['attributes']['latency'] = self.latency
        blob['attributes']['bandwidth'] = self.bandwidth
        blob = self.finalize_json(blob)
        return blob


