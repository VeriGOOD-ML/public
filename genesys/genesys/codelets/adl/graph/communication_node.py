from codelets.adl.graph.architecture_node import ArchitectureNode
from typing import Dict

class CommunicationNode(ArchitectureNode):

    def __init__(self, name, comm_type=None, latency=None, bw=None, index=None):
        super(CommunicationNode, self).__init__(name, index=index)
        self._node_color = self.viz_color
        self._comm_type = comm_type
        self._latency = latency
        self._bandwidth = bw

    @property
    def node_type(self):
        return 'communication'

    @property
    def viz_color(self):
        return "#BFFFBF"

    @property
    def node_color(self):
        return self._node_color

    @property
    def comm_type(self):
        return self._comm_type

    @property
    def latency(self):
        return self._latency

    @property
    def bandwidth(self):
        return self._bandwidth

    def set_comm_type(self, comm_type):
        self._comm_type = comm_type

    def get_comm_type(self):
        return self._comm_type

    def set_latency(self, latency):
        self._latency = latency

    def get_latency(self):
        return self._latency

    def set_bw(self, bw):
        self._bw = bw

    def get_bw(self):
        return self._bw

    def get_viz_attr(self):
        return f"CommType: {self.get_comm_type()}\\n" \
               f"Latency: {self.get_latency()}\\n" \
               f"BW: {self.get_bw()}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['communication_type'] = self.comm_type
        blob['attributes']['latency'] = self.latency
        blob['attributes']['bandwidth'] = self.bandwidth
        blob = self.finalize_json(blob)
        return blob


