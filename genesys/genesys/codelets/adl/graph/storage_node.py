from codelets.adl.graph.architecture_node import ArchitectureNode
from typing import Dict

class StorageNode(ArchitectureNode):
    ACCESS_TYPES = ["FIFO", "RAM"]
    BUFF_SCHEMES = {"single": 1,
                    "double": 2,
                    "quadruple": 4}
    def __init__(self, name,
                 access_type=None,
                 banks=-1,
                 buffering_scheme=None,
                 latency=0,
                 input_ports=1,
                 output_ports=1,
                 width=-1,
                 depth=-1,
                 indirection=False,
                 on_chip=True,
                 index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        self.set_attr("node_color", self.viz_color)
        self.access_type = access_type
        self.banks = banks
        # self.size = size
        self.width = width
        self.depth = depth
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.buffering_scheme = buffering_scheme or "single"
        self.indirection = indirection
        self.latency = latency
        self.on_chip = on_chip

    @property
    def attribute_names(self):
        return ["access_type", "size", "width", "input_ports", "output_ports", "buffering_scheme",
                "indirection", "latency", "on_chip"]

    @property
    def viz_color(self):
        return "#7FFFFF"

    @property
    def buffering_scheme(self):
        return self._buffering_scheme

    @buffering_scheme.setter
    def buffering_scheme(self, scheme):
        if isinstance(scheme, int):
            self._buffering_scheme = scheme
        else:
            assert isinstance(scheme, str) and scheme in StorageNode.BUFF_SCHEMES
            self._buffering_scheme = StorageNode.BUFF_SCHEMES[scheme]

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def banks(self):
        return self._banks

    @property
    def latency(self):
        return self._latency

    @property
    def on_chip(self):
        return self._on_chip

    @property
    def input_ports(self):
        return self._input_ports

    @property
    def output_ports(self):
        return self._output_ports

    @property
    def access_type(self):
        return self._access_type

    @property
    def size(self):
        return self.depth*self.width*self.banks

    @property
    def size_bytes(self):
        return self.size // 8

    @property
    def indirection(self):
        return self._indirection

    @banks.setter
    def banks(self, banks):
        self._banks = banks

    @width.setter
    def width(self, width):
        self._width = width

    @depth.setter
    def depth(self, depth):
        self._depth = depth

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @on_chip.setter
    def on_chip(self, on_chip):
        self._on_chip = on_chip

    @size.setter
    def size(self, size):
        self.set_size(size)

    @access_type.setter
    def access_type(self, access_type):
        self.set_access_type(access_type)


    @indirection.setter
    def indirection(self, indirection):
        self._indirection = indirection

    @input_ports.setter
    def input_ports(self, input_ports):
        self._input_ports = input_ports

    @output_ports.setter
    def output_ports(self, output_ports):
        self._output_ports = output_ports

    @property
    def node_type(self):
        return 'storage'

    def set_buffer_scheme(self, scheme):
        self._buffering_scheme = scheme

    def set_input_ports(self, input_buffers):
        self._input_ports = input_buffers

    def set_output_ports(self, output_buffers):
        self._output_ports = output_buffers

    def set_access_type(self, access_type):
        assert access_type in StorageNode.ACCESS_TYPES
        self._access_type = access_type

    def get_access_type(self):
        return self._access_type
    
    def set_size(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def get_viz_attr(self):
        return f"Access Type: {self.get_access_type()}\\n" \
               f"Size: {self.get_size()}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['access_type'] = self.access_type
        blob['attributes']['size'] = self.size
        blob['attributes']['input_ports'] = self.input_ports
        blob['attributes']['output_ports'] = self.output_ports
        blob['attributes']['buffering_scheme'] = self.width
        blob['attributes']['width'] = self.buffering_scheme
        blob['attributes']['latency'] = self.latency
        blob['attributes']['on_chip'] = self.on_chip
        blob = self.finalize_json(blob)
        return blob

