class NamespaceItem(object):

    def __init__(self, src_id: int, data_id: int, ready_cycle: int, written_cycle=-1, value=-1):
        self.src_id = src_id
        self.data_id = data_id
        self.valid = True
        self._state_update = False
        self._sink_src = -1
        self.uses = 1
        self.value = value
        self.ready_cycle = ready_cycle
        self.written_cycle = written_cycle

    def __str__(self):
        return f"Source ID: {self.src_id}\n" \
            f"Data ID: {self.data_id}"
    @property
    def is_state_updated(self) -> bool:
        return self._state_update

    def set_ready_cycle(self, cycle):
        self.ready_cycle = cycle

    def set_written_cycle(self, cycle, value=None):
        assert self.written_cycle <= 0
        self.written_cycle = cycle
        if value:
            self.value = value

    def add_use(self):
        self.uses += 1

    def updated_state(self, sink_src: int):
        self._sink_src = sink_src
        self._state_update = True

    def is_valid(self) -> bool:
        return self.valid

    def invalidate(self):
        if not self.valid:
            raise RuntimeError(f"Namespace item {self.data_id} is already invalid.")
        self.valid = False

    def update_data(self, src_id: int, data_id: int, value=None):
        if self.valid:
            raise RuntimeError(f"Namespace item {self.data_id} is already valid.")
        self.src_id = src_id
        self.data_id = data_id
        self.uses += 1
        self.valid = True
        self.value = value


