import copy
from typing import Dict, Any
import pickle
class State(object):

    def __init__(self, cycle: int, state_name: str, metadata: Dict[str, Any]):
        self.cycle = cycle
        self._metadata = metadata
        self._state_name = state_name

    def update_state(self, state_name: str, metadata: Dict[str, Any]):
        if len(metadata.keys()) > 0 and isinstance(metadata, dict):
            for key, value in metadata.items():
                self.add_metadata(key, value)
        self._state_name = state_name

    def update_state_name(self, state_name: str):
        self._state_name = state_name

    def get_cycle(self) -> int:
        return int(self.cycle)

    def add_metadata(self, key: str, value: Any):
        self._metadata[key] = value

    def update_metadata_item(self, key: str, value: Any):
        if key not in self._metadata:
            raise ValueError(f"Cannot update metadata with key '{key}' "
                             f"because it does not exists in metadata.")
        self._metadata[key] = value

    def update_metadata(self, new_metadata: Dict[str, Any]) -> Dict[str, Any]:

        for key, value in new_metadata.items():
            self.update_metadata_item(key, value)
        return self._metadata

    def get_metadata_by_key(self, key: str) -> Any:
        if key not in self._metadata:
            raise ValueError(f"Cannot get metadata with key '{key}' "
                             f"because it does not exists in metadata."
                             f"\nPossible keys: {self._metadata.keys()}")
        return self._metadata[key]

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @property
    def state_name(self):
        return self._state_name

    def __copy__(self) -> 'State':
        return pickle.loads(pickle.dump(self))

    def __str__(self):
        s = f"cycle: {self.cycle}, name: {self._state_name}, meta: {self._metadata}"
        return s