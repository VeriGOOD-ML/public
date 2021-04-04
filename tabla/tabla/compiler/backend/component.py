from . import State
from collections import defaultdict
from typing import List, Any, Dict


class Component(object):
    """A base class for all objects within an architecture.

    Attributes
    ----------
    resource_id : int
            This represents the globally unique id given to each component upon instantiation.
    category_id : int
            This represents the unique id within components of a specific `component_type`.
    component_type : str
            The string name corresponding to the component type, e.g., `bus`.
    component_subtype : str
            The string name corresponding to a subtype, e.g., `PEGB` for `bus`.

    """

    resource_counter = 0
    category_resource_counter = defaultdict(int)
    _component_dict = {}
    _category_component_dict = defaultdict(dict)

    def __init__(self, component_type: str, component_subtype: str):
        """

        Parameters
        ----------
        component_type : str
                The string name corresponding to the component type, e.g., `bus`.
        component_subtype : str
                The string name corresponding to a subtype, e.g., `PEGB` for `bus`.
        """

        if not isinstance(component_type, str):
            raise TypeError(f"'component_type' must be a string: {component_type}")

        if not isinstance(component_subtype, str):
            raise TypeError(f"'component_subtype' must be a string: {component_subtype}")

        self.resource_id = Component.resource_counter
        _ = self.add_component(self.resource_id)
        self.category_id = Component.category_resource_counter[component_type]
        _ = self.add_component_category(component_type, self.category_id)
        Component.category_resource_counter[component_type] += 1
        Component.resource_counter += 1

        self.component_type = component_type
        self.component_subtype = component_subtype
        self._states = {}
        self.active_states = []
        self.idle_states = []


    @property
    def states(self):
        return self._states

    def cycle_state_exists(self, cycle: int) -> bool:
        """Check if a state has been created for the component in a specific cycle.

        Checks if the input cycle is invalid (less than 0) or greater than the current maximum cycle.

        Parameters
        ----------
        cycle : int
            The cycle to check if the state exists.

        Returns
        -------
        bool
            True if the cycle state exists, otherwise false.

        """

        if self.max_cycle < cycle or cycle < 0:
            return False
        else:
            return True

    def add_idle_state(self, cycle: int) -> State:
        """Creates an idle state in the specified cycle.

        Parameters
        ----------
        cycle : int
            The cycle to create an idle state for.
        Returns
        -------
        State
            The idle state for the cycle.

        """
        return self.add_cycle_state(cycle, self.idle_state, {})

    def get_cycle_state(self, cycle: int) -> State:
        """Gets a copy of the component state in the specified cycle.

        This will raise an error if the specified cycle state does not exist.

        Parameters
        ----------
        cycle : int
            The cycle to retrieve the component state for.

        Returns
        -------
        State
            The state for the given cycle.

        """
        if not self.cycle_state_exists(cycle):
            raise RuntimeError(f"Cycle state {cycle} does not exist for"
                               f" {self.component_type} - {self.resource_id}")
        return self.states[cycle]

    def add_cycle_state(self, cycle: int, state_name: str, metadata: dict) -> State:
        """Adds a cycle state for the component.

        This will generate intermediate idle cycle states until the specified cycle,
        then generate a new cycle state using the `state_name` and `metadata`. If
        the cycle state already exists, this will raise an error.

        Parameters
        ----------
        cycle : int
            The cycle to create a new state in.
        state_name : str
            The name of the newly added state.
        metadata : dict
            A mapping of string keys to metadata values specific to the particular component type.

        Returns
        -------
        State
            The newly created state.

        """

        if cycle < 0:
            raise ValueError(f"Cannot add state for negative cycle in "
                             f"{self.component_type} - {self.component_subtype}")

        if state_name not in self.possible_states:
            raise ValueError(f"{state_name} is not a possible state for "
                             f"{self.component_type}.")


        if cycle in self.states.keys():
            raise ValueError(f"Cannot add cycle {cycle} "
                             f"because cycle state {cycle} already exists")

        new_state = State(cycle, state_name, metadata)

        self.states[cycle] = new_state
        return new_state

    def update_cycle_state(self, cycle: int, state_name: str, metadata: dict) -> State:
        """Updates an existing cycle state for the component.


        Parameters
        ----------
        cycle : int
            The cycle corresponding to the state that needs to be updated.
        state_name : str
            The new state string to update to.
        metadata : dict
            The new metadata to change state to.

        Returns
        -------
        State
            The newly updated state.

        """

        if state_name not in self.possible_states:
            raise ValueError(f"Invalid state name '{state_name}'")

        self.states[cycle].update_state(state_name, metadata)
        return self.get_cycle_state(cycle)

    def update_cycle_state_name(self, cycle: int, state_name: str) -> State:
        """Updates the state name for a given cycle.

        Parameters
        ----------
        cycle : int
            The cycle in which to update the state name.
        state_name : str
            The string name to update the state to.

        Returns
        -------
        State
            The newly updated state

        """
        if state_name not in self.possible_states:
            raise ValueError(f"Invalid state name '{state_name}'")
        self.states[cycle].update_state_name(state_name)
        return self.get_cycle_state(cycle)

    def update_cycle_state_metadata_item(self, cycle: int, key: str, value: Any) -> State:
        """Updates an individual metadata item for a given key.

        Parameters
        ----------
        cycle : int
            The cycle in which the state metadata will be updated.
        key : str
            The name of the metadata item to be updated.
        value : Any
            The value used for updating the metadata.

        Returns
        -------
        State
            The newly updated state.
        """
        self.states[cycle].update_metadata_item(key, value)
        return self.get_cycle_state(cycle)

    def update_cycle_state_metadata(self, cycle: int, metadata: Dict[str, Any]) -> State:
        """Updates all metadata for a given cycle state.

        Parameters
        ----------
        cycle : int
            The cycle in which the state metadata will be updated.
        metadata : Dict[str, Any]
            The mapping of metadata keys to values.
        Returns
        -------
        State
            The newly updated state.
        """
        self.states[cycle].update_metadata(metadata)
        return self.get_cycle_state(cycle)

    def is_idle(self, cycle: int) -> bool:
        """Checks if the component is idle in a given cycle.

        Parameters
        ----------
        cycle : int
            The cycle corresponding to the state to be checked.

        Returns
        -------
        bool
            True if the component is idle in the given state, else false.

        """
        if cycle in self.states:
            return False
        else:
            return True

    def get_state_name(self, cycle: int) -> str:
        """Retrieves the `state_name` for the `cycle` state.

        Parameters
        ----------
        cycle : int
            Cycle to retrieve the `state_name` in.

        Returns
        -------
        str
            String name of the state in the specific cycle.

        """
        state = self.get_cycle_state(cycle)
        return state.state_name

    @property
    def max_cycle(self) -> int:
        """Provides the current maximum cycle number.

        This property provides the maximum cycle for which states are specified
        for this particular component. Cycles start with 1, which means the
        length of the state list specifies the maximum cycle.

        Returns
        -------
        int
            The maximum cycle for this component.

        """
        return 0 if len(self.states.keys()) == 0 else max(list(self.states.keys()))

    def shift_states(self, cycle):
        if cycle not in self.states:
            return
        else:
            start_max = max(list(self.states.keys()))
            prev_state = self.states.pop(cycle)
            while cycle <= start_max and (cycle+1) in self.states.keys():
                future_state = self.states.pop(cycle + 1)
                self.states[cycle + 1] = prev_state
                prev_state = future_state
                cycle += 1
            self.states[cycle + 1] = prev_state


    @property
    def num_states(self):
        return self.max_cycle

    @property
    def category_component_dict(self) -> Dict[str, Dict[int, 'Component']]:
        return Component._category_component_dict

    @property
    def component_dict(self) -> Dict[int, 'Component']:
        return Component._component_dict

    def add_component_category(self, category_str: str, category_id: int) -> 'Component':
        """
        Adds a categorized component using it's category name and category id.

        Parameters
        ----------
        category_str : str
                The string name of the component category.
        category_id : int
                The integer id of the component relative to the category.

        Returns
        -------
        Component
            The component corresponding to the category and name

        """
        Component._category_component_dict[category_str][category_id] = self
        return self

    def add_component(self, resource_id: int) -> 'Component':
        """
        Adds a component using it's resource id.

        Parameters
        ----------
        resource_id : int
                The integer id of the component relative to the category.

        Returns
        -------
        Component
            The component corresponding to the component id

        """
        Component._component_dict[resource_id] = self
        return self

    def get_component(self, resource_id: int) -> 'Component':
        """
        Adds a component using it's category id.

        Parameters
        ----------
        resource_id : int
                The integer id of the component relative to the category.

        Returns
        -------
        Component
            The component corresponding to the category and name

        """
        if resource_id not in Component._component_dict:
            raise KeyError
        return self.component_dict[resource_id]

    def get_component_category(self, category_str: str, category_id: int) -> 'Component':
        """
        Adds a categorized component using it's category name and category id.

        Parameters
        ----------
        category_str : str
                The string name of the component category.
        category_id : int
                The integer id of the component relative to the category.

        Returns
        -------
        Component
            The component corresponding to the category and name

        """
        if category_str not in self.category_component_dict:
            raise KeyError(f"{category_str} not found in component categories.")
        if category_id not in self.category_component_dict[category_str]:
            raise KeyError(f"{category_id} not found in {category_str} component ids.")
        return self.category_component_dict[category_str][category_id]

    @property
    def possible_states(self) -> List[str]:
        raise NotImplementedError

    @property
    def idle_state(self) -> str:
        raise NotImplementedError

    def create_initial_state(self) -> State:
        raise NotImplementedError

    def is_valid_instruction(self, cycle: int, instruction) -> bool:
        raise NotImplementedError

    def add_instruction(self, cycle: int, instruction) -> bool:
        raise NotImplementedError

    @staticmethod
    def reset_ids():
        Component.resource_counter = 0
        Component.category_resource_counter = defaultdict(int)
