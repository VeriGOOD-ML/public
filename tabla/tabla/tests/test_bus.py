import pytest
from backend.bus import Bus
from backend.component import Component

def test_bus_creation():
    new_bus = Bus('PENB')

def test_bus_string():
    Component.reset_ids()
    new_bus = Bus('PENB')
    assert str(new_bus) == "Type: bus\n\t" \
                           "Subtype: PENB\n\t" \
                           "ID: 0"

    second_bus = Bus('PUGB')
    assert str(new_bus) == "Type: bus\n\t" \
                           "Subtype: PENB\n\t" \
                           "ID: 0"
    assert str(second_bus) == "Type: bus\n\t" \
                           "Subtype: PUGB\n\t" \
                           "ID: 1"

def test_bus_state_creation():
    cycle = 2
    new_bus = Bus("PENB")
    metadata = {'src' : 2,
                'dst' : 3,
                'data_id' : 1}
    new_state = new_bus.create_busy_state(cycle, metadata)

def test_get_metadata():
    cycle = 2
    new_bus = Bus("PENB")
    metadata = {'src': 2,
                'dst': 3,
                'data_id': 1}
    new_state = new_bus.create_busy_state(cycle, metadata)

    test_src = new_state.get_metadata_by_key('src')
    test_dsc = new_state.get_metadata_by_key('dst')
    test_data_id = new_state.get_metadata_by_key('data_id')

def test_get_state_src():
    cycle = 0
    bus = Bus('PENB')


def test_get_state_dst():
    pass

def test_get_state_data_id():
    pass

def test_is_data_present():
    pass

def test_is_same_bus():
    pass

def test_add_instruction():
    pass

def test_is_valid_instruction():
    pass
