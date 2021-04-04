import pytest
from backend.namespace import Namespace
from backend.component import Component

def test_ns_init_valid():
    ns = Namespace('NI', 256)
    assert ns.component_type == 'namespace'
    assert ns.component_subtype == 'NI'
    assert ns._capacity == 256

def test_ns_init_invalid():
    with pytest.raises(TypeError):
        invalid_ns = Namespace('NG', '256')

def test_ns_string():
    Component.reset_ids()
    ns = Namespace('NG', 256)
    assert str(ns) == "Type: namespace\n\t" \
                      "Subtype: NG\n\t" \
                      "ID: 0"

def test_item_count_zero():
    ns = Namespace('NI', 256)
    assert ns.item_count() == 0

def test_is_full_false_success():
    ns = Namespace('NI', 256)
    assert ns.is_full() == False

def test_is_full_true_success():
    ns_size = 1
    src_id = 0
    data_id = 0

    ns = Namespace('NI', ns_size)
    ns.insert_data(0, src_id, data_id)
    assert ns.is_full() == True

def test_insert_data_success():
    ns_size = 256
    src_id = 0
    data_id = 0

    ns = Namespace('NI', ns_size)
    state = ns.insert_data(1, src_id, data_id)
    assert state.cycle == 1
    assert state.state_name == 'free'

    ns = Namespace('NW', ns_size)
    state = ns.insert_data(0, src_id, data_id)
    assert state.cycle == 0
    assert state.state_name == 'free'

def test_is_data_present_success():
    ns = Namespace('NI', 256)
    assert ns.is_data_present(0, 0) == False

def test_remove_data_success():
    ns_size = 256
    src_id = 0
    data_id = 0

    ns = Namespace('NI', ns_size)
    ns.insert_data(0, src_id, data_id)
    # @Sean: remove_data() gives an error (Cycle state 1 does not exist)
    ns.remove_data(0, data_id)
