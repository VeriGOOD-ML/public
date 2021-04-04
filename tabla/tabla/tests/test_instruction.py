from backend.pu import PU
from backend.component import Component
from backend.instruction import DataLocation, Source, Dest, Instruction

def test_data_location_init():
    namespace = 'NI'
    component_id = 0

    loc = DataLocation(component_id, namespace)
    assert loc.location == namespace
    assert loc.data_id == component_id

def test_is_local_success():
    namespace = 'NI'
    component_id = 0

    loc = DataLocation(component_id, namespace)
    assert loc.is_local() == True

def test_source_init():
    ns = 'NI'
    comp_id = 0

    src = Source(0, location=ns, data_id=comp_id)
    assert src.location == ns
    assert src.data_id == comp_id

def test_dest_init():
    ns = 'NI'
    comp_id = 0

    dst = Dest(0, location=ns, data_id=comp_id)
    assert dst.location == ns
    assert dst.data_id == comp_id

# TODO: Complete the following test cases once we finish implementing instruction.py
def test_get_source_dict():
    pass

def test_get_dest_dict():
    pass

def test_get_dests():
    pass

def test_get_sources():
    pass

def test_set_component_id():
    pass

def test_has_ni_dest():
    pass

def test_has_nw_dest():
    pass

def test_has_nd_dest():
    pass

def test_has_ng_dest():
    pass

def test_has_penb_dest():
    pass

def test_has_pegb_dest():
    pass

def test_has_punb_dest():
    pass

def test_has_pugb_test():
    pass
