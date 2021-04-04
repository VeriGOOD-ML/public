from backend.pu import PU
from backend.component import Component
from backend.instruction import Instruction

def test_pu_creation():
    Component.reset_ids()
    num_pes = 6
    namespace_size = 256
    namespace_interim_size = 256
    test_pu = PU(num_pes,-1,namespace_size, namespace_interim_size)
    assert str(test_pu) == "Type: pu\n\t" \
                           "PE IDs: [0, 1, 2, 3, 4, 5]\n\t" \
                           "ID: 0"
    second_pu = PU(num_pes,-1,namespace_size, namespace_interim_size)
    assert str(second_pu) == "Type: pu\n\t" \
                           "PE IDs: [6, 7, 8, 9, 10, 11]\n\t" \
                           "ID: 44"

def test_get_bus():
    Component.reset_ids()
    pu = PU(8,-1, 256, 256)
    bus = pu.get_bus('PEGB')
    assert bus.component_type == 'bus'
    assert bus.component_subtype == 'PEGB'

def test_are_sources_ready():
    Component.reset_ids()
    pu = PU(8,-1,256, 256)
    assert pu.are_sources_ready(0, []) == True

def test_is_valid_instruction():
    Component.reset_ids()
    pu = PU(8,-1,256, 256)
    inst = Instruction(0, '+')
    assert pu.is_valid_instruction(0, inst) == True

def test_find_avail_pe():
    Component.reset_ids()
    cycle = 0
    dest1_id = 0
    dest2_id = 1

    dest1_data_id = 0
    dest2_data_id = 1
    pu = PU(8,-1,256, 256)
    pe = pu.get_pe(0)
    assert pe.category_id == 0
    pe.get_namespace('ND').insert_data(cycle, dest1_id, dest1_data_id)
    pe.get_namespace('NW').insert_data(cycle, dest2_id, dest2_data_id)
    nd_pe = pe.get_namespace('ND')
    nw_pe = pe.get_namespace('NW')

    dst1_ns = pe.get_namespace('NI').component_id
    inst = Instruction(0, '+')
    inst.add_source(dest1_id, 'ND', dest1_data_id, 0)
    inst.add_source(dest2_id, 'NW', dest2_data_id, 0)

    inst.add_dest(2, 'NI', dst1_ns, 0)
    assert pu.find_avail_pe(0, inst) == 0
