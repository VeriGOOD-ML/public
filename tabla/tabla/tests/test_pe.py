import pytest
import backend.pe
from backend.pe import PE
from backend.pu import PU
from backend.instruction import Source, Dest, Instruction

def test_pe_init_valid():
    pe = PE(1, 256, 256)
    assert pe.component_type == 'pe'
    assert pe.component_subtype == '1'

def test_get_namespace():
    pe = PE(1, 256, 256)
    ns = pe.get_namespace('NI')
    assert ns.component_type == 'namespace'
    assert ns.component_subtype == 'NI'
    assert ns._capacity == 256

def test_is_local_namespace():
    pe = PE(1, 256, 256)
    assert pe.is_local_namespace('NI') == True

def test_are_sources_ready():
    pe = PE(1, 256, 256)
    assert pe.are_sources_ready(0, []) == True

def test_are_dests_full():
    pe = PE(1, 256, 256)
    assert pe.are_dests_full(0, []) == True

def test_add_instruction_cycle0():
    pe = PE(1, 256, 256)
    src1_ns = pe.get_namespace('ND').component_id
    src2_ns = pe.get_namespace('NW').component_id
    dst1_ns = pe.get_namespace('NI').component_id
    inst = Instruction(0, '+')
    inst.add_source(0, 'ND', src1_ns, 0)
    inst.add_source(1, 'NW', src2_ns, 0)
    inst.add_dest(2, 'NI', dst1_ns, 0)
    assert pe.add_instruction(0, inst) == True

def test_add_instruction_cycle1():
    pe = PE(1, 256, 256)
    src1_ns = pe.get_namespace('ND').component_id
    src2_ns = pe.get_namespace('NW').component_id
    dst1_ns = pe.get_namespace('NI').component_id
    inst = Instruction(0, '+')
    inst.add_source(0, 'ND', src1_ns, 0)
    inst.add_source(1, 'NW', src2_ns, 0)
    inst.add_dest(2, 'NI', dst1_ns, 0)

    assert pe.add_instruction(1, inst) == True
