import pytest
from backend.namespace_item import NamespaceItem

def test_init():
    ni = NamespaceItem(0, 0, 0)
    assert ni.src_id == 0
    assert ni.data_id == 0
    assert ni.valid == True
    assert ni.uses == 1

def test_is_valid():
    ni = NamespaceItem(0, 0, 0)
    assert ni.valid == True

def test_invalidate():
    ni = NamespaceItem(0, 0, 0)
    ni.invalidate()
    assert ni.valid == False

def test_invalidate_exception():
    ni = NamespaceItem(0, 0, 0)
    ni.valid = False
    with pytest.raises(RuntimeError):
        ni.invalidate()

def test_update_data():
    ni = NamespaceItem(0, 0, 0)
    ni.valid = False
    ni.update_data(0, 1)
    assert ni.src_id == 0
    assert ni.data_id == 1
    assert ni.uses == 2
    assert ni.valid == True

def test_update_data_exception():
    ni = NamespaceItem(0, 0, 0)
    with pytest.raises(RuntimeError):
        ni.update_data(0, 1)
