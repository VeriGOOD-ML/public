from backend.tabla_template import TablaTemplate
from backend.component import Component
import json


def test_pu_creation():
    """
    Test all the PUs are created correctly as specified in the config.
    # TODO Currently we're only checking the string representation of PU object matches, but we can do more thorough testing in the future.

    Returns
    -------
    None

    """
    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    pu_map = new_arch.pu_map
    assert len(pu_map) == 8

    pu0 = pu_map["PU0"]
    assert str(pu0) == "Type: pu\n\t" \
                                "PE IDs: [0, 1, 2, 3, 4, 5, 6, 7]\n\t" \
                                "ID: 2"
    pu1 = pu_map["PU1"]
    assert str(pu1) == "Type: pu\n\t" \
                                "PE IDs: [8, 9, 10, 11, 12, 13, 14, 15]\n\t" \
                                "ID: 60"
    pu2 = pu_map["PU2"]
    assert str(pu2) == "Type: pu\n\t" \
                                "PE IDs: [16, 17, 18, 19, 20, 21, 22, 23]\n\t" \
                                "ID: 118"
    pu3 = pu_map["PU3"]
    assert str(pu3) == "Type: pu\n\t" \
                                "PE IDs: [24, 25, 26, 27, 28, 29, 30, 31]\n\t" \
                                "ID: 176"
    pu4 = pu_map["PU4"]
    assert str(pu4) == "Type: pu\n\t" \
                                "PE IDs: [32, 33, 34, 35, 36, 37, 38, 39]\n\t" \
                                "ID: 234"
    pu5 = pu_map["PU5"]
    assert str(pu5) == "Type: pu\n\t" \
                                "PE IDs: [40, 41, 42, 43, 44, 45, 46, 47]\n\t" \
                                "ID: 292"
    pu6 = pu_map["PU6"]
    assert str(pu6) == "Type: pu\n\t" \
                                "PE IDs: [48, 49, 50, 51, 52, 53, 54, 55]\n\t" \
                                "ID: 350"
    pu7 = pu_map["PU7"]
    assert str(pu7) == "Type: pu\n\t" \
                                "PE IDs: [56, 57, 58, 59, 60, 61, 62, 63]\n\t" \
                                "ID: 408"


def test_pe_creation():
    """
    # TODO: Test all PEs are created correctly as specified in the config.

    Returns
    -------
    None

    """
    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    pu_map = new_arch.pu_map


def test_pu_global_bus_creation():
    """
    # TODO: Test the PU Global Bus is created correctly as specified in the config.

    Returns
    -------
    None

    """
    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    bus_map = new_arch.bus_map
    assert len(bus_map) == 9


def test_pu_neighbor_bus_creation():
    """
    # TODO: Test all PU Neighbor Buses are created correctly as specified in the config.
    # TODO: Add API to Bus that returns source PE/PU and destination PE/PU and test it here.

    Returns
    -------
    None

    """
    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    bus_map = new_arch.bus_map
    assert len(bus_map) == 9


def test_component_map():
    """
    # TODO: Test component map returns components correctly.

    Returns
    -------
    None

    """
    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    comp_map = new_arch.component_map
    pu_map = new_arch.pu_map
    pu4_component_id = 234  # Number 233 was obtained by printing out the IDs beforehand.
    pu4 = comp_map[pu4_component_id]
    assert pu4 == pu_map["PU4"]

def test_pes_per_pu():

    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    assert new_arch.pes_per_pu == 8



def test_neighbor_vals():

    Component.reset_ids()
    with open("config.json") as config_file:
        data = json.load(config_file)

    new_arch = TablaTemplate(data)
    head_pe = new_arch.cat_component_map['pe'][8]
    assert new_arch.get_pe_neighbor(head_pe.component_id) == 67

    non_head_pe = new_arch.cat_component_map['pe'][15]
    assert new_arch.get_pe_neighbor(non_head_pe.component_id) == 61

    head_pu = new_arch.cat_component_map['pu'][0]
    assert new_arch.get_pu_neighbor(head_pu.component_id) == 60

    non_head_pu = new_arch.cat_component_map['pu'][7]
    assert new_arch.get_pu_neighbor(non_head_pu.component_id) == 2


