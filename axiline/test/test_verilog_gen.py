from axiline.verilog_template import VerilogTemplate
import os

def test_ip_gen():
    path="templates/ip.v"
    print(os.path.exists(path))
    parameters={
        "i":4,
    }
    ip=VerilogTemplate(path,parameters)
    ip.generate_ip_expr(4)
    print(ip.parameters)
    ip.generate_verilog()
    ip.write_to_file("output/ip.v")

def test_comb_gen():
    path="templates/comb.v"
    print(os.path.exists(path))
    comb=VerilogTemplate(path)
    comb.generate_comb("logisticReg")
    print(comb.parameters)
    comb.generate_verilog()
    comb.write_to_file("output/comb.v")

def test_sgd_gen():
    path = "templates/sgd.v"
    print(os.path.exists(path))
    parameters = {
        "i": 4,
    }
    sgd = VerilogTemplate(path,parameters)
    print(sgd.parameters)
    sgd.generate_verilog()
    sgd.write_to_file("output/sgd.v")

def test_template_gen():
    path = "templates/template.v"
    print(os.path.exists(path))
    parameters = {
        "logNumCycle": '1',
        "NumCycle": '1',
        "size":'4',
        "i":4
    }
    sgd = VerilogTemplate(path, parameters)
    print(sgd.parameters)
    sgd.generate_verilog()
    sgd.write_to_file("output/template.v")