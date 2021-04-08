from polymath.polymath.srdfg.passes import register_pass, Pass
# from polymath.polymath.srdfg.util import _flatten_iterable, is_iterable, extend_indices, squeeze_indices, get_indices
# from polymath.polymath import func_op, DEFAULT_SHAPES, UNSET_SHAPE, SCALAR_IDX
import polymath.polymath as pm
# from numbers import Integral
# from collections import defaultdict
# from itertools import product
# import numpy as np

@register_pass
class VerilogGenerateFixedBitwidth(Pass):

    def __init__(self, param):
        self.param = param
        self.counter=0
        self.weight=[]
        self.activation=[]
        self.bias=[]
        self.rate=[]
        self.output=[]
        self.param_list=param["param_list"]
        self.verilog_code=""
        with open("./templates/non_embedded.v", 'r') as file:
            self.non_embedded_temp = file.read()
        with open("./templates/embedded.v", 'r') as file:
            self.embedded_temp = file.read()
        super(VerilogGenerateFixedBitwidth, self).__init__()

    def create_verilog_nonembedded(self):
        verilog=""
        temp=self.non_embedded_temp.split("\n")
        for line in temp:
            for param in self.variable_list:
                symbol = "$%" + param + "%$"
                if symbol in line:
                    new_line=""
                    if param=="Activation":
                        symbol_list=self.activation
                    elif param=="Weight":
                        symbol_list = self.weight
                    elif param=="Bias":
                        symbol_list = self.bias
                    elif param=="Rate":
                        symbol_list = self.rate
                    elif param=="Output":
                        symbol_list = self.output
                    for symbol_value in symbol_list:
                        new_line += line.replace(symbol, symbol_value)+"\n"
                    line=new_line
                    #print(line)

            verilog += line+"\n"
            if "//*Implementation of the logic*//" in line:
                verilog += self.verilog_code+"\n"


        self.verilog_code=verilog
        #print(self.output)
        self.write_verilog_to_file("./test.v")

    def create_verilog_embedded(self, param):
        verilog = ""
        temp = self.embedded_temp.split("\n")
        for line in temp:
            for key in self.verilog_param.keys():
                symbol = "$" + key + "$"
                if symbol in line:
                    if (isinstance(self.verilog_param[key], dict)):
                        line = line.replace(symbol, str(self.str_bitwidth(self.verilog_param[key])))
                    else:
                        line = line.replace(symbol, str(self.verilog_param[key]))

            if "%i%" in line:
                if "$Wi$" in line:
                    for i in range(len(param["weight"])):
                        line = line.replace("%i%",str(i))
                        line = line.replace("$Wi$", str(param["weight"][i]))
                elif "$bi$" in line:
                    for i in range(len(param["bias"])):
                        line = line.replace("%i%",str(i))
                        line = line.replace("$Wi$", str(param["weight"][i]))


            elif "//*Implementation of the logic*//" in line:
                line += "\n"+self.verilog_code + "\n"

            verilog += line + "\n"


        self.verilog_code = verilog
        self.write_verilog_to_file("./test.v")

    def apply_pass(self, node, ctx):
        self.print_node(node)
        if isinstance(node, pm.placeholder):
            # input node
            if (node.type_modifier=="input"):
                # activation input
                node_name = self.normalize_node_name(node)
                node.kwargs["verilog"] = "  reg  [input_bitwidth-1:0] _{};\n".format(node_name)
                node.kwargs["verilog_name"] =  "_{}".format(node_name)
                # activation input
                if(str(node.name).startswith("x")and node.type_modifier=="input"):
                    self.activation.append(node_name)
                # bias input
                elif (str(node.name).startswith("y") and node.type_modifier=="input"):
                    self.bias.append(node_name)
                # rate input (reco system)
                elif (str(node .name).startswith("r")):
                    self.rate.append(node_name)

            # weight input
            elif (node.type_modifier == "state" and node.op_name=="state"):
                node_name = self.normalize_node_name(node)
                node.kwargs["verilog"] = "  reg  [input_bitwidth-1:0] _{};\n".format(node_name)
                node.kwargs["verilog_name"] = "_{}".format(node_name)
                self.weight.append(node_name)

            # parameter input
            elif (node.type_modifier=="param"):
                if(node.name in self.param_list.keys()):
                    node_name = self.normalize_node_name(node)
                    node.kwargs["verilog"] = "  wire  [input_bitwidth-1:0] _{};\n".format(node_name) + \
                                             "  assign _{}= {};\n".format(node_name, node_name)
                    node.kwargs["verilog_name"] = node_name
                elif(str(node.name).startswith("sub")):
                    node.kwargs["verilog_name"] = node.name.replace(':', '_')

        #output
        elif (node.type_modifier == "state" and node.op_name == "write"):
            self.verilog_param["OutputN"] += 1
            node_name = self.normalize_node_name(node)
            node.kwargs["verilog"] = "  assign _{}= {};\n".format(node_name, node.args[0].kwargs["verilog_name"])
            node.kwargs["verilog_name"] = node_name

            self.output.append(node_name)

        # operation nodes
        elif(isinstance(node,pm.Node) and len(node.args)>=1):
            node_name="temp_"+str(self.counter)
            node.kwargs["verilog_name"]=node_name
            if (len(node.args)==2):
                # multipliication node
                if(node.op_name=="mul"):
                    node.kwargs["op"]="*"
                # addition or substraction ndoe
                elif(node.op_name=="add" or node.op_name=="sub"):
                    node.kwargs["op"] = "+" if (node.op_name=="add") else "-"
                # implement the verilog

                #great than node
                if(node.op_name=="gt"):
                    node.kwargs["bitwidth"]["bits"] = 2
                    args0,args1=self.args_str(node)
                    #print(parent0,parent1)
                    node.kwargs["verilog"]="  wire  [bitwidth-1:0]{};\n".format(node.kwargs["verilog_name"])
                    node.kwargs["verilog"]+="   assign {}= ({}>={})? 1:-1;\n".format(node.kwargs["verilog_name"],args0,args1)
                else:
                    #print(node.args[0].kwargs.keys())
                    args0, args1 = self.args_str(node)
                    node.kwargs["verilog"] = "  wire  [bitwidth-1:0]{};\n".format(node.kwargs["verilog_name"]) + \
                                             "  assign {} = {} {} {};\n".format(node.kwargs["verilog_name"], args0,
                                                                                node.kwargs["op"], args1)

            elif (node.op_name == "sigmoid"):
                node.kwargs["verilog"] = "  wire  [bitwidth-1:0] {};\n".format(node_name) + \
                                         "  sigmoid sig_{} (.in({}), .out({}));\n".format(self.counter, node.args[0].kwargs["verilog_name"], node_name)
                #self.print_node(node)
            elif (node.op_name == "write"):
                node.kwargs["verilog"] = "  wire  [bitwidth-1:0] {};\n".format(node_name) + \
                                         "  assign {}= {};\n".format(node_name, node.args[0].kwargs["verilog_name"])
                #self.print_node(node)

        if ("verilog" in node.kwargs.keys()):
            self.verilog_code += node.kwargs["verilog"]
        self.counter += 1

        return node

    def finalize_pass(self, node, ctx):
        # if (isinstance(node,pm.Node) and len(node.args)==2):
        #     print(node)
        #     print(node.op_name)
        #     if (isinstance(node.args[0], pm.Node)):
        #         print(node.args[0])
        #     if (isinstance(node.args[1], pm.Node)):
        #         print(node.args[1])
        #     print("\n")
        return node

    def normalize_node_name(self,node):
        if("/" in node.name):
            node_name = node.name.split("/")[-1]
        else:
            node_name=node.name
        # syntax modification
        node_name = node_name.replace(" ", "")
        node_name = node_name.replace(",", "_")
        node_name = node_name.replace(":", "_")
        node_name = node_name.replace("(", "_")
        node_name = node_name.replace(")", "_")
        return node_name

    def write_verilog_to_file(self,file_dir):
        f = open(file_dir, "w")
        f.write(self.verilog_code)
        f.close()

    def print_node(self,node):
        print(
            f"{self.counter}{node.type_modifier},{node.op_name},{node.name},{node.__class__},args:{node.args}")

    # return the args in string format
    def args_str(self,node):
        if(node.args[0].type_modifier=="param"):
            if(node.op_name == "mul"):
                args0='mu'
            else:
                args0 = node.args[0].name.split(':')[-1]
        elif (isinstance(node.args[0], pm.Node)):
            args0 = node.args[0].kwargs["verilog_name"]
        elif (isinstance(node.args[0], int)):
            args0 = str(node.args[0])
        else:
            print("args not found")

        if (node.args[1].type_modifier == "param"):
            if (node.op_name == "mul"):
                args0 = 'mu'
            else:
                args0 = node.args[0].name.split(':')[-1]
        elif (isinstance(node.args[1], pm.Node)):
            args1 = node.args[1].kwargs["verilog_name"]
            #parent1 = node.args[1].name
        elif (isinstance(node.args[1], int)):
            args1 = str(node.args[1])
        else:
            print("args not foundgit merge")
        # print(parent0)
        # print(parent1)
        return args0, args1


@register_pass
class VerilogGenerateFlexBitwidth(Pass):

    def __init__(self, param):
        self.param = param
        self.counter=0
        self.weight=[]
        self.activation=[]
        self.bias=[]
        self.rate=[]
        self.output=[]
        self.output_bitwidth={"input":0,"bits":0}
        self.verilog_code=""
        self.variable_list=["Activation","Weight","Bias","Rate","Output"]
        self.param_list=param["param_list"]

        with open('./non_embedded_temp.sv', 'r') as file:
            self.non_embedded_temp = file.read()
        with open("./embedded_temp.sv", 'r') as file:
            self.embedded_temp = file.read()
        # print(self.verilog_code)
        super(VerilogGenerateFlexBitwidth, self).__init__()

    def create_verilog_nonembedded(self, fdir):
        verilog=""
        temp=self.non_embedded_temp.split("\n")
        for line in temp:
            # for param in self.verilog_param.keys():
            #     symbol = "$" + param + "$"
            #     if symbol in line:
            #         print(symbol)
            #         if(isinstance(self.verilog_param[param], dict)):
            #             line = line.replace(symbol, str(self.str_bitwidth(self.verilog_param[param])))
            #         else:
            #             line = line.replace(symbol, str(self.verilog_param[param]))
            for param in self.variable_list:
                symbol = "$%" + param + "%$"
                if symbol in line:
                    new_line=""
                    if param=="Activation":
                        symbol_list=self.activation
                    elif param=="Weight":
                        symbol_list = self.weight
                    elif param=="Bias":
                        symbol_list = self.bias
                    elif param=="Rate":
                        symbol_list = self.rate
                        print("rate")
                    elif param=="Output":
                        symbol_list = self.output
                    if symbol_list:
                        for symbol_value in symbol_list:
                            new_line += line.replace(symbol, symbol_value)+"\n"
                        line=new_line
                    else:
                        line=f"//{line}"
                        break
            verilog += line+"\n"
            if "//*Implementation of the logic*//" in line:
                verilog += self.verilog_code+"\n\nendmodule"
                break
        print(self.rate)
        self.verilog_code=verilog
        self.write_verilog_to_file(fdir)

    def create_verilog_embedded(self, param):
        verilog = ""
        temp = self.embedded_temp.split("\n")
        for line in temp:
            for key in self.verilog_param.keys():
                symbol = "$" + key + "$"
                if symbol in line:
                    if (isinstance(self.verilog_param[key], dict)):
                        line = line.replace(symbol, str(self.str_bitwidth(self.verilog_param[key])))
                    else:
                        line = line.replace(symbol, str(self.verilog_param[key]))

            if "%i%" in line:
                if "$Wi$" in line:
                    for i in range(len(param["weight"])):
                        line = line.replace("%i%",str(i))
                        line = line.replace("$Wi$", str(param["weight"][i]))
                elif "$bi$" in line:
                    for i in range(len(param["bias"])):
                        line = line.replace("%i%",str(i))
                        line = line.replace("$Wi$", str(param["weight"][i]))


            elif "//*Implementation of the logic*//" in line:
                line += "\n"+self.verilog_code + "\n"

            verilog += line + "\n"


        self.verilog_code = verilog
        self.write_verilog_to_file("./test.v")



    def apply_pass(self, node, ctx):

        bitwidth_0={
            "input":0,
            "bits":0
            }
        node.kwargs["bitwidth"]=bitwidth_0.copy()

        #self.print_node(node)
        # bitwidth for input nodes
        if isinstance(node, pm.placeholder):
            if (node.type_modifier=="input"):
                # activation input
                if(str(node.name).startswith("x")and node.type_modifier=="input"):
                    node.kwargs["bitwidth"]["input"]+=1
                    # create verilog code for this activation node
                    node_name=node.name.split("/")[0]
                    node_string=node.name.split("/")[-1]
                    node_id=node_string[node_string.find("(")+1:node_string.find(")")].replace(",","_").strip()
                    node.kwargs["verilog_id"] =  node_id
                    name=node_name+'_'+node_id
                    name = name.replace(':', '_')
                    node.kwargs["verilog_name"] = name
                    self.activation.append(name)
                    node.kwargs["verilog"]="  wire  [{}-1:0] _{};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]), name)+\
                                           "  assign _{}= {};\n".format(name, name)
                # bias input
                elif (str(node.name).startswith("y") and node.type_modifier=="input"):
                    node.kwargs["bitwidth"]["input"] += 1
                    # create verilog code for this bias node
                    node_name = node.name.split("/")[0]
                    node_string = node.name.split("/")[-1]
                    # print(node_string)
                    node_id = node_string[node_string.find("(") + 1:node_string.find(")")].replace(",", "_").strip()
                    node.kwargs["verilog_id"] = node_id
                    name = node_name + '_' + node_id
                    name = name.replace(':', '_')
                    node.kwargs["verilog_name"] = name
                    self.bias.append(name)
                    node.kwargs["verilog"] = "  wire  [{}-1:0] _{};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]), name) + \
                                             "  assign _{}= {};\n".format(name, name)

                # rate input (reco system)
                elif (str(node .name).startswith("r")):
                    node.kwargs["bitwidth"]["input"] += 1
                    # create verilog code for this bias node
                    node_name = node.name.split("/")[0]
                    node_string = node.name.split("/")[-1]
                    # print(node_string)
                    node_id = node_string[node_string.find("(") + 1:node_string.find(")")].replace(",", "_").strip()
                    node.kwargs["verilog_id"] = node_id
                    name = node_name + '_' + node_id
                    name = name.replace(':', '_')
                    node.kwargs["verilog_name"] = name
                    self.rate.append(name)
                    node.kwargs["verilog"] = "  wire  [{}-1:0] _{};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]), name) + \
                                             "  assign _{}= {};\n".format(name, name)

            # weight input
            elif (node.type_modifier == "state"):
                node.kwargs["bitwidth"]["input"] += 1
                # create verilog code for this weight node
                node_name = node.name.split("/")[0]
                node_string = node.name.split("/")[-1]
                node_id = node_string[node_string.find("(") + 1:node_string.find(")")].replace(",", "_").replace(" ","")
                node.kwargs["verilog_id"] = node_id
                name = node_name + '_' + node_id
                name = name.replace(':', '_')
                node.kwargs["verilog_name"] = name
                self.weight.append(name)
                node.kwargs["verilog"] = "  wire  [{}-1:0] _{};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]), name) + \
                                         "  assign _{}= {};\n".format(name, name)

            #parameter input
            elif (node.type_modifier=="param"):
                if(node.name in self.param_list.keys()):
                    node.kwargs["bitwidth"]["input"] += 1
                    name = node.name.replace(':', '_')
                    node.kwargs["verilog_name"] = name
                    node.kwargs["verilog"] = "  wire  [{}-1:0] {};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]), name) + \
                                             "  assign {}=  {};\n".format(name, self.param_list[node.name])
                elif(str(node.name).startswith("sub")):
                    node.kwargs["bitwidth"]["input"] += 1
                    node.kwargs["verilog_name"] = node.name.replace(':', '_')


        # operation nodes
        elif(isinstance(node,pm.Node) and len(node.args)>=1):
            node.kwargs["verilog_name"]="temp_"+str(self.counter)
            if (len(node.args)==2):
                # multipliication node
                if(node.op_name=="mul"):
                    node.kwargs["op"]="*"
                    for a in node.args:
                        if isinstance(a, pm.Node):
                            for key in a.kwargs["bitwidth"].keys():
                                node.kwargs["bitwidth"][key] += a.kwargs["bitwidth"][key]
                        else:
                            node.kwargs["bitwidth"]["input"] +=1
                # addition or substraction ndoe
                elif(node.op_name=="add" or node.op_name=="sub"):
                    node.kwargs["op"] = "+" if (node.op_name=="add") else "-"
                    for a in node.args:
                        if isinstance(a, pm.Node):
                            # print(a.kwargs["verilog_name"])
                            a_bits= self.bits_caculate(a.kwargs["bitwidth"])
                            node_bits=self.bits_caculate(node.kwargs["bitwidth"])
                            if (a_bits + 1 >node_bits):
                                #print(a.kwargs["bitwidth"])
                                node.kwargs["bitwidth"]=a.kwargs["bitwidth"].copy()
                                node.kwargs["bitwidth"]["bits"]+=1
                        elif isinstance(a,int):
                            a_bits = self.param["param_bitwidth"]
                            node_bits = self.bits_caculate(node.kwargs["bitwidth"])
                            if (a_bits + 1 > node_bits):
                                # print(a.kwargs["bitwidth"])
                                node.kwargs["bitwidth"] = bitwidth_0.copy()
                                node.kwargs["bitwidth"]["bits"]= a_bits+1
                # implement the verilog

                #great than node
                if(node.op_name=="gt"):
                    node.kwargs["bitwidth"]["bits"] = 2
                    args0,args1=self.args_str(node)
                    node.kwargs["verilog"]="  wire [{}-1:0]{};\n".format(self.str_bitwidth(node.kwargs["bitwidth"]),node.kwargs["verilog_name"])
                    node.kwargs["verilog"]+="   assign {}= ({}>={})? 2'b01:2'b11;\n".format(node.kwargs["verilog_name"],args0,args1)
                else:
                    args0, args1 = self.args_str(node)
                    node.kwargs["verilog"] = "  wire [{}-1:0]{};\n".format(
                        self.str_bitwidth(node.kwargs["bitwidth"]), node.kwargs["verilog_name"]) + \
                                             "  assign {} = {} {} {};\n".format(node.kwargs["verilog_name"], args0,
                                                                                node.kwargs["op"], args1)

            # cast node
            elif(len(node.args)==1 and node.op_name=="cast"):
                node.kwargs["bitwidth"]=node.args[0].kwargs["bitwidth"]
                args=node.args[0]
                node.kwargs["verilog"] = "  wire [{}-1:0]{};\n".format(
                    self.str_bitwidth(node.kwargs["bitwidth"]), node.kwargs["verilog_name"])
                node.kwargs["verilog"] += "   assign {}= {};\n".format(node.kwargs["verilog_name"],args.kwargs["verilog_name"])


            if(self.is_output(node)):
                if (self.bits_caculate(node.kwargs["bitwidth"]) >self.bits_caculate(self.output_bitwidth)):
                    self.output_bitwidth = node.kwargs["bitwidth"].copy()
                name=self.output_name(node)
                print(name)
                node.kwargs["verilog_id"]=name.replace("out","")
                self.output.append(name)
                node.kwargs["verilog"] += ("  assign _{} = {};\n".format(name,node.kwargs["verilog_name"]))

        if ("verilog" in node.kwargs.keys()):
            self.verilog_code += node.kwargs["verilog"]
        self.counter += 1
        return node


    def finalize_pass(self, node, ctx):
        # if (isinstance(node,pm.Node) and len(node.args)==2):
        #     print(node)
        #     print(node.op_name)
        #     if (isinstance(node.args[0], pm.Node)):
        #         print(node.args[0])
        #     if (isinstance(node.args[1], pm.Node)):
        #         print(node.args[1])
        #     print("\n")
        return node

    def is_output(self,node):
        re = False
        if(len(node.args)==2 and node.op_name=="sub"):
            if (isinstance(node.args[0], pm.Node) and isinstance(node.args[1], pm.Node)):
                if(node.args[0].type_modifier=="state" or node.args[1].type_modifier=="state"):
                    re= True

        return re

    def bits_caculate(self,bitwidth):
        #print(bitwidth)
        if("input" in bitwidth.keys() and "bits" in bitwidth.keys()):
            bits = bitwidth["input"] * self.param["input_bitwidth"] + bitwidth["bits"]
            return bits


    def str_bitwidth(self,bitwidth):
        if isinstance(bitwidth,dict):
            bits = "input_bitwidth *"+str( bitwidth["input"])+"+"+ str(bitwidth["bits"])
            return bits
        else:
            exit("error")


    def write_verilog_to_file(self,file_dir):
        f = open(file_dir, "w")
        f.write(self.verilog_code)
        f.close()

    def print_node(self,node):
        node_name=node.name
        op = node.op_name
        type_m=node.type_modifier
        print("node:{},op_name:{},type:{}".format(node_name, op,type_m))
        print("args:{}".format(len(node.args)))
        if("verilog_name" in node.kwargs.keys()):
            verilog_name=node.kwargs["verilog_name"]
            print("verilog:{}".format(verilog_name))

        if(len(node.args)==2):
            if(isinstance(node.args[0], pm.Node)):
                args0 = node.args[0].name
                print("args0:{}".format(args0))
            if (isinstance(node.args[1], pm.Node)):
                args1 = node.args[1].name
                print("args1:{}".format(args1))
        elif(len(node.args)==1 ):
            if (isinstance(node.args[0], pm.Node)):
                args = node.args[0].name
                print("args:{}".format(args))
        elif (len(node.args) ==0):
            if(isinstance(node,pm.temp)):
                print(node.name)
                s=node.current_value()
                print(s.source)
                print(node.current_value())
                print(node.write_count)
                print(node.current_value().name)

    # generate args of a node in format of string
    def args_str(self,node):
        if (isinstance(node.args[0], pm.Node)):
            args0 = node.args[0].kwargs["verilog_name"]
        elif (isinstance(node.args[0], int)):
            args0 = str(node.args[0])
        else:
            print("args not found")
        if (isinstance(node.args[1], pm.Node)):
            args1 = node.args[1].kwargs["verilog_name"]
        elif (isinstance(node.args[1], int)):
            args1 = str(node.args[1])
        else:
            print("args not found")

        return args0, args1

    # generate output nodes name based on weight
    def output_name(self,node):
        if (isinstance(node.args[0], pm.Node) and node.args[0].type_modifier == "state"):
            weight_name= node.args[0].kwargs["verilog_name"]
        elif (isinstance(node.args[1], pm.Node) and node.args[1].type_modifier == "state"):
            weight_name = node.args[1].kwargs["verilog_name"]
        else:
            print("args don't have a verilog ID")
        output_name = weight_name.replace("W","out")
        output_name = output_name.replace("w", "out")
        return output_name