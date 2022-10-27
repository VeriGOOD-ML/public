from pathlib import Path
from axiline.compiler.templates import Templates
from axiline.compiler.impl_node import ImplNode

def generate_verilog(impl_graph, template_path , output_path):
    temp_dir = f"{template_path}/accelerator.v"
    file_temp = open(temp_dir, 'r')
    lines = file_temp.readlines()
    in_ports = ''
    out_ports = ''
    logic = ''
    stage = ''

    for node in impl_graph:
        node.verilog_name = verilog_string(node.name)
        if node.level == 3:
            verilog_name=verilog_string(node.name)
            in_ports += f"input [INPUT_BITWIDTH-1:0]{verilog_name},\n"
        elif node.level == 2:
            template_dir = f"{template_path}/{node.op_name}.v"
            parameters={}
            for i in range(len(node.predecessors)):
                pre=node.predecessors[i]
                args_name=verilog_string(impl_graph[pre].name)
                args_op_name = impl_graph[pre].op_name
                parameters[args_op_name]=args_name
            parameters['output']=verilog_string(node.name)
            rtl = module_rtl(template_dir, parameters)
            stage+=rtl
        elif node.level ==0:
            temp = low_level_verilog(node,impl_graph)
            logic +=temp
        # add out ports
        if len(node.successors) == 0:
            out_ports += f"output [BITWIDTH-1:0]{node.verilog_name}\n"



    for i in range(len(lines)):
        if "%input%" in lines[i]:
            lines[i] = in_ports
        elif "%output%" in lines[i]:
            lines[i] = out_ports
        elif "%stage%" in lines[i]:
            lines[i] = stage
        elif "%logic%" in lines[i]:
            lines[i] = logic

    with open(f"{output_path}/accelerator.v", 'w') as f:
        f.writelines(lines)



def low_level_verilog(node,impl_graph):
    if isinstance(node, ImplNode):
        # basic operation nodes
        if len(node.predecessors)>=2:
            arg0 = impl_graph[node.predecessors[0]].verilog_name
            arg1 = impl_graph[node.predecessors[1]].verilog_name
            if node.op_name == 'mul':
                temp =f"wire[BITWIDTH-1:0]{node.verilog_name};\n"
                temp +=f"assign {node.verilog_name}={arg0} *{arg1};\n"
            elif node.op_name == 'gt':
                temp = f"wire {node.verilog_name};\n"
                temp += f"assign {node.verilog_name}={arg0}>{arg1};\n"
            elif node.op_name == 'add':
                temp = f"wire[BITWIDTH-1:0]{node.verilog_name};\n"
                temp += f"assign {node.verilog_name}={arg0}+{arg1};\n"
            elif node.op_name == 'sub':
                temp = f"wire[BITWIDTH-1:0]{node.verilog_name};\n"
                temp += f"assign {node.verilog_name}={arg0}-{arg1};\n"
        elif len(node.predecessors)==1 and node.op_name == 'pipe':
            arg0 = impl_graph[node.predecessors[0]].verilog_name
            temp = f"reg[BITWIDTH-1:0]{node.verilog_name};\n"
            temp += f"always@(posedge clk or negedge rst_n) begin\n" \
                    f"  if(~rst_n) {node.verilog_name}<=0;\n" \
                    f"  else {node.verilog_name}<={arg0};\n" \
                    f"end\n"
        return temp

def module_rtl(template_dir, parameters):
    if not Path(template_dir).exists():
        exit(f"{template_dir}")
    file = open(template_dir, 'r')
    lines = file.readlines()
    new_lines = []
    for line in lines:
        for parameter in parameters.keys():
            if f"%{parameter}%" in line:
                line = line.replace(f"%{parameter}%",parameters[parameter])
        new_lines.append(line)
    # need to update the output wire declaration and connect

    rtl = "".join(new_lines)
    return rtl


def verilog_string(name):
    if isinstance(name,str):
        new_name= name.replace('/',"_")
        new_name = new_name.replace(':', "_")
        return new_name
    else:
        exit("need a string name")
