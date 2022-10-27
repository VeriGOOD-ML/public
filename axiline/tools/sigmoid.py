import argparse
import math

def sigmoid_func(x):
    return 1 / (1 + math.exp(-x))

def inv_sigmoid(y):
    if(y>0 and y<1):
        return math.log(y/(1 - y))
    else:
        exit("math error with log")

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

class sigmoid():

    def __init__(self,bitwidth=8,fracture=3):
        self.fracture_index=0
        self.integer_index=0
        self.bitwidth=bitwidth
        self.fracture=fracture
        self.temp_dir = "./templates/sigmoid_func.v"
        self.write_dir = "./output/sigmoid_func.v"
        with open(self.temp_dir, 'r') as file:
            self.temp = file.readlines()
        self.param={}
        self.compute(bitwidth,fracture)

    def compile(self):
        verilog=[]
        for line in self.temp:
            for parameter in self.param.keys():
                if(f"%{parameter}%" in line):
                    line=line.replace(f"%{parameter}%",str(self.param[parameter]))
            verilog.append(line)
        # print(verilog)

        fracIndex=self.param["fracIndex"]
        new_verilog=[]
        for line in verilog:
            if(("%$index$%" in line) and ("%$data$%" in line)):
                # positive sigmoid and 0
                max=2 ** (self.param["indexLen"] - 1)
                for index in range(max):
                    value=index/(2**fracIndex)
                    sigmoid = sigmoid_func(value)
                    data=bin(math.floor(sigmoid*(2**self.fracture)))[2:]
                    new_line=line.replace("%$index$%",str(index))
                    new_line = new_line.replace("%$data$%", str(data).zfill(self.bitwidth))
                    new_verilog.append(new_line)

                # negative sigmoid
                for index in range (1,max):
                    value=(max-index)/(2**fracIndex)
                    sigmoid = sigmoid_func(-value)
                    # print(index,max-index,value,sigmoid)
                    data=bin(math.floor(sigmoid*(2**self.fracture)))[2:]
                    new_line=line.replace("%$index$%",str(index+max))
                    new_line = new_line.replace("%$data$%", str(data).zfill(self.bitwidth))
                    new_verilog.append(new_line)


            else:
                new_verilog.append(line)
        self.verilog=new_verilog
        #print(self.verilog)


    def compute(self,bitwidth,fracture):
        self.param["dataLen"]=bitwidth
        self.param["fracLen"]=fracture
        self.param['fracIndex']= fracture-2 # index = fracture-2
        self.param["indexLen"] = fracture+2 # 3 int and 1 sign
        self.param["intIndex"] =3
        # print(self.param)


    def write(self):
        with open(self.write_dir, 'w') as file:
            file.writelines(self.verilog)



if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description='Sigmoid Function Generator for fixed bit numbers')
    argparser.add_argument('-b', '--bitwidth', required=True,
                           help='Bitwidth of the fixed point numbers')
    argparser.add_argument('-fb', '--fracture', required=True,
                           help='bitwidth for fracture in fixed bit numbers')
    args = argparser.parse_args()
    args.bitwidth=int(args.bitwidth)
    args.fracture=int(args.fracture)
    if (args.bitwidth>args.fracture and args.fracture>0):
        s = sigmoid(args.bitwidth,args.fracture)
        s.compile()
        s.write()
    else:
        raise RuntimeError(f"Invalid Bidwidth and Fractures")

