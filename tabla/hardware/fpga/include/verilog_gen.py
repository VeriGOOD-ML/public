import numpy as np

### Derive these variables from config
addr_width = 10
data_width = 16
num_pe = 4
inst_num = 10
#####

##output file
wr_file = open('iBuffer_ASIC.v','w')

## The instructions have to be appended to contents of the file below.
## I read this file first and write its contents to output file
fin = open("iBuffer_ASIC_header.v", "r")
data1 = fin.read()
wr_file.write(data1)
fin.close()
#############################

wr_file.write("\n\n\n")
wr_file.write("generate\nif(peId == "+str(0)+" ) begin\n")
for k in range(num_pe):
    wr_file.write("\talways @(*) begin\n") 
    wr_file.write("\t\tcase(address)\n")    
    for i in range(inst_num):
        instruction = i ### replace this with actual instruction
        if i != inst_num-1:
            wr_file.write("\t\t\t"+str(addr_width)+"'d"+str(i)+" : ")
        else:
            wr_file.write("\t\t\tdefault : ")
        wr_file.write("rdata = "+str(data_width)+"'b"+str(bin(instruction)[2:].zfill(data_width))+";\n") 

    wr_file.write("\t\tendcase\n\tend\n")
    if k ==num_pe-2:
        wr_file.write("end\nelse begin\n")
    elif k == num_pe-1:
        wr_file.write("end\n")
    else:
        wr_file.write("end\nelse if(peId == "+str(k+1)+" ) begin\n")
wr_file.write("endgenerate\n\n")
wr_file.write("endmodule\n")
wr_file.close() 
