import numpy as np
array = np.zeros(10)
addr_width = 10
data_width = 16
wr_file = open('out.v','w')
num_pe = 4
k = 0;
wr_file.write("generate\nif(peId == "+str(k)+" ) begin\n")
for k in range(num_pe):
    wr_file.write("\talways @(*) begin\n") 
    wr_file.write("\t\tcase(address)\n") 
    
    for i in range(len(array)+1):
        if i != len(array):
            wr_file.write("\t\t\t"+str(addr_width)+"'d"+str(i)+" : ")
            data = i
        else:
            wr_file.write("\t\t\tdefault : ")
            data = 0
        wr_file.write("rdata = "+str(data_width)+"'b"+str(bin(data)[2:].zfill(data_width))+";\n") 

    wr_file.write("\t\tendcase\n\tend\n")
    if k ==num_pe-2:
        wr_file.write("end\nelse begin\n")
    elif k == num_pe-1:
        wr_file.write("end\n")
    else:
        wr_file.write("end\nelse if(peId == "+str(k+1)+" ) begin\n")
wr_file.write("endgenerate\n")
wr_file.close() 
