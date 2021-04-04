cfg_template = """
`timescale 1ns / 100ps
`define FPGA 1
`define NS_SIZE {ns_size}
//INDEXES FOR EACH NAMESPACE
`define INDEX_INST 			{index_inst}
`define INDEX_DATA 			`NS_SIZE
`define INDEX_WEIGHT 		`NS_SIZE
`define INDEX_META 			`NS_SIZE
`define INDEX_INTERIM   	`NS_SIZE

//INDEX INSIDE THE INSTRUCTION
`define INDEX_IN_INST		`NS_SIZE

`define BUS_READ_DEPTH  {bus_read_depth}
`define BUS_FIFO_DEPTH  {bus_fifo_depth}
`define NEIGH_FIFO_DEPTH  {nb_fifo_depth}

//NUMBER OF VALID PEs
`define LOG_NUM_PU 		{log_num_pu}
`define LOG_NUM_PE 		{log_num_pe}
`define NUM_PE_VALID 		{num_pe}

`define LOG_MEM_NAME_SPACES {log_mem_ns}
//COMPUTE ELEMENTS
//`define GAUSSIAN
//`define DIV
//`define SQRT
//`define SIGMOID

//MEM INSTRUCTION FILE 
`define MEM_INST_INIT  "{program_name}/mem-inst/memInst_init.v"
`define MEM_INST_INIT_FPGA  		"{program_name}/mem-inst/memInst.txt"

//COMPUTE INSTRUCTION FILE 
`define COMPUTE_INST_INIT	"{program_name}/compute-inst/"

//META PARAMETER FILE 
`define META_DATA_FILE		"{program_name}/meta.txt"

//PE COUNTER FOR WEIGHTS
//unused in asic flow as of now
`define WEIGHT_CTRL_INIT "{program_name}/mem-inst/weightInst.txt"
"""


compute_instr_template = """
`timescale 1ns/1ps
module instruction_memory #(
    parameter integer addrLen = 5,
    parameter integer dataLen = 32,
    parameter integer peId  = 1
)(
    input clk,
    input rstn,
    
    input stall,
    input start,
    input restart,
    
    output reg [dataLen - 1: 0] data_out
);
//--------------------------------------------------------------------------------------
//reg [dataLen - 1: 0] mem  [0: (1 << addrLen) - 1];
reg [addrLen-1:0]        address;
reg enable;
reg [dataLen - 1: 0] rdata;
wire end_of_instruction;
always @(posedge clk or negedge rstn)
    if(~rstn)
        enable <= 1'b0;
    else if(start)
        enable <= 1'b1;
    else if(end_of_instruction)
       enable <= 1'b0;
always @(posedge clk or negedge rstn) begin
    if(~rstn)
        address <= {{addrLen{{1'b0}}}};
    else begin
        if(end_of_instruction)
            address <= {{addrLen{{1'b0}}}};
        else if(~stall && enable )
            address <= address + {{{{addrLen-1{{1'b0}}}},1'b1}};   
    end     
end
always @(posedge clk or negedge rstn) begin
    if(~rstn)
        data_out <= {{1'b1,{{dataLen-1{{1'b0}}}}}};
    else if((~stall && enable && ~end_of_instruction)||(end_of_instruction && start))
       data_out <= rdata;
end
    
assign end_of_instruction = (data_out[dataLen-1:dataLen-5] == 5'b0);
/****************************************************************************/
{compute_instr}
/*****************************************************************************/
endmodule
"""
