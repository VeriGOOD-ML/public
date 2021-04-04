`timescale 1ns / 1ps


module vector_memory
#(
  parameter integer DATA_WIDTH    = 10,
  parameter integer ADDR_WIDTH    = 12,
  parameter NUM_ELEM              = 16
)(
    input  wire                         clk,
    input  wire                         reset,

    input  wire [ NUM_ELEM  -1 : 0 ]             read_req,
    input  wire [ ADDR_WIDTH*NUM_ELEM  -1 : 0 ]  read_addr,
    output wire [ DATA_WIDTH*NUM_ELEM  -1 : 0 ]  read_data,

    input  wire [ NUM_ELEM  -1 : 0 ]             write_req,
    input  wire [ ADDR_WIDTH*NUM_ELEM  -1 : 0 ]  write_addr,
    input  wire [ DATA_WIDTH*NUM_ELEM  -1 : 0 ]  write_data
    );
    
    generate
    for ( genvar gv = 0 ; gv < NUM_ELEM ; gv = gv + 1) begin
        
        wire [DATA_WIDTH - 1 : 0] mem_data_in,mem_data_out;
        
        assign mem_data_in = write_data[gv*DATA_WIDTH+:DATA_WIDTH];
        assign read_data[gv*DATA_WIDTH+:DATA_WIDTH] = mem_data_out;
        
        ram
        #(
          .DATA_WIDTH(DATA_WIDTH),
          .ADDR_WIDTH(ADDR_WIDTH )
        ) vector_memory_bank
        (
          .clk		   (    clk                 ),
          .reset       (	reset               ),
        
          .read_req    (    read_req[gv]                                    ),
          .read_addr   (	read_addr[ADDR_WIDTH*gv +: ADDR_WIDTH]          ),
          .read_data   (	mem_data_out                                    ),   

          .write_req   (	write_req[gv]                                   ),   
          .write_addr  (	write_addr[ADDR_WIDTH*gv +: ADDR_WIDTH]         ),   
          .write_data  (	mem_data_in                                     )    
        );
        
    end
    endgenerate 
    
endmodule
