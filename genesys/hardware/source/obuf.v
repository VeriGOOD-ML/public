//
// OBUF - Output Buffer
//
// Soroush Ghodrati

`timescale 1ns/1ps
module obuf #(
  parameter integer  NUM_TAGS                     = 2,  // Log number of banks
  parameter integer  TAG_W                        = $clog2(NUM_TAGS),
  parameter integer  ARRAY_M                      = 16,
  parameter integer  DATA_WIDTH                   = 32,
  parameter integer  BUF_ADDR_WIDTH               = 10
)
(
  input  wire                                            clk,
  input  wire                                            reset,

  input  wire  [ NUM_TAGS*ARRAY_M                      -1 : 0 ]   buf_read_req,
  input  wire  [ NUM_TAGS*ARRAY_M*BUF_ADDR_WIDTH       -1 : 0 ]   buf_read_addr,
  output wire  [ NUM_TAGS*ARRAY_M*DATA_WIDTH           -1 : 0 ]   buf_read_data,
  
  input  wire  [ NUM_TAGS*ARRAY_M                      -1 : 0 ]   buf_write_req,
  input  wire  [ NUM_TAGS*ARRAY_M*BUF_ADDR_WIDTH       -1 : 0 ]   buf_write_addr,
  input  wire  [ NUM_TAGS*ARRAY_M*DATA_WIDTH           -1 : 0 ]   buf_write_data 
  );

  genvar n;
  generate
      for (n=0; n<NUM_TAGS; n=n+1) begin
          
          banked_scratchpad #(
              .DATA_WIDTH           (DATA_WIDTH),
              .ADDR_WIDTH           (BUF_ADDR_WIDTH),
              .NUM_BANKS            (ARRAY_M)            
          ) obuf_mem (
              .clk                  (clk),
              .reset                (reset),
              
              .bs_read_req          (buf_read_req[(n+1)*ARRAY_M-1:n*ARRAY_M]),
              .bs_read_addr         (buf_read_addr[(n+1)*ARRAY_M*BUF_ADDR_WIDTH-1:n*ARRAY_M*BUF_ADDR_WIDTH]),
              .bs_read_data         (buf_read_data[(n+1)*ARRAY_M*DATA_WIDTH-1:n*ARRAY_M*DATA_WIDTH]),
              .bs_write_req         (buf_write_req[(n+1)*ARRAY_M-1:n*ARRAY_M]),
              .bs_write_addr        (buf_write_addr[(n+1)*ARRAY_M*BUF_ADDR_WIDTH-1:n*ARRAY_M*BUF_ADDR_WIDTH]),
              .bs_write_data        (buf_write_data[(n+1)*ARRAY_M*DATA_WIDTH-1:n*ARRAY_M*DATA_WIDTH])
          );
      end
  endgenerate
  
endmodule
