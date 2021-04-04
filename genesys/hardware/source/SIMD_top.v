`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/02/2020 11:42:19 AM
// Design Name: 
// Module Name: SIMD_top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module SIMD_top #(
	parameter NS_ID_BITS 			=	3,
	parameter NS_INDEX_ID_BITS 		=	5,
	parameter OPCODE_BITS 			=	4,
	parameter FUNCTION_BITS 		=	4,
	parameter INSTRUCTION_WIDTH 	= 	OPCODE_BITS + FUNCTION_BITS + 3*(NS_ID_BITS + NS_INDEX_ID_BITS),

	parameter IMEM_ADDR_WIDTH			= 10,
	
	parameter NUM_ELEM             = 16,
	
	parameter DATA_WIDTH = 32,
	
	parameter DATA_BUS_WIDTH = DATA_WIDTH*NUM_ELEM,
	parameter VMEM_DATA_WIDTH = DATA_WIDTH,
	parameter IMM_DATA_WIDTH = DATA_WIDTH,
	parameter IMMEDIATE_WIDTH = 32,
	
	parameter VMEM_ADDR_WIDTH = 11,
	parameter IMM_ADDR_WIDTH = 6,
	parameter BASE_STRIDE_WIDTH     = 4*(NS_INDEX_ID_BITS + NS_ID_BITS),
	parameter OBUF_ADDR_WIDTH = BASE_STRIDE_WIDTH,
	parameter IBUF_ADDR_WIDTH = BASE_STRIDE_WIDTH,
	parameter EXT_MEM_ADDR_WIDTH = BASE_STRIDE_WIDTH,
	
	parameter GROUP_ID_W = 4,
	parameter MAX_NUM_GROUPS = (1<<GROUP_ID_W),

    
    parameter MEM_REQ_W           = 16,
    parameter BASE_ADDR_SEGMENT_W = 16,
    parameter NUM_TAGS  = 1,
    parameter TAG_W     = $clog2(NUM_TAGS),
    parameter VMEM_TAG_BUF_ADDR_W = TAG_W + VMEM_ADDR_WIDTH,
    parameter LD_ST_LOW_DATA_WIDTH = 8,
    
    parameter AXI_ADDR_WIDTH  = 42,
    parameter AXI_DATA_WIDTH = 256,
    parameter AXI_BURST_WIDTH = 8,
    parameter WSTRB_W         = AXI_DATA_WIDTH / 8,
    parameter AXI_ID_WIDTH    = 1,

	
	parameter INTERLEAVE = 1,
    parameter integer  GROUP_ENABLED                = 0
	
	
)(
    input       								 clk,
    input       								 reset,
									 
    input       								 start,
    input  [GROUP_ID_W-1:0]     				 group_id_s,
    output      								 ready,
    output      								 simd_tiles_done,
												 
    input       								 block_done,
    input                                        imem_wr_req,
    input   [IMEM_ADDR_WIDTH-1:0]                imem_wr_addr,
    input   [INSTRUCTION_WIDTH-1:0]              imem_wr_data,
    
    input   [DATA_BUS_WIDTH -1:0]                obuf_data,
    output  [OBUF_ADDR_WIDTH*NUM_ELEM-1:0]       obuf_rd_addr,
    output  [NUM_ELEM-1:0]                       obuf_rd_req,

    output  [DATA_BUS_WIDTH -1:0]                ibuf_wr_data,
    output  [IBUF_ADDR_WIDTH*NUM_ELEM-1:0]       ibuf_wr_addr,
    output  [NUM_ELEM-1:0]                       ibuf_wr_req,
    
    input   [DATA_BUS_WIDTH -1:0]                ext_mem_data,
    output  [EXT_MEM_ADDR_WIDTH*NUM_ELEM-1:0]    ext_mem_rd_addr,
    output  [NUM_ELEM-1:0]                       ext_mem_rd_req,

    output  [DATA_BUS_WIDTH -1:0]                ext_mem_wr_data,
    output  [EXT_MEM_ADDR_WIDTH*NUM_ELEM-1:0]    ext_mem_wr_addr,
    output  [NUM_ELEM-1:0]                       ext_mem_wr_req,
    
    output                                       done,
    output  [3:0]                                group_id,
 
    output wire  [ AXI_ADDR_WIDTH       -1 : 0 ]        mws_awaddr,
    output wire  [ AXI_BURST_WIDTH      -1 : 0 ]        mws_awlen,
    output wire  [ 3                    -1 : 0 ]        mws_awsize,
    output wire  [ 2                    -1 : 0 ]        mws_awburst,
    output wire                                         mws_awvalid,
    input  wire                                         mws_awready,
// Master Interface Write Data
    output wire  [ AXI_DATA_WIDTH       -1 : 0 ]        mws_wdata,
    output wire  [ WSTRB_W              -1 : 0 ]        mws_wstrb,
    output wire                                         mws_wlast,
    output wire                                         mws_wvalid,
    input  wire                                         mws_wready,
// Master Interface Write Response
    input  wire  [ 2                    -1 : 0 ]        mws_bresp,
    input  wire                                         mws_bvalid,
    output wire                                         mws_bready,
// Master Interface Read Address
    output wire  [ AXI_ADDR_WIDTH       -1 : 0 ]        mws_araddr,
    output wire  [ AXI_ID_WIDTH         -1 : 0 ]        mws_arid,
    output wire  [ AXI_BURST_WIDTH      -1 : 0 ]        mws_arlen,
    output wire  [ 3                    -1 : 0 ]        mws_arsize,
    output wire  [ 2                    -1 : 0 ]        mws_arburst,
    output wire                                         mws_arvalid,
    input  wire                                         mws_arready,
// Master Interface Read Data
    input  wire  [ AXI_DATA_WIDTH       -1 : 0 ]        mws_rdata,
    input  wire  [ AXI_ID_WIDTH         -1 : 0 ]        mws_rid,
    input  wire  [ 2                    -1 : 0 ]        mws_rresp,
    input  wire                                         mws_rlast,
    input  wire                                         mws_rvalid,
    output wire                                         mws_rready 

    );
    
/*************************************************/   
localparam ENABLE_PIPELINE_AFTER_NAMESPACE_MUX = 1;


wire  [VMEM_ADDR_WIDTH*NUM_ELEM-1:0]     vmem_rd_addr1,vmem_wr_addr1;
wire  [VMEM_ADDR_WIDTH*NUM_ELEM-1:0]     vmem_rd_addr2,vmem_wr_addr2;
wire  [NUM_ELEM-1:0] vmem_rd_req1,vmem_wr_req1;
wire  [NUM_ELEM-1:0] vmem_rd_req2,vmem_wr_req2;
wire  [ DATA_WIDTH*NUM_ELEM  -1 : 0 ] vmem_data1,vmem_wr_data1;
wire  [ DATA_WIDTH*NUM_ELEM  -1 : 0 ] vmem_data2,vmem_wr_data2;

wire  [IMM_ADDR_WIDTH-1:0]     imm_rd_addr,imm_wr_addr;
wire   imm_rd_req,imm_wr_req;
wire  [ IMM_DATA_WIDTH  -1 : 0 ] imm_mem_out;
wire  [ IMM_DATA_WIDTH*NUM_ELEM  -1 : 0 ] imm_data;
wire  [ IMM_DATA_WIDTH  -1 : 0 ] imm_wr_data;
wire  [ 15 : 0 ] imm_wr_data_w;


wire [IMEM_ADDR_WIDTH-1:0] imem_rd_address;
wire imem_rd_req;
wire [INSTRUCTION_WIDTH-1:0] imem_rd_data;

wire [OPCODE_BITS-1 : 0] opcode,opcode_compute;
wire [FUNCTION_BITS-1 : 0] fn,fn_compute;
wire [NS_ID_BITS-1 : 0] dest_ns_id, src1_ns_id, src2_ns_id;
wire [NS_INDEX_ID_BITS-1 : 0] dest_ns_index_id, src1_ns_index_id, src2_ns_index_id;

wire  [ DATA_WIDTH*NUM_ELEM  -1 : 0 ] src1_muxed,src2_muxed;

reg [DATA_WIDTH-1 : 0] src1_data[0:NUM_ELEM-1];
reg [DATA_WIDTH-1 : 0] src2_data[0:NUM_ELEM-1];
wire [DATA_WIDTH-1 : 0] data_compute_out[0:NUM_ELEM-1];
wire [DATA_WIDTH*NUM_ELEM-1:0] compute_out_bus;

wire cond_move_inst,in_loop;
//////////////////////////////////////////////////////

//////////////////////////////////
wire [5:0]						iterator_read_req;
wire [5:0]						iterator_write_req_base;
wire [5:0]						iterator_write_req_stride;
wire [5:0]						buffer_write_req;
wire [5:0]						buffer_read_req;
wire [5:0]						mem_bypass;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_0;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_0;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_0;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_0;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_0;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_0;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_1;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_1;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_1;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_1;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_1;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_1;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_2;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_2;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_2;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_2;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_2;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_2;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_3;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_3;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_3;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_3;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_3;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_3;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_4;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_4;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_4;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_4;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_4;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_4;
//////////////////////////////////
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_read_addr_5;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_base_5;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_base_5;
wire [NS_INDEX_ID_BITS-1 :0] 		iterator_write_addr_stride_5;
wire [BASE_STRIDE_WIDTH-1 : 0]		iterator_data_stride_5;
wire [BASE_STRIDE_WIDTH-1 : 0]		base_plus_stride_5;
//////////////////////////////////

wire   [FUNCTION_BITS + OPCODE_BITS-1:0]      opcode_fn,opcode_fn_to_iter_mem,opcode_fn_to_mem_stage,opcode_fn_to_mux_stage,opcode_fn_to_compute_stage;
wire   [IMMEDIATE_WIDTH-1:0]      immediate_to_iter_mem,immediate_to_mem_stage,immediate_to_mux_stage,immediate_to_compute_stage,immediate_final;
wire   [15:0]      integer_bits_to_iter_mem,integer_bits_to_mem_stage,integer_bits_to_mux_stage,integer_bits_to_compute_stage;
wire  [NS_ID_BITS-1:0]            src1_ns_id_to_iter_mem,src1_ns_id_to_mem_stage,src1_ns_id_to_mux_stage;
wire  [NS_ID_BITS-1:0]            src2_ns_id_to_iter_mem,src2_ns_id_to_mem_stage,src2_ns_id_to_mux_stage;
wire  [NS_ID_BITS-1:0]            dest_ns_id_to_iter_mem,dest_ns_id_to_mem_stage;
wire  [NS_INDEX_ID_BITS-1:0]      dest_ns_index_id_to_iter_mem,src1_ns_index_id_to_iter_mem,src2_ns_index_id_to_iter_mem;
wire [5:0]						buffer_write_req_to_mem_stage,buffer_write_req_to_mux_stage,buffer_write_req_to_compute_stage,buffer_write_req_final;
wire [5:0]						buffer_read_req_to_mem_stage;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_0;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_0;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_1;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_1;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_2;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_2;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_3;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_3;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_4;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_4;
wire [BASE_STRIDE_WIDTH-1 : 0]	    iterator_stride_5;
wire [BASE_STRIDE_WIDTH-1 : 0]	    buffer_address_5;
localparam BUFFERS_WR_ADDR_WIDTH = BASE_STRIDE_WIDTH;
reg [BUFFERS_WR_ADDR_WIDTH-1:0] buffers_wr_addr;
wire [BUFFERS_WR_ADDR_WIDTH*6-1:0] buffers_rd_addr;
wire [BUFFERS_WR_ADDR_WIDTH-1:0] buffers_wr_addr_to_mux_stage,buffers_wr_addr_to_compute_stage,buffers_wr_addr_final;

wire [BASE_STRIDE_WIDTH-1:0] buf_wr_addr_interleaved[0:NUM_ELEM-1];
wire [BASE_STRIDE_WIDTH*6-1:0] buf_rd_addr_interleaved[0:NUM_ELEM-1];
wire [IMM_DATA_WIDTH-1:0] imm_data_interleaved[0:NUM_ELEM-1];
wire [5:0] buf_wr_req_interleaved[0:NUM_ELEM-1];
wire [5:0] buf_rd_req_interleaved[0:NUM_ELEM-1];
reg  [2:0] src1_ns_id_interleaved[0:NUM_ELEM-1];
reg  [2:0] src2_ns_id_interleaved[0:NUM_ELEM-1];

wire nested_loop_done;

wire                                        stall;
wire                                        ld_mem_simd_done;
wire                                        st_mem_simd_done;
wire [GROUP_ID_W                -1:0]       ld_st_group_id;
wire [MAX_NUM_GROUPS            -1:0]       ld_config_done;
wire [MAX_NUM_GROUPS            -1:0]       st_config_done;    

wire                                        in_ld_st; 


wire  [NUM_ELEM        -1:0]               ld_st_vmem1_write_req;
wire  [NUM_ELEM*VMEM_TAG_BUF_ADDR_W-1:0]   ld_st_vmem1_write_addr;
wire  [NUM_ELEM*DATA_WIDTH -1:0]           ld_st_vmem1_write_data;
wire  [NUM_ELEM        -1:0]               ld_st_vmem1_read_req;
wire  [NUM_ELEM*VMEM_TAG_BUF_ADDR_W-1:0]   ld_st_vmem1_read_addr;
wire  [NUM_ELEM*DATA_WIDTH -1:0]           ld_st_vmem1_read_data;
// VMEM2
wire  [NUM_ELEM        -1:0]               ld_st_vmem2_write_req;
wire  [NUM_ELEM*VMEM_TAG_BUF_ADDR_W-1:0]   ld_st_vmem2_write_addr;
wire  [NUM_ELEM*DATA_WIDTH -1:0]           ld_st_vmem2_write_data;
wire  [NUM_ELEM        -1:0]               ld_st_vmem2_read_req;
wire  [NUM_ELEM*VMEM_TAG_BUF_ADDR_W-1:0]   ld_st_vmem2_read_addr;
wire  [NUM_ELEM*DATA_WIDTH -1:0]           ld_st_vmem2_read_data;

wire  [VMEM_ADDR_WIDTH*NUM_ELEM-1:0]     _vmem_rd_addr1,_vmem_wr_addr1;
wire  [VMEM_ADDR_WIDTH*NUM_ELEM-1:0]     _vmem_rd_addr2,_vmem_wr_addr2;
wire  [NUM_ELEM-1:0] _vmem_rd_req1,_vmem_wr_req1;
wire  [NUM_ELEM-1:0] _vmem_rd_req2,_vmem_wr_req2;
wire  [ DATA_WIDTH*NUM_ELEM  -1 : 0 ] _vmem_data1,_vmem_wr_data1;
wire  [ DATA_WIDTH*NUM_ELEM  -1 : 0 ] _vmem_data2,_vmem_wr_data2;



///////////////////////////////////////// Stage 0 ///////////////////////////
///////////////////////////////////////// Instruction Memory ///////////////////////////
ram
#(
  .DATA_WIDTH(INSTRUCTION_WIDTH),
  .ADDR_WIDTH(IMEM_ADDR_WIDTH )
) instruction_memory
(
  .clk		   (    clk                 ),
  .reset       (	reset               ),

  .read_req    (    imem_rd_req	        ),
  .read_addr   (	imem_rd_address     ),
  .read_data   (	imem_rd_data        ),

  .write_req   (	imem_wr_req         ),
  .write_addr  (	imem_wr_addr        ),
  .write_data  (	imem_wr_data        )
);

//=========================== GROUP START ADDR Buffer=====================
localparam integer INST_GROUP_OPCODE        = 10;
localparam integer INST_GROUP_START         = 0;
localparam integer INST_GROUP_END           = 1;
localparam integer INST_GROUP_SA            = 0;
localparam integer INST_GROUP_SIMD          = 1;

wire                                        group_mem_rd_req;
wire  [ GROUP_ID_W             -1: 0]       group_mem_rd_addr;
wire  [ IMEM_ADDR_WIDTH        -1: 0]       group_mem_rd_data;

wire                                        group_mem_wr_req;
wire  [ GROUP_ID_W             -1: 0]       group_mem_wr_addr;
wire  [ IMEM_ADDR_WIDTH        -1: 0]       group_mem_wr_data;

reg                                         _group_mem_wr_req;
reg  [ GROUP_ID_W             -1: 0]        _group_mem_wr_addr;
reg  [ IMEM_ADDR_WIDTH        -1: 0]        _group_mem_wr_data;

wire                                        group_mem_rd_v;
reg                                         _group_mem_rd_v;

wire                                        inst_group_v;
wire                                        inst_group_id;


assign inst_group_v = imem_wr_req && (imem_wr_data[31:28] == INST_GROUP_OPCODE && imem_wr_data[27] == INST_GROUP_SIMD && imem_wr_data[26] == INST_GROUP_START);
assign inst_group_id = imem_wr_data[25:22];

always @(posedge clk) begin
   if (reset) begin
       _group_mem_wr_req <= 1'b0;
       _group_mem_wr_addr <= 0;
       _group_mem_wr_data <= 0;
   end 
   else begin
       _group_mem_wr_req <= inst_group_v;
       _group_mem_wr_addr <= inst_group_id;
       _group_mem_wr_data <= imem_wr_addr;    
   end 
end

assign group_mem_wr_req = _group_mem_wr_req;
assign group_mem_wr_addr = _group_mem_wr_addr;
assign group_mem_wr_data = _group_mem_wr_data;

ram
#(
  .DATA_WIDTH(IMEM_ADDR_WIDTH),
  .ADDR_WIDTH(GROUP_ID_W )
) group_start_addr_memory
(
  .clk         (    clk                 ),
  .reset       (    reset               ),

  .read_req    (    group_mem_rd_req         ),
  .read_addr   (    group_mem_rd_addr        ),
  .read_data   (    group_mem_rd_data        ),

  .write_req   (    group_mem_wr_req         ),
  .write_addr  (    group_mem_wr_addr        ),
  .write_data  (    group_mem_wr_data        )
);

always @(posedge clk) begin
   if (reset) 
       _group_mem_rd_v <= 1'b0;
   else
       _group_mem_rd_v <= group_mem_rd_req;
end
assign group_mem_rd_v = _group_mem_rd_v;
///////////////////////////////////////// Stage 1 ///////////////////////////
///////////////////////////////////////// Instruction Decode ///////////////////////////

/* Splits the instruction, processes loop instruction and generates rd address */
SIMD_instruction_decoder #(   
    .NS_ID_BITS         ( NS_ID_BITS            ), 
    .NS_INDEX_ID_BITS   ( NS_INDEX_ID_BITS      ),
    .OPCODE_BITS        ( OPCODE_BITS           ),
    .FUNCTION_BITS      ( FUNCTION_BITS         ),
    .GROUP_ID_W         ( GROUP_ID_W            ),
    .INSTRUCTION_WIDTH  ( INSTRUCTION_WIDTH     ),
    
    .IMEM_ADDR_WIDTH    ( IMEM_ADDR_WIDTH       )
       
) SIMD_inst_decoder (

	.clk				(	clk          	    ),
	.reset              (	reset        	    ),

	.start              (	start        	    ),
	
	.group_id_s         (   group_id_s          ),
	
	.in_ld_st           ( in_ld_st              ),

	.instruction_in     (	imem_rd_data	    ),
	.instruction_in_v   (	1'b1         	    ),
	
	.group_buf_rd_data  (   group_mem_rd_data   ),
	.group_buf_rd_v     (   group_mem_rd_v      ),
	.group_buf_rd_req   (   group_mem_rd_req    ),
	.group_buf_rd_addr  (   group_mem_rd_addr   ),
	
	.ld_mem_simd_done   (   ld_mem_simd_done    ),
	.st_mem_simd_done   (   st_mem_simd_done    ),
	.ld_config_done     (   ld_config_done      ),
	.st_config_done     (   st_config_done      ),
	.ld_st_group_id     (   ld_st_group_id      ),
	
	.stall              (   stall               ),
	
	.ready              (   ready               ),
	

	.imem_rd_address    (	imem_rd_address	    ),
	.imem_rd_req        (	imem_rd_req  	    ),

	.opcode             (	opcode           	),
	.fn                 (	fn               	),

	.dest_ns_id         (	dest_ns_id      	),
	.dest_ns_index_id   (	dest_ns_index_id	),

	.src1_ns_id         (	src1_ns_id      	),
	.src1_ns_index_id   (	src1_ns_index_id	),

	.src2_ns_id         (	src2_ns_id      	),
	.src2_ns_index_id   (	src2_ns_index_id	),
	
	.integer_bits       (   integer_bits_to_iter_mem),
	
	.nested_loop_done   (   nested_loop_done    ),
	
	
	.in_a_loop          (   in_loop             ),
	.done               (   done                ),
	.group_id           (   group_id            )

);

// ************ SIMD LD/ST Interface *******************

simd_ld_st_interface #(
    .NUM_TAGS                           ( NUM_TAGS              ),
    .SIMD_DATA_WIDTH                    ( DATA_WIDTH            ),
    .LD_ST_LOW_DATA_WIDTH               ( LD_ST_LOW_DATA_WIDTH  ),
    .AXI_ADDR_WIDTH                     ( AXI_ADDR_WIDTH        ),
    .AXI_DATA_WIDTH                     ( AXI_DATA_WIDTH        ),
    .AXI_BURST_WIDTH                    ( AXI_BURST_WIDTH       ),
    .WSTRB_W                            ( WSTRB_W               ),
    .NUM_SIMD_LANES                     ( NUM_ELEM              ),
    .VMEM_BUF_ADDR_W                    ( VMEM_ADDR_WIDTH       ),
    .BASE_ADDR_SEGMENT_W                ( BASE_ADDR_SEGMENT_W   ),
    .ADDR_WIDTH                         ( AXI_ADDR_WIDTH        ),
    .GROUP_ENABLED                      ( GROUP_ENABLED         )
) ld_st_interface_inst (
    .clk                                ( clk                   ),
    .reset                              ( reset                 ),    
    .block_done                         ( block_done            ),
    
    .opcode                             ( opcode                ),
    .fn                                 ( fn                    ),
    .dest_ns_id                         ( dest_ns_id            ),
    .dest_ns_index_id                   ( dest_ns_index_id      ),
    .src1_ns_id                         ( src1_ns_id            ),
    .src1_ns_index_id                   ( src1_ns_index_id      ),
    .src2_ns_id                         ( src2_ns_id            ),
    .src2_ns_index_id                   ( src2_ns_index_id      ),
    
    .ld_config_done                     ( ld_config_done        ),
    .st_config_done                     ( st_config_done        ),
    .ld_st_group_id                     ( ld_st_group_id        ),
    .ld_mem_simd_done                   ( ld_mem_simd_done      ),
    .st_mem_simd_done                   ( st_mem_simd_done      ),
    
    .vmem1_write_req                    ( ld_st_vmem1_write_req ),
    .vmem1_write_addr                   ( ld_st_vmem1_write_addr),
    .vmem1_write_data                   ( ld_st_vmem1_write_data),
    .vmem1_read_req                     ( ld_st_vmem1_read_req  ),
    .vmem1_read_addr                    ( ld_st_vmem1_read_addr ),
    .vmem1_read_data                    ( ld_st_vmem1_read_data ),
    
    .vmem2_write_req                    ( ld_st_vmem2_write_req ),
    .vmem2_write_addr                   ( ld_st_vmem2_write_addr),
    .vmem2_write_data                   ( ld_st_vmem2_write_data),
    .vmem2_read_req                     ( ld_st_vmem2_read_req  ),
    .vmem2_read_addr                    ( ld_st_vmem2_read_addr ),
    .vmem2_read_data                    ( ld_st_vmem2_read_data ),    
    
    .simd_tiles_done                    ( simd_tiles_done       ),
    
    .mws_awaddr                         ( mws_awaddr            ),
    .mws_awlen                          ( mws_awlen             ),
    .mws_awsize                         ( mws_awsize            ),
    .mws_awburst                        ( mws_awburst           ),
    .mws_awvalid                        ( mws_awvalid           ),
    .mws_awready                        ( mws_awready           ),
    .mws_wdata                          ( mws_wdata             ),
    .mws_wstrb                          ( mws_wstrb             ),
    .mws_wlast                          ( mws_wlast             ),
    .mws_wvalid                         ( mws_wvalid            ),
    .mws_wready                         ( mws_wready            ),
    .mws_bresp                          ( mws_bresp             ),
    .mws_bvalid                         ( mws_bvalid            ),
    .mws_bready                         ( mws_bready            ),
    .mws_araddr                         ( mws_araddr            ),
    .mws_arid                           ( mws_arid              ),
    .mws_arlen                          ( mws_arlen             ),
    .mws_arsize                         ( mws_arsize            ),
    .mws_arburst                        ( mws_arburst           ),
    .mws_arvalid                        ( mws_arvalid           ),
    .mws_arready                        ( mws_arready           ),
    .mws_rdata                          ( mws_rdata             ),
    .mws_rid                            ( mws_rid               ),
    .mws_rresp                          ( mws_rresp             ),
    .mws_rlast                          ( mws_rlast             ),
    .mws_rvalid                         ( mws_rvalid            ),
    .mws_rready                         ( mws_rready            )
    
);





iterator_address_gen #(
    .NS_ID_BITS         ( NS_ID_BITS            ), 
    .NS_INDEX_ID_BITS   ( NS_INDEX_ID_BITS      ),
    .OPCODE_BITS        ( OPCODE_BITS           ),
    .FUNCTION_BITS      ( FUNCTION_BITS         )
    
) iterator_address_gen_inst (

	.clk				(	clk          	    ),
	.reset              (	reset        	    ),

	.opcode             (	opcode           	),
	.fn                 (	fn               	),

	.dest_ns_id         (	dest_ns_id      	),
	.dest_ns_index_id   (	dest_ns_index_id	),

	.src1_ns_id         (	src1_ns_id      	),
	.src1_ns_index_id   (	src1_ns_index_id	),

	.src2_ns_id         (	src2_ns_id      	),
	.src2_ns_index_id   (	src2_ns_index_id	),
	
	.in_loop            (   in_loop             ),
	
	.iterator_stride_0		            (	iterator_stride_0		        ),
    .iterator_base_0		            (	buffer_address_0	            ),

    .iterator_stride_1		            (	iterator_stride_1		        ),
    .iterator_base_1		            (	buffer_address_1	            ),

    .iterator_stride_2		            (	iterator_stride_2		        ),
    .iterator_base_2		            (	buffer_address_2	            ),

    .iterator_stride_3		            (	iterator_stride_3		        ),
    .iterator_base_3		            (	buffer_address_3	            ),

    .iterator_stride_4		            (	iterator_stride_4		        ),
    .iterator_base_4		            (	buffer_address_4	            ),

    .iterator_stride_5		            (	iterator_stride_5		        ),
    .iterator_base_5		            (	buffer_address_5	            ),
	
    ////////////////////////////////// outputs  ////////////////////////    
    .iterator_read_req_out              (	iterator_read_req               ),
    .iterator_write_req_base_out        (	iterator_write_req_base         ),
    .iterator_write_req_stride_out      (	iterator_write_req_stride       ),

    .buffer_write_req                   (	buffer_write_req                ),
    .buffer_read_req                    (	buffer_read_req                 ),
    .mem_bypass                         (	mem_bypass                      ),


    .iterator_read_addr_out_0           (	iterator_read_addr_0            ),

    .iterator_write_addr_base_out_0     (	iterator_write_addr_base_0      ),
    .iterator_data_in_base_out_0        (	iterator_data_base_0         ),

    .iterator_write_addr_stride_out_0   (	iterator_write_addr_stride_0    ),
    .iterator_data_in_stride_out_0      (	iterator_data_stride_0       ),

    .base_plus_stride_out_0             (	base_plus_stride_0              ),


    .iterator_read_addr_out_1           (	iterator_read_addr_1            ),

    .iterator_write_addr_base_out_1     (	iterator_write_addr_base_1      ),
    .iterator_data_in_base_out_1        (	iterator_data_base_1         ),

    .iterator_write_addr_stride_out_1   (	iterator_write_addr_stride_1    ),
    .iterator_data_in_stride_out_1      (	iterator_data_stride_1       ),

    .base_plus_stride_out_1             (	base_plus_stride_1              ),


    .iterator_read_addr_out_2           (	iterator_read_addr_2            ),

    .iterator_write_addr_base_out_2     (	iterator_write_addr_base_2      ),
    .iterator_data_in_base_out_2        (	iterator_data_base_2         ),

    .iterator_write_addr_stride_out_2   (	iterator_write_addr_stride_2    ),
    .iterator_data_in_stride_out_2      (	iterator_data_stride_2       ),

    .base_plus_stride_out_2             (	base_plus_stride_2              ),


    .iterator_read_addr_out_3           (	iterator_read_addr_3            ),

    .iterator_write_addr_base_out_3     (	iterator_write_addr_base_3      ),
    .iterator_data_in_base_out_3        (	iterator_data_base_3         ),

    .iterator_write_addr_stride_out_3   (	iterator_write_addr_stride_3    ),
    .iterator_data_in_stride_out_3      (	iterator_data_stride_3       ),

    .base_plus_stride_out_3             (	base_plus_stride_3              ),


    .iterator_read_addr_out_4           (	iterator_read_addr_4            ),

    .iterator_write_addr_base_out_4     (	iterator_write_addr_base_4      ),
    .iterator_data_in_base_out_4        (	iterator_data_base_4         ),

    .iterator_write_addr_stride_out_4	(	iterator_write_addr_stride_4    ),
    .iterator_data_in_stride_out_4      (	iterator_data_stride_4       ),

    .base_plus_stride_out_4             (   base_plus_stride_4              ),


    .iterator_read_addr_out_5           (	iterator_read_addr_5            ),

    .iterator_write_addr_base_out_5     (	iterator_write_addr_base_5      ),
    .iterator_data_in_base_out_5        (	iterator_data_base_5         ),

    .iterator_write_addr_stride_out_5	(	iterator_write_addr_stride_5    ),
    .iterator_data_in_stride_out_5      (	iterator_data_stride_5       ),

    .base_plus_stride_out_5             (   base_plus_stride_5              ),
	
    .immediate_out                      (   immediate_to_iter_mem           )

);

assign opcode_fn = {opcode,fn};

pipeline #( .NUM_BITS	( FUNCTION_BITS + OPCODE_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) opcode_fn_delay1 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	opcode_fn	), .data_out	(	opcode_fn_to_iter_mem    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) dest_ns_id_delay1 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	dest_ns_id	), .data_out	(	dest_ns_id_to_iter_mem    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src1_sel_delay1 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src1_ns_id	), .data_out	(	src1_ns_id_to_iter_mem    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src2_sel_delay1 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src2_ns_id	), .data_out	(	src2_ns_id_to_iter_mem    ) );

//pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) dest_ns_index_id_delay (
//    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	dest_ns_index_id	), .data_out	(	dest_ns_index_id_to_iter_mem    ) );

//pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src1_index_id_delay (
//    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src1_ns_index_id	), .data_out	(	src1_ns_index_id_to_iter_mem    ) );

//pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src2_index_id_delay (
//    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src2_ns_index_id	), .data_out	(	src2_ns_index_id_to_iter_mem    ) );       
///////////////////////////////////////// Stage 2 ///////////////////////////
///////////////////////////////////////// Iterator memories ///////////////////////////

iterator_memories #(
    .NS_ID_BITS         ( NS_ID_BITS            ), 
    .NS_INDEX_ID_BITS   ( NS_INDEX_ID_BITS      ),
    .OPCODE_BITS        ( OPCODE_BITS           ),
    .FUNCTION_BITS      ( FUNCTION_BITS         ),
    .IMMEDIATE_WIDTH    ( IMMEDIATE_WIDTH       )
    
) iterator_memory_inst (

	.clk				(	clk          	    ),
	.reset              (	reset        	    ),

    .opcode             (   opcode_fn_to_iter_mem[FUNCTION_BITS+:OPCODE_BITS]  ),
    .fn                 (   opcode_fn_to_iter_mem[FUNCTION_BITS-1 : 0]  ),
    
    .loop_id            (   dest_ns_id_to_iter_mem                  ),
    .immediate          (   immediate_to_iter_mem                   ),
    .iterator_read_req					(	iterator_read_req				),
    .iterator_write_req_base            (	iterator_write_req_base         ),
    .iterator_write_req_stride          (	iterator_write_req_stride       ),
    .mem_bypass                         (	mem_bypass                      ),
    
    .iterator_read_addr_in_0            (	iterator_read_addr_0            ),
    .iterator_write_addr_base_in_0      (	iterator_write_addr_base_0      ),
    .iterator_data_in_base_in_0         (	iterator_data_base_0            ),
    .iterator_write_addr_stride_in_0    (	iterator_write_addr_stride_0    ),
    .iterator_data_in_stride_in_0       (	iterator_data_stride_0          ),
    .base_plus_stride_in_0              (	base_plus_stride_0              ),
    
    .iterator_read_addr_in_1            (	iterator_read_addr_1            ),
    .iterator_write_addr_base_in_1      (	iterator_write_addr_base_1      ),
    .iterator_data_in_base_in_1         (	iterator_data_base_1            ),
    .iterator_write_addr_stride_in_1    (	iterator_write_addr_stride_1    ),
    .iterator_data_in_stride_in_1       (	iterator_data_stride_1          ),
    .base_plus_stride_in_1              (	base_plus_stride_1              ),
    
    .iterator_read_addr_in_2            (	iterator_read_addr_2            ),
    .iterator_write_addr_base_in_2      (	iterator_write_addr_base_2      ),
    .iterator_data_in_base_in_2         (	iterator_data_base_2            ),
    .iterator_write_addr_stride_in_2    (	iterator_write_addr_stride_2    ),
    .iterator_data_in_stride_in_2       (	iterator_data_stride_2          ),
    .base_plus_stride_in_2              (	base_plus_stride_2              ),
    
    .iterator_read_addr_in_3            (	iterator_read_addr_3            ),
    .iterator_write_addr_base_in_3      (	iterator_write_addr_base_3      ),
    .iterator_data_in_base_in_3         (	iterator_data_base_3            ),
    .iterator_write_addr_stride_in_3    (	iterator_write_addr_stride_3    ),
    .iterator_data_in_stride_in_3       (	iterator_data_stride_3          ),
    .base_plus_stride_in_3              (	base_plus_stride_3              ),
    
    .iterator_read_addr_in_4            (	iterator_read_addr_4            ),
    .iterator_write_addr_base_in_4      (	iterator_write_addr_base_4      ),
    .iterator_data_in_base_in_4         (	iterator_data_base_4            ),
    .iterator_write_addr_stride_in_4    (	iterator_write_addr_stride_4    ),
    .iterator_data_in_stride_in_4       (	iterator_data_stride_4          ),
    .base_plus_stride_in_4              (	base_plus_stride_4              ),
     
    .iterator_read_addr_in_5            (	iterator_read_addr_5            ),
    .iterator_write_addr_base_in_5      (	iterator_write_addr_base_5      ),
    .iterator_data_in_base_in_5         (	iterator_data_base_5            ),
    .iterator_write_addr_stride_in_5    (	iterator_write_addr_stride_5    ),
    .iterator_data_in_stride_in_5       (	iterator_data_stride_5          ),
    .base_plus_stride_in_5              (	base_plus_stride_5              ),       
    
    /////////////////////////////outputs
    

    .iterator_stride_0                  (	iterator_stride_0               ),
    .buffer_address_0                   (	buffer_address_0                ),
    .iterator_stride_1                  (	iterator_stride_1               ),
    .buffer_address_1                   (	buffer_address_1                ),
    .iterator_stride_2                  (	iterator_stride_2               ),
    .buffer_address_2                   (	buffer_address_2                ),
    .iterator_stride_3                  (	iterator_stride_3               ),
    .buffer_address_3                   (	buffer_address_3                ),
    .iterator_stride_4                  (	iterator_stride_4               ),
    .buffer_address_4                   (	buffer_address_4                ),
	.iterator_stride_5                  (	iterator_stride_5               ),
    .buffer_address_5                   (	buffer_address_5                ),
    .loop_done_out                      (   nested_loop_done                )
);


pipeline #( .NUM_BITS	( FUNCTION_BITS + OPCODE_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) opcode_fn_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	opcode_fn_to_iter_mem	), .data_out	(	opcode_fn_to_mem_stage    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) dest_ns_id_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	dest_ns_id_to_iter_mem	), .data_out	(	dest_ns_id_to_mem_stage    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src1_ns_id_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src1_ns_id_to_iter_mem	), .data_out	(	src1_ns_id_to_mem_stage    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src2_ns_id_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src2_ns_id_to_iter_mem	), .data_out	(	src2_ns_id_to_mem_stage    ) );

pipeline #( .NUM_BITS	( IMMEDIATE_WIDTH	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) immediate_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	immediate_to_iter_mem	), .data_out	(	immediate_to_mem_stage    ) );

pipeline #( .NUM_BITS	( 6	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) buffer_wr_req_delay (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffer_write_req	), .data_out	(	buffer_write_req_to_mem_stage    ) );

pipeline #( .NUM_BITS	( 6	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) buffer_rd_req_delay (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffer_read_req	), .data_out	(	buffer_read_req_to_mem_stage    ) );

pipeline #( .NUM_BITS	( 16	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) integer_bits_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	integer_bits_to_iter_mem	), .data_out	(	integer_bits_to_mem_stage    ) );

    
///////////////////////////////////////// Stage 3 ///////////////////////////
///////////////////////////////////////// memories ///////////////////////////

assign buffers_rd_addr = {buffer_address_5,buffer_address_4,buffer_address_3,buffer_address_2,buffer_address_1,buffer_address_0};

generate 
for(genvar i = 0 ; i< NUM_ELEM ; i=i+1) begin : SIMD_RD

    reg [5:0] buf_rd_req;
    reg [BASE_STRIDE_WIDTH*6-1:0] buf_rd_addr;
    reg [IMM_DATA_WIDTH-1:0] imm_data_w;
    if(INTERLEAVE == 0) begin
        always @(*) begin

            buf_rd_req = buffer_read_req_to_mem_stage;
            buf_rd_addr = buffers_rd_addr;
            imm_data_w = imm_mem_out;
        end
    end
    else begin
        if( i == 0) begin
            always @(*) begin
                buf_rd_req = buffer_read_req_to_mem_stage;
                buf_rd_addr = buffers_rd_addr;
                imm_data_w = imm_mem_out;
            end
        end
        else begin
            always @(posedge clk) begin
                buf_rd_req <= SIMD_RD[i-1].buf_rd_req;
                buf_rd_addr <= SIMD_RD[i-1].buf_rd_addr;
                imm_data_w <= SIMD_RD[i-1].imm_data_w;
            end
        end
    end
    assign buf_rd_addr_interleaved[i] = buf_rd_addr;
    assign buf_rd_req_interleaved[i] = buf_rd_req;
    assign imm_data_interleaved[i] = imm_data_w;
    
end
endgenerate

generate
    for(genvar i = 0; i< NUM_ELEM; i= i+1) begin
        assign obuf_rd_addr[OBUF_ADDR_WIDTH*i+: OBUF_ADDR_WIDTH] = buf_rd_addr_interleaved[i][BASE_STRIDE_WIDTH*0+: OBUF_ADDR_WIDTH];
        assign vmem_rd_addr1[VMEM_ADDR_WIDTH*i+: VMEM_ADDR_WIDTH] = buf_rd_addr_interleaved[i][BASE_STRIDE_WIDTH*2+: VMEM_ADDR_WIDTH];
        assign vmem_rd_addr2[VMEM_ADDR_WIDTH*i+: VMEM_ADDR_WIDTH] = buf_rd_addr_interleaved[i][BASE_STRIDE_WIDTH*5+: VMEM_ADDR_WIDTH];
        //assign imm_rd_addr[IMM_ADDR_WIDTH*i+: IMM_ADDR_WIDTH] = buf_rd_addr_interleaved[i][BASE_STRIDE_WIDTH*3+: IMM_ADDR_WIDTH];
        assign imm_data[IMM_DATA_WIDTH*i+: IMM_DATA_WIDTH] = imm_data_interleaved[i];
        assign ext_mem_rd_addr[EXT_MEM_ADDR_WIDTH*i+: EXT_MEM_ADDR_WIDTH] = buf_rd_addr_interleaved[i][BASE_STRIDE_WIDTH*4 +: EXT_MEM_ADDR_WIDTH];
        
        assign obuf_rd_req[i] = buf_rd_req_interleaved[i][0];
        assign vmem_rd_req1[i] = buf_rd_req_interleaved[i][2];
        assign vmem_rd_req2[i] = buf_rd_req_interleaved[i][5];
        //assign imm_rd_req[i] = buf_rd_req_interleaved[i][3];
        assign ext_mem_rd_req[i] = buf_rd_req_interleaved[i][4];
    end
endgenerate

always @(*) begin
    case(dest_ns_id_to_mem_stage)
        3'b000: buffers_wr_addr = buffer_address_0;
        3'b001: buffers_wr_addr = buffer_address_1;
        3'b010: buffers_wr_addr = buffer_address_2;
        3'b011: buffers_wr_addr = buffer_address_3;
        3'b100: buffers_wr_addr = buffer_address_4;
        default: buffers_wr_addr = buffer_address_5;
    endcase
end




assign _vmem_rd_req1 = in_ld_st ? ld_st_vmem1_read_req : vmem_rd_req1;
assign _vmem_rd_addr1 =  in_ld_st ? ld_st_vmem1_read_addr : vmem_rd_addr1;
assign _vmem_data1 = in_ld_st ? ld_st_vmem1_read_data : vmem_data1;
assign _vmem_wr_req1 = in_ld_st ? ld_st_vmem1_write_req : vmem_wr_req1;
assign _vmem_wr_addr1 = in_ld_st ? ld_st_vmem1_write_addr : vmem_wr_addr1;
assign _vmem_wr_data1 = in_ld_st ? ld_st_vmem1_write_data : vmem_wr_data1;

assign _vmem_rd_req2 = in_ld_st ? ld_st_vmem2_read_req : vmem_rd_req2;
assign _vmem_rd_addr2 =  in_ld_st ? ld_st_vmem2_read_addr : vmem_rd_addr2;
assign _vmem_data2 = in_ld_st ? ld_st_vmem2_read_data : vmem_data2;
assign _vmem_wr_req2 = in_ld_st ? ld_st_vmem2_write_req : vmem_wr_req2;
assign _vmem_wr_addr2 = in_ld_st ? ld_st_vmem2_write_addr : vmem_wr_addr2;
assign _vmem_wr_data2 = in_ld_st ? ld_st_vmem2_write_data : vmem_wr_data2;

    
vector_memory #(
    .NUM_ELEM           ( NUM_ELEM              ), 
    .ADDR_WIDTH         ( VMEM_ADDR_WIDTH       ),
    .DATA_WIDTH         ( VMEM_DATA_WIDTH       )
    
) vector_mem_inst (

	.clk				(	clk          	    ),
	.reset              (	reset        	    ),
	
	.read_req           (   _vmem_rd_req1         ),
    .read_addr          (	_vmem_rd_addr1        ),
    .read_data          (	_vmem_data1           ),

    .write_req          (	_vmem_wr_req1         ),
    .write_addr         (	_vmem_wr_addr1        ),
    .write_data         (	_vmem_wr_data1        )
	
);

vector_memory #(
    .NUM_ELEM           ( NUM_ELEM              ), 
    .ADDR_WIDTH         ( VMEM_ADDR_WIDTH       ),
    .DATA_WIDTH         ( VMEM_DATA_WIDTH       )
    
) vector_mem_inst2 (

	.clk				(	clk          	    ),
	.reset              (	reset        	    ),
	
	.read_req           (   _vmem_rd_req2         ),
    .read_addr          (	_vmem_rd_addr2        ),
    .read_data          (	_vmem_data2           ),

    .write_req          (	_vmem_wr_req2         ),
    .write_addr         (	_vmem_wr_addr2        ),
    .write_data         (	_vmem_wr_data2        )
	
);

assign imm_wr_data = immediate_to_mem_stage;
assign imm_wr_addr = buffers_wr_addr[IMM_ADDR_WIDTH-1:0];
assign imm_wr_req = buffer_write_req_to_mem_stage[3];

ram
#(
  .DATA_WIDTH           (IMM_DATA_WIDTH ),
  .ADDR_WIDTH           (IMM_ADDR_WIDTH )
) immediate_memory
(
  .clk		            (   clk                 ),
  .reset                (	reset               ),

  .read_req             (   buffer_read_req_to_mem_stage[3]	        ),
  .read_addr            (	buffer_address_3[IMM_ADDR_WIDTH-1:0]         ),
  .read_data            (	imm_mem_out            ),

  .write_req            (	imm_wr_req          ),
  .write_addr           (	imm_wr_addr         ),
  .write_data           (	imm_wr_data         )
);

pipeline #( .NUM_BITS	( BUFFERS_WR_ADDR_WIDTH	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) buffers_wr_address_delay (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffers_wr_addr	), .data_out	(	buffers_wr_addr_to_mux_stage    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src1_sel_delay (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src1_ns_id_to_mem_stage	), .data_out	(	src1_ns_id_to_mux_stage    ) );

pipeline #( .NUM_BITS	( NS_ID_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) src2_sel_delay (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	src2_ns_id_to_mem_stage	), .data_out	(	src2_ns_id_to_mux_stage    ) );

pipeline #( .NUM_BITS	( 6	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) buffer_wr_req_delay2 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffer_write_req_to_mem_stage	), .data_out	(	buffer_write_req_to_mux_stage    ) );

pipeline #( .NUM_BITS	( FUNCTION_BITS + OPCODE_BITS	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) opcode_fn_delay3 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	opcode_fn_to_mem_stage	), .data_out	(	opcode_fn_to_mux_stage    ) );

pipeline #( .NUM_BITS	( 16	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) integer_bits_delay3 (
    .clk		(	clk		    ), .rst		(	reset		), .data_in	(	integer_bits_to_mem_stage	), .data_out	(	integer_bits_to_mux_stage    ) );
///////////////////////////////////////// Stage 4 ///////////////////////////
///////////////////////////////////////// namespace mux ///////////////////////////

generate 
for(genvar i = 0 ; i< NUM_ELEM ; i=i+1) begin : MUX
    
    if(INTERLEAVE == 0) begin
        always @(*) begin
            src1_ns_id_interleaved[i] = src1_ns_id_to_mux_stage;
            src2_ns_id_interleaved[i] = src2_ns_id_to_mux_stage;
       end

    end
    else begin
        if( i == 0) begin
            always @(*) begin
                src1_ns_id_interleaved[i] = src1_ns_id_to_mux_stage;
                src2_ns_id_interleaved[i] = src2_ns_id_to_mux_stage;
           end
        end
        else begin
            always @(posedge clk) begin
                src1_ns_id_interleaved[i] <= src1_ns_id_interleaved[i-1];
                src2_ns_id_interleaved[i] <= src2_ns_id_interleaved[i-1];
            end
        end
    end
    
    namespace_mux
    #(
      .DATA_WIDTH       ( DATA_WIDTH    )
    ) src1_mux
    (   
        .obuf_data      (   obuf_data[i*DATA_WIDTH+:DATA_WIDTH]     ),
        .ibuf_data      (   'd0                                     ),
        .vmem_data1     (   vmem_data1[i*DATA_WIDTH+:DATA_WIDTH]    ),
        .vmem_data2     (   vmem_data2[i*DATA_WIDTH+:DATA_WIDTH]    ),
        .imm_data       (   imm_data[i*DATA_WIDTH+:DATA_WIDTH]      ),
        .ext_data       (   ext_mem_data[i*DATA_WIDTH+:DATA_WIDTH]  ),
        
        .data_sel       (   src1_ns_id_interleaved[i]               ),
        
        .data_out       (   src1_muxed[i*DATA_WIDTH+:DATA_WIDTH]    )
    );
        
    namespace_mux
    #(
      .DATA_WIDTH       ( DATA_WIDTH    )
    ) src2_mux
    (   
        .obuf_data      (   obuf_data[i*DATA_WIDTH+:DATA_WIDTH]     ),
        .ibuf_data      (   'd0                                     ),
        .vmem_data1     (   vmem_data1[i*DATA_WIDTH+:DATA_WIDTH]    ),
        .vmem_data2     (   vmem_data2[i*DATA_WIDTH+:DATA_WIDTH]    ),
        .imm_data       (   imm_data[i*DATA_WIDTH+:DATA_WIDTH]      ),
        .ext_data       (   ext_mem_data[i*DATA_WIDTH+:DATA_WIDTH]  ),
        
        .data_sel       (   src2_ns_id_interleaved[i]               ),
        
        .data_out       (   src2_muxed[i*DATA_WIDTH+:DATA_WIDTH]    )
    );
   
end
endgenerate   


generate 
for(genvar i = 0 ; i< NUM_ELEM ; i=i+1) begin : mux_out

    if(ENABLE_PIPELINE_AFTER_NAMESPACE_MUX == 1) begin
        always @(posedge clk) begin
            src1_data[i] <= src1_muxed[DATA_WIDTH*i +: DATA_WIDTH];
            src2_data[i] <= src2_muxed[DATA_WIDTH*i +: DATA_WIDTH];
        end
    end
    else begin
        always @(*) begin
            src1_data[i] = src1_muxed[DATA_WIDTH*i +: DATA_WIDTH];
            src2_data[i] = src2_muxed[DATA_WIDTH*i +: DATA_WIDTH];
        end
    end
end
endgenerate

localparam extra_stages = 4;

if(ENABLE_PIPELINE_AFTER_NAMESPACE_MUX == 1) begin
    
    pipeline #( .NUM_BITS	( BUFFERS_WR_ADDR_WIDTH	), .NUM_STAGES	( extra_stages	), .EN_RESET   ( 0 ) ) buffers_wr_address_delay_ns (
        .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffers_wr_addr_to_mux_stage	), .data_out	(	buffers_wr_addr_to_compute_stage    ) );

    pipeline #( .NUM_BITS	( 6	), .NUM_STAGES	( extra_stages	), .EN_RESET   ( 0 ) ) buffer_wr_req_delay_ns (
        .clk		(	clk		    ), .rst		(	reset		), .data_in	(	buffer_write_req_to_mux_stage	), .data_out	(	buffer_write_req_to_compute_stage    ) );
    
    pipeline #( .NUM_BITS	( FUNCTION_BITS + OPCODE_BITS	), .NUM_STAGES	( extra_stages	), .EN_RESET   ( 0 ) ) opcode_fn_delay4 (
        .clk		(	clk		    ), .rst		(	reset		), .data_in	(	opcode_fn_to_mux_stage	), .data_out	(	opcode_fn_to_compute_stage    ) );

    pipeline #( .NUM_BITS	( 16	), .NUM_STAGES	( 1	), .EN_RESET   ( 0 ) ) integer_bits_delay4 (
        .clk		(	clk		    ), .rst		(	reset		), .data_in	(	integer_bits_to_mux_stage	), .data_out	(	integer_bits_to_compute_stage    ) );

end
else begin
    assign buffer_write_req_to_compute_stage = buffer_write_req_to_mux_stage;
    assign opcode_fn_to_compute_stage = opcode_fn_to_mux_stage;
    assign buffers_wr_addr_to_compute_stage = buffers_wr_addr_to_mux_stage;
end
    
    
///////////////////////////////////////// Stage 5 ///////////////////////////
///////////////////////////////////////// compute ///////////////////////////
assign fn_compute = opcode_fn_to_compute_stage[FUNCTION_BITS-1:0];
assign opcode_compute = opcode_fn_to_compute_stage[OPCODE_BITS+FUNCTION_BITS-1:FUNCTION_BITS];

/// CONDITIONAL MOVE

assign cond_move_inst = (fn_compute[FUNCTION_BITS-1:1] == 3'b101) && (opcode_compute == 4'b0000); 

generate 
for(genvar i = 0 ; i< NUM_ELEM ; i=i+1) begin : SIMD

    assign compute_out_bus[DATA_WIDTH*i +: DATA_WIDTH] = data_compute_out[i];
    
    reg [FUNCTION_BITS-1:0] fn_in;
    reg [OPCODE_BITS-1:0] opcode_in;
    reg [5:0] buf_wr_req;
    reg [BASE_STRIDE_WIDTH-1:0] buf_wr_addr;
    reg cond_move;
    reg disable_write;
    
    if(INTERLEAVE == 0) begin
        always @(*) begin
            fn_in = fn_compute;
            opcode_in = opcode_compute;
            cond_move = cond_move_inst;
            disable_write = cond_move && (fn_in[0] ^ src1_data[i][0]);
       end
       always @(posedge clk) begin
            buf_wr_req <= buffer_write_req_to_compute_stage &{6{~disable_write}};
            buf_wr_addr <= buffers_wr_addr_to_compute_stage;
        end

    end
    else begin
        if( i == 0) begin
            always @(*) begin
                fn_in = fn_compute;
                opcode_in = opcode_compute;
                cond_move = cond_move_inst;
                disable_write = cond_move && (fn_in[0] ^ src1_data[i][0]);
           end
           always @(posedge clk) begin
                buf_wr_req <= buffer_write_req_to_compute_stage &{6{~disable_write}};
                buf_wr_addr <= buffers_wr_addr_to_compute_stage;
            end
        end
        else begin
            always @(*) begin
                disable_write = cond_move && (fn_in[0] ^ src1_data[i][0]);
            end
            always @(posedge clk) begin
                fn_in <= SIMD[i-1].fn_in;
                opcode_in <= SIMD[i-1].opcode_in;
                cond_move <= SIMD[i-1].cond_move;

                buf_wr_req <= SIMD[i-1].buf_wr_req &{6{~disable_write}};
                buf_wr_addr <= SIMD[i-1].buf_wr_addr;
                
            end
        end
    end
    assign buf_wr_addr_interleaved[i] = buf_wr_addr;
    assign buf_wr_req_interleaved[i] = buf_wr_req;
    
    compute_unit 
    #(
        .OPCODE_BITS(OPCODE_BITS),
        .FUNCTION_BITS(FUNCTION_BITS),
        .DATA_WIDTH(DATA_WIDTH)
    ) compute_inst(
    .clk            (   clk       ),
    .reset          (   reset     ),

    .data_in0       (   src1_data[i]  ),
    .data_in1       (   src2_data[i]  ),
    
    .integer_bits   (   integer_bits_to_compute_stage    ),
    
    .data_out       (   data_compute_out[i]   ),

    .opcode         (   opcode_in     ),
    .fn             (   fn_in         )
    );
    
end
endgenerate
    

assign ibuf_wr_data = compute_out_bus;
assign vmem_wr_data1 = compute_out_bus;
assign vmem_wr_data2 = compute_out_bus;
assign ext_mem_wr_data = compute_out_bus;


generate
    for(genvar i = 0; i< NUM_ELEM; i= i+1) begin
        assign ibuf_wr_addr[IBUF_ADDR_WIDTH*i+: IBUF_ADDR_WIDTH] = buf_wr_addr_interleaved[i][IBUF_ADDR_WIDTH-1:0];
        assign vmem_wr_addr1[VMEM_ADDR_WIDTH*i+: VMEM_ADDR_WIDTH] = buf_wr_addr_interleaved[i][VMEM_ADDR_WIDTH-1:0];
        assign vmem_wr_addr2[VMEM_ADDR_WIDTH*i+: VMEM_ADDR_WIDTH] = buf_wr_addr_interleaved[i][VMEM_ADDR_WIDTH-1:0];
        assign ext_mem_wr_addr[EXT_MEM_ADDR_WIDTH*i+: EXT_MEM_ADDR_WIDTH] = buf_wr_addr_interleaved[i][EXT_MEM_ADDR_WIDTH-1:0];
        
        assign ibuf_wr_req[i] = buf_wr_req_interleaved[i][1];
        assign vmem_wr_req1[i] = buf_wr_req_interleaved[i][2];
        assign vmem_wr_req2[i] = buf_wr_req_interleaved[i][5];
        assign ext_mem_wr_req[i] = buf_wr_req_interleaved[i][4];
    end
endgenerate


endmodule
