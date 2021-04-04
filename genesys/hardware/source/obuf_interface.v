//
// Interface for Output Buffer
//
// Soroush Ghodrati

`timescale 1ns/1ps
module obuf_interface #(
  // Internal Parameters
    parameter integer  MEM_ID                       = 2,
    parameter integer  NUM_BASE_LOOPS               = 7,
    parameter integer  STORE_ENABLED                = 1,
    parameter integer  MEM_REQ_W                    = 16,
    parameter integer  ADDR_WIDTH                   = 8,
    parameter integer  DATA_WIDTH                   = 32,
    parameter integer  LOOP_ITER_W                  = 16,
    parameter integer  ADDR_STRIDE_W                = 32,
    parameter integer  LOOP_ID_W                    = 6,
    parameter integer  BUF_TYPE_W                   = 2,
    parameter integer  NUM_BUF_TYPE                 = 2**(BUF_TYPE_W),
    parameter integer  NUM_TAGS                     = 2,
    parameter integer  TAG_W                        = $clog2(NUM_TAGS),
    parameter integer  TAG_REUSE_COUNTER_W          = 3,
    parameter integer  INST_GROUP_ID_W              = 4,

  // AXI
    parameter integer  AXI_ID_WIDTH                 = 1,
    parameter integer  AXI_ADDR_WIDTH               = 42,
    parameter integer  AXI_DATA_WIDTH               = 64,
    parameter integer  AXI_BURST_WIDTH              = 8,
    parameter integer  WSTRB_W                      = AXI_DATA_WIDTH/8,

  // Buffer
    parameter integer  ARRAY_M                      = 32,
    parameter integer  ARRAY_N                      = 32,
    parameter integer  BUF_ADDR_W                   = 16,
    parameter integer  TAG_BUF_ADDR_W               = BUF_ADDR_W + TAG_W,
    parameter integer  BUF_WRITE_GROUP_SIZE_EXT     = AXI_DATA_WIDTH / DATA_WIDTH,
    parameter integer  BUF_WRITE_NUM_GROUP_EXT      = ARRAY_M / BUF_WRITE_GROUP_SIZE_EXT,
    parameter integer  COUNTER_BUF_WRITE_GROUP_W    = $clog2(BUF_WRITE_NUM_GROUP_EXT),
    parameter integer  BUF_READ_GROUP_SIZE_EXT      = BUF_WRITE_GROUP_SIZE_EXT,
    parameter integer  BUF_READ_NUM_GROUP_EXT       = BUF_WRITE_NUM_GROUP_EXT,
    parameter integer  COUNTER_BUF_READ_GROUP_W     = COUNTER_BUF_WRITE_GROUP_W,
    parameter integer  WAIT_CYCLES_COMPUTE_ST       = ARRAY_N,
    parameter integer  WAIT_CYCLES_COMPUTE_ST_W     = $clog2(ARRAY_N),
    parameter integer  GROUP_ENABLED                = 0
) (
    input  wire                                         clk,
    input  wire                                         reset,

    input  wire                                         tag_req,
    input  wire                                         tag_reuse,
    input  wire                                         tag_bias_prev_sw,
// This signal is used to determine if the current tag needs to be stored on SIMD or Offchip
// I think this signall needs to come with tag_base_addr
// This signal needs to be activated for all the intermediate fused layers that need to go to SIMD
    input  wire                                         tag_ddr_pe_sw,
    output wire                                         tag_ready,
    output wire                                         tag_done,
    input  wire                                         compute_done,
    input  wire                                         block_done,
    input  wire  [ ADDR_WIDTH           -1 : 0 ]        tag_base_ld_addr,
    input  wire  [ ADDR_WIDTH           -1 : 0 ]        tag_base_st_addr,
    
    input  wire                                         base_ld_addr_v,
    input  wire                                         base_st_addr_v,
    
    input  wire                                         compute_start,

    output wire                                         compute_ready,
    output wire                                         compute_bias_prev_sw,
    
    // TODO: controller (base_addr_gen)
    // This signal needs to be generated to show that if the next tile is a partial sum and if it needs to be loaded from ddr
    // If the ic_outer_loop is !=0, this is a partial sum and needs to be loaded from ddr
    // it may be similar to tag_ddr_pe_sw
    input  wire                                         first_ic_outer_loop_ld,
    // This signal is used to identify that there is a SIMD group after this systolic group
    input  wire                                         next_group_simd,

  // Programming
    input  wire                                         cfg_loop_stride_v,
    input  wire  [ 2                    -1 : 0 ]        cfg_loop_stride_type,
    input  wire  [ ADDR_STRIDE_W        -1 : 0 ]        cfg_loop_stride,
    input  wire  [ LOOP_ID_W            -1 : 0 ]        cfg_loop_stride_loop_id,
    input  wire  [ BUF_TYPE_W           -1 : 0 ]        cfg_loop_stride_id,
    input  wire                                         cfg_loop_stride_segment, 

    input  wire                                         cfg_loop_iter_v,
    input  wire  [ LOOP_ITER_W          -1 : 0 ]        cfg_loop_iter,
    input  wire  [ LOOP_ID_W            -1 : 0 ]        cfg_loop_iter_loop_id,
    input  wire  [ LOOP_ID_W            -1 : 0 ]        cfg_loop_iter_level,

    input  wire                                         cfg_mem_req_v,
    input  wire  [ BUF_TYPE_W           -1 : 0 ]        cfg_mem_req_id,
    input  wire  [ MEM_REQ_W            -1 : 0 ]        cfg_mem_req_size,
    input  wire  [ LOOP_ID_W            -1 : 0 ]        cfg_mem_req_loop_id,
    input  wire  [ 2                    -1 : 0 ]        cfg_mem_req_type,
    
   // The Group instructions for filling OBUF for fused conv/fc layers   
    input wire   [ INST_GROUP_ID_W       -1 : 0 ]       inst_group_id,
    input wire                                          inst_group_type,
    input wire                                          inst_group_s_e,
    input wire                                          inst_group_v,
    input wire   [ LOOP_ID_W             -1 : 0 ]       inst_group_sa_loop_id,
    input wire                                          inst_group_last,
    
    
  // Systolic Array to the Interface to handle data transfer
  // In the systolic array, the addresses of the obuf are not concatenated with the tag, but ibuf, wbuf, bbuf are concatenated.
  // TODO: Check the sys_obuf_write_req for systolic array and compute address generator
    input  wire  [ ARRAY_M              -1 : 0 ]        sys_buf_write_req_in,
    input  wire  [ ARRAY_M*BUF_ADDR_W   -1 : 0 ]        sys_buf_write_addr_in,
    input  wire  [ ARRAY_M*DATA_WIDTH   -1 : 0 ]        sys_buf_write_data_in,

    input  wire  [ ARRAY_M              -1 : 0 ]        sys_buf_read_req_in,
    input  wire  [ ARRAY_M*BUF_ADDR_W   -1 : 0 ]        sys_buf_read_addr_in,
    output wire  [ ARRAY_M*DATA_WIDTH   -1 : 0 ]        sys_buf_read_data_out,  
 
  // SIMD
    input  wire                                         simd_ready,
    output wire                                         simd_activate, 
    input  wire  [ ARRAY_M              -1 : 0 ]        simd_buf_read_req,
    input  wire  [ ARRAY_M*BUF_ADDR_W   -1 : 0 ]        simd_buf_read_addr,
    output wire  [ ARRAY_M*DATA_WIDTH   -1 : 0 ]        simd_buf_read_data,
    output wire  [ ARRAY_M              -1 : 0 ]        simd_data_valid,
      
  // BUF---Interface
    output wire  [ NUM_TAGS*ARRAY_M              -1 : 0 ]        ld_st_sys_buf_write_req_out     ,
    output wire  [ NUM_TAGS*ARRAY_M*BUF_ADDR_W   -1: 0 ]         ld_st_sys_buf_write_addr_out    ,
    output wire  [ NUM_TAGS*ARRAY_M*DATA_WIDTH   -1 : 0 ]        ld_st_sys_buf_write_data_out    ,
    output wire  [ NUM_TAGS*ARRAY_M              -1 : 0 ]        ld_st_sys_buf_read_req_out      ,
    output wire  [ NUM_TAGS*ARRAY_M*BUF_ADDR_W   -1: 0 ]         ld_st_sys_buf_read_addr_out     ,
    input  wire  [ NUM_TAGS*ARRAY_M*DATA_WIDTH   -1 : 0 ]        ld_st_sys_buf_read_data_in      ,

  // CL_wrapper -> DDR AXI4 interface
    // Master Interface Write Address
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

//==============================================================================
// Localparams
//==============================================================================
    localparam integer  LDMEM_IDLE                   = 0;
    localparam integer  LDMEM_CHECK_RAW              = 1;
    localparam integer  LDMEM_BUSY                   = 2;
    localparam integer  LDMEM_DUMMY                  = 3;
    localparam integer  LDMEM_WAIT_0                 = 4;
    localparam integer  LDMEM_WAIT_1                 = 5;
    localparam integer  LDMEM_WAIT_2                 = 6;
    localparam integer  LDMEM_WAIT_3                 = 7;
    localparam integer  LDMEM_DONE                   = 8;

    localparam integer  STMEM_IDLE                   = 0;
    localparam integer  STMEM_COMPUTE_WAIT           = 1;
    localparam integer  STMEM_DDR                    = 2;
    localparam integer  STMEM_SIMD                   = 3;
    localparam integer  STMEM_WAIT_0                 = 4;
    localparam integer  STMEM_WAIT_1                 = 5;
    localparam integer  STMEM_WAIT_2                 = 6;
    localparam integer  STMEM_WAIT_3                 = 7;
    localparam integer  STMEM_DONE                   = 8;


    localparam integer  MEM_LD                       = 0;
    localparam integer  MEM_ST                       = 1;
    localparam integer  MEM_RD                       = 2;
    localparam integer  MEM_WR                       = 3;

    localparam integer      SA_GROUP                 = 0;
    localparam integer      SIMD_GROUP               = 1;
    localparam integer      GROUP_START              = 0;
    localparam integer      GROUP_END                = 1;
//==============================================================================

//==============================================================================
// Wires/Regs
//==============================================================================
    wire                                        compute_tag_done;
    wire                                        compute_tag_reuse;
    wire                                        compute_tag_ready;
    wire [ TAG_W                -1 : 0 ]        compute_tag;
    wire [ TAG_W                -1 : 0 ]        compute_tag_delayed_buf_write;
    wire [ TAG_W                -1 : 0 ]        compute_tag_delayed_buf_read;
    wire                                        ldmem_tag_done;
    wire                                        ldmem_tag_ready;
    wire [ TAG_W                -1 : 0 ]        ldmem_tag;
    wire                                        stmem_tag_done;
    wire                                        stmem_tag_ready;

    reg  [ 4                    -1 : 0 ]        ldmem_state_d;
    reg  [ 4                    -1 : 0 ]        ldmem_state_q;
    reg  [ 4                    -1 : 0 ]        ldmem_state_qq;

    reg  [ 4                    -1 : 0 ]        stmem_state_d;
    reg  [ 4                    -1 : 0 ]        stmem_state_q;

    wire                                        ld_mem_req_v;
    wire                                        st_mem_req_v;

    wire [ TAG_W                -1 : 0 ]        tag;


    reg                                         ld_iter_v_q;
    reg                                         st_iter_v_q;
    reg  [ LOOP_ITER_W          -1 : 0 ]        iter_q;
    reg  [ LOOP_ID_W            -1 : 0 ]        loop_id_q;
    reg  [ LOOP_ID_W            -1 : 0 ]        loop_level_q;



    wire [ LOOP_ID_W            -1 : 0 ]        mws_ld_loop_iter_loop_id;
    wire [ LOOP_ITER_W          -1 : 0 ]        mws_ld_loop_iter;
    wire                                        mws_ld_loop_iter_v;
    wire  [ LOOP_ID_W            -1 : 0 ]       mws_ld_loop_iter_loop_level;
    wire                                        mws_ld_start;
    wire                                        mws_ld_done;
    wire                                        mws_ld_stall;
    wire                                        mws_ld_init;
    wire                                        mws_ld_enter;
    wire                                        mws_ld_exit;
    wire [ LOOP_ID_W            -1 : 0 ]        mws_ld_index;
    wire                                        mws_ld_index_valid;
    wire                                        mws_ld_step;
    wire [ LOOP_ID_W            -1 : 0 ]        mws_ld_loop_id;


    wire [ LOOP_ID_W            -1 : 0 ]        mws_st_loop_iter_loop_id;
    wire [ LOOP_ITER_W          -1 : 0 ]        mws_st_loop_iter;
    wire                                        mws_st_loop_iter_v;
    wire  [ LOOP_ID_W            -1 : 0 ]       mws_st_loop_iter_loop_level;
    wire                                        mws_st_start;
    wire                                        mws_st_done;
    wire                                        mws_st_stall;
    wire                                        mws_st_init;
    wire                                        mws_st_enter;
    wire                                        mws_st_exit;
    wire [ LOOP_ID_W            -1 : 0 ]        mws_st_index;
    wire                                        mws_st_index_valid;
    wire                                        mws_st_step;
    wire [ LOOP_ID_W            -1 : 0 ]        mws_st_loop_id;

    wire                                        ld_stride_v;
    wire [ ADDR_STRIDE_W        -1 : 0 ]        ld_stride;
    wire                                        ld_stride_segment;

    
    wire                                        st_stride_v;
    wire [ ADDR_STRIDE_W        -1 : 0 ]        st_stride;
    wire                                        st_stride_segment;

    wire [ ADDR_WIDTH           -1 : 0 ]        ld_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        mws_ld_base_addr;
    wire                                        ld_addr_v;
    wire [ ADDR_WIDTH           -1 : 0 ]        st_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        mws_st_base_addr;
    wire                                        st_addr_v;


    wire                                        st_buf_start;
    
    wire                                        next_compute_last_group;
    wire                                        next_st_last_group;

    reg  [ TAG_W                -1 : 0 ]        next_tag_compute_counter;
    reg  [ TAG_W                -1 : 0 ]        next_tag_st_counter;
    reg  [ INST_GROUP_ID_W      -1 : 0 ]        next_group_compute_counter;
    reg  [ INST_GROUP_ID_W      -1 : 0 ]        next_group_st_counter;



    reg  [ MEM_REQ_W            -1 : 0 ]        ld_req_size;
    reg  [ MEM_REQ_W            -1 : 0 ]        st_req_size;

    wire                                        ld_req_valid_d;
    wire                                        st_req_valid_d;

    reg                                         ld_req_valid_q;
    reg                                         st_req_valid_q;

    reg  [ ADDR_WIDTH           -1 : 0 ]        tag_ld_addr[0:NUM_TAGS-1];
    reg  [ ADDR_WIDTH           -1 : 0 ]        tag_st_addr[0:NUM_TAGS-1];

    reg  [ ADDR_WIDTH           -1 : 0 ]        ld_req_addr;
    reg  [ ADDR_WIDTH           -1 : 0 ]        st_req_addr;

    reg  [ MEM_REQ_W            -1 : 0 ]        st_req_loop_id;

    wire                                        axi_rd_req;
    wire [ AXI_ID_WIDTH         -1 : 0 ]        axi_rd_req_id;
    wire                                        axi_rd_done;
    wire [ MEM_REQ_W            -1 : 0 ]        axi_rd_req_size;
    wire                                        axi_rd_ready;
    wire [ AXI_ADDR_WIDTH       -1 : 0 ]        axi_rd_addr;

    wire                                        axi_wr_req;
    wire [ AXI_ID_WIDTH         -1 : 0 ]        axi_wr_req_id;
    wire                                        axi_wr_done;
    wire [ MEM_REQ_W            -1 : 0 ]        axi_wr_req_size;
    wire                                        axi_wr_ready;
    wire [ AXI_ADDR_WIDTH       -1 : 0 ]        axi_wr_addr;


    wire                                        mem_write_req;
    wire [ AXI_ID_WIDTH         -1 : 0 ]        mem_write_id;
    wire [ AXI_DATA_WIDTH       -1 : 0 ]        mem_write_data;
    wire                                        mem_write_ready;
    
    wire [ AXI_DATA_WIDTH       -1 : 0 ]        mem_read_data;
    wire                                        axi_mem_read_req;
    wire                                        axi_mem_read_ready;
    wire                                        mem_read_ready;
    
    reg  [ COUNTER_BUF_WRITE_GROUP_W     -1 : 0 ]   buf_write_ext_counter_group; 
  reg  [ BUF_ADDR_W             -1 : 0 ]          _buf_ext_write_addr;

  wire [ BUF_ADDR_W             -1 : 0 ]          buf_ext_write_addr;
//  wire [ TAG_BUF_ADDR_W         -1 : 0 ]          tag_buf_ext_write_addr;
  wire [ BUF_WRITE_GROUP_SIZE_EXT*BUF_ADDR_W-1 : 0]  group_buf_ext_write_addr;

  wire [ ARRAY_M*BUF_ADDR_W -1 : 0 ]              buf_ext_write_addr_out;
  wire [ ARRAY_M                -1 : 0 ]          buf_ext_write_req_out;
  wire [ ARRAY_M*DATA_WIDTH     -1 : 0 ]          buf_ext_write_data_out;

  wire [ BUF_WRITE_GROUP_SIZE_EXT   -1 : 0]       group_buf_ext_write_req;
  wire                                            buf_ext_write_req;



  reg  [ COUNTER_BUF_READ_GROUP_W     -1 : 0 ]   buf_read_ext_counter_group; 
  reg  [ BUF_ADDR_W             -1 : 0 ]          _buf_ext_read_addr;

  wire [ BUF_ADDR_W             -1 : 0 ]          buf_ext_read_addr;
//  wire [ TAG_BUF_ADDR_W         -1 : 0 ]          tag_buf_ext_read_addr;
  wire [ BUF_READ_GROUP_SIZE_EXT*BUF_ADDR_W-1 : 0]  group_buf_ext_read_addr;

  wire [ ARRAY_M*BUF_ADDR_W     -1 : 0 ]          buf_ext_read_addr_out;
  wire [ ARRAY_M                -1 : 0 ]          buf_ext_read_req_out;
  reg  [ ARRAY_M*DATA_WIDTH     -1 : 0 ]          _buf_ext_read_data;

  wire [ BUF_READ_GROUP_SIZE_EXT   -1 : 0]        group_buf_ext_read_req;
  wire                                            buf_ext_read_req;
  
  wire [ ARRAY_M                -1 : 0 ]          buf_simd_read_req_out;
  wire [ ARRAY_M*BUF_ADDR_W     -1 : 0 ]          buf_simd_read_addr_out;
  reg  [ARRAY_M*DATA_WIDTH      -1 : 0 ]          _simd_buf_read_data;
  
  wire [ ARRAY_M                -1 : 0 ]          _buf_simd_ext_read_req_out;
  wire [ ARRAY_M*BUF_ADDR_W     -1 : 0 ]          _buf_simd_ext_read_addr_out;
  
  wire  [ ARRAY_M               -1 : 0 ]          sys_buf_write_req_out;
  wire  [ ARRAY_M*BUF_ADDR_W    -1 : 0 ]          sys_buf_write_addr_out;
  wire  [ ARRAY_M*DATA_WIDTH    -1 : 0 ]          sys_buf_write_data_out;
  wire  [ ARRAY_M               -1 : 0 ]          sys_buf_read_req_out;
  wire  [ ARRAY_M*BUF_ADDR_W    -1 : 0 ]          sys_buf_read_addr_out;
  reg   [ ARRAY_M*DATA_WIDTH    -1 : 0 ]          _sys_buf_read_data_in;
  

  reg   [ ARRAY_M              -1 : 0 ]        _ld_st_sys_buf_write_req_out     [NUM_TAGS  -1: 0];
  reg   [ ARRAY_M*BUF_ADDR_W   -1: 0 ]         _ld_st_sys_buf_write_addr_out    [NUM_TAGS  -1: 0];
  reg   [ ARRAY_M*DATA_WIDTH   -1 : 0 ]        _ld_st_sys_buf_write_data_out    [NUM_TAGS  -1: 0];
  reg   [ ARRAY_M              -1 : 0 ]        _ld_st_sys_buf_read_req_out      [NUM_TAGS  -1: 0];
  reg   [ ARRAY_M*BUF_ADDR_W   -1: 0 ]         _ld_st_sys_buf_read_addr_out     [NUM_TAGS  -1: 0];
  wire  [ ARRAY_M*DATA_WIDTH   -1 : 0 ]        _ld_st_sys_buf_read_data_in      [NUM_TAGS  -1: 0];
  
  wire  [ 4                    -1 : 0 ]        stmem_state;
  wire  [ TAG_W                -1 : 0 ]        stmem_tag;
  wire                                         stmem_ddr_pe_sw;
//==============================================================================

  wire                                      st_buf_simd_start;
  wire                                      st_buf_simd_done;
  wire                                      st_buf_ddr_start;
  
    reg                                         raw;
  reg  [ TAG_W                -1 : 0 ]        raw_stmem_tag_d;
  reg  [ TAG_W                -1 : 0 ]        raw_stmem_tag_q;
  wire [ TAG_W                -1 : 0 ]        raw_stmem_tag;
  wire                                        raw_stmem_tag_ready;
  wire [ ADDR_WIDTH           -1 : 0 ]        raw_stmem_st_addr;
    
  reg                                         ldmem_dummy_q;
  
  
  reg  [ WAIT_CYCLES_COMPUTE_ST_W -1        : 0 ]        wait_cycles_d;
  reg  [ WAIT_CYCLES_COMPUTE_ST_W -1        : 0 ]        wait_cycles_q;
  
  
  reg                                         simd_start_d;
  reg                                         simd_start_q;
  
  wire                                        simd_st_done;
  
  wire  [ (1 << LOOP_ID_W)       : 0 ]        fsm_ld_iter_done,fsm_st_iter_done;
//==============================================================================
// Counting maximum number of group during decode
//==============================================================================
    wire                                             sa_group_v;  
    reg     [ INST_GROUP_ID_W           - 1 : 0 ]    max_groups_counter;
    reg                                              ld_st_group_loop_v;
    
    assign  sa_group_v = (inst_group_type == SA_GROUP && inst_group_s_e == GROUP_START && inst_group_v);
    
    always @(posedge clk) begin
       if (reset || block_done)
          max_groups_counter <= 0;
      else if (sa_group_v)
          max_groups_counter <= max_groups_counter + 1'b1;
    end
    
    // For OBUF, just the last group (layer) load/store data from/to offchip. We identify that through assuming that the instruction end group comes at the beginning of the group after instruction start!
    always @(posedge clk) begin
        if (reset)
            ld_st_group_loop_v <= 1'b0;
        else if (inst_group_type == SA_GROUP && inst_group_s_e == GROUP_END && inst_group_v && inst_group_last)
            ld_st_group_loop_v <= 1'b1;
    end    

//==============================================================================
// Walker stride configuration to LD OBUF from Offchip memory
//==============================================================================
    assign ld_stride = cfg_loop_stride;
    assign ld_stride_segment = cfg_loop_stride_segment; 
    assign ld_stride_v = ld_st_group_loop_v && cfg_loop_stride_v && (cfg_loop_stride_loop_id > NUM_BASE_LOOPS - 1) && cfg_loop_stride_type == MEM_LD && cfg_loop_stride_id == MEM_ID;
    assign mws_ld_loop_id = cfg_loop_stride_loop_id;

    assign mws_ld_base_addr = tag_ld_addr[ldmem_tag];
//==============================================================================
// Walker stride configuration to LD OBUF from Offchip memory
//==============================================================================
    assign st_stride = cfg_loop_stride;
    assign st_stride_segment = cfg_loop_stride_segment; 
    assign st_stride_v = ld_st_group_loop_v && cfg_loop_stride_v && (cfg_loop_stride_loop_id > NUM_BASE_LOOPS - 1) && cfg_loop_stride_type == MEM_ST && cfg_loop_stride_id == MEM_ID;

    assign mws_st_base_addr = tag_st_addr[stmem_tag];
//==============================================================================

//==============================================================================
// Address generators
//==============================================================================
    // The walker and the controller fsm needs to be programmed only with the loop IDs associated with the last instrcution group.
    // The assumption is that the systolic instructions of a group are after group start and end instructions    
    
    // TODO: some signal might be added 
    assign mws_ld_stall = ~ldmem_tag_ready || ~axi_rd_ready;
    
//    assign mws_ld_step = mws_ld_index_valid && !mws_ld_stall;
  

mem_walker_stride_group #(
    .ADDR_WIDTH                     ( ADDR_WIDTH                     ),
    .ADDR_STRIDE_W                  ( ADDR_STRIDE_W                  ),
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
	.GROUP_ID_W                     ( INST_GROUP_ID_W                ),
    .GROUP_ENABLED                  ( GROUP_ENABLED                  )
  ) mws_ld (
    .clk                            ( clk                            ), //input
    .reset                          ( reset                          ), //input
    
    .base_addr                      ( mws_ld_base_addr               ), //input
    .iter_done                      ( fsm_ld_iter_done               ), //input
    .start                          ( mws_ld_start                   ), //input
    .stall                          ( mws_ld_stall                   ),
 
    .block_done                     ( block_done                     ),
    .base_addr_v                    ( mws_ld_start                   ), //input

    .cfg_loop_id                    ( mws_ld_loop_id                 ), //input
    .cfg_addr_stride_v              ( ld_stride_v                    ), //input
    .cfg_addr_stride                ( ld_stride                      ), //input
    // NEW
    .cfg_loop_group_id		        ( inst_group_id                  ), //input
    .loop_group_id          		( inst_group_id                  ), //input
    //
    .addr_out                       ( ld_addr                        ), //output
    .addr_out_valid                 ( ld_addr_v                      )  //output
  );

    assign mws_st_step = mws_st_index_valid && !mws_st_stall;
 
 // TODO: make sure the following is ok
    assign mws_st_stall = ~stmem_tag_ready || ~axi_wr_ready || (stmem_tag_ready && ~next_st_last_group);
//    wire                                        _mws_st_done;
//    assign _mws_st_done = mws_st_done || mws_ld_done; // Added for the cases when the mws_st is programmed but not used
    
  
  mem_walker_stride_group #(
    .ADDR_WIDTH                     ( ADDR_WIDTH                     ),
    .ADDR_STRIDE_W                  ( ADDR_STRIDE_W                  ),
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
	.GROUP_ID_W                     ( INST_GROUP_ID_W                ),
    .GROUP_ENABLED                  ( GROUP_ENABLED                  )
  ) mws_st (
    .clk                            ( clk                            ), //input
    .reset                          ( reset                          ), //input
    
    .base_addr                      ( mws_st_base_addr               ), //input
    .iter_done                      ( fsm_st_iter_done               ), //input
    .start                          ( mws_st_start                   ), //input
    .stall                          ( mws_st_stall                   ),
 
    .block_done                     ( block_done                     ),
    .base_addr_v                    ( mws_st_start                   ), //input

    .cfg_loop_id                    ( mws_st_loop_id                 ), //input
    .cfg_addr_stride_v              ( st_stride_v                    ), //input
    .cfg_addr_stride                ( st_stride                      ), //input
    // NEW
    .cfg_loop_group_id		        ( inst_group_id                  ), //input
    .loop_group_id          		( inst_group_id                  ), //input
    //
    .addr_out                       ( st_addr                        ), //output
    .addr_out_valid                 ( st_addr_v                      )  //output
  );
//==============================================================================

//=============================================================
// Loop controller for OBUF
//=============================================================
 always @(posedge clk)
  begin
    if (reset)
      ld_iter_v_q <= 1'b0;
    else begin
      if (cfg_loop_iter_v && (cfg_loop_iter_loop_id > NUM_BASE_LOOPS - 1))
        ld_iter_v_q <= 1'b1;
      else if (cfg_loop_iter_v || ld_stride_v)
        ld_iter_v_q <= 1'b0;
    end
  end

  always @(posedge clk)
  begin
    if (reset)
      st_iter_v_q <= 1'b0;
    else begin
      if (cfg_loop_iter_v && (cfg_loop_iter_loop_id > NUM_BASE_LOOPS - 1))
        st_iter_v_q <= 1'b1;
      else if (cfg_loop_iter_v || st_stride_v)
        st_iter_v_q <= 1'b0;
    end
  end

  always @(posedge clk)
  begin
    if (reset) begin
      iter_q <= 0;
      loop_id_q <= 0;
      loop_level_q <= 0;
    end
    else if (cfg_loop_iter_v && (cfg_loop_iter_loop_id > NUM_BASE_LOOPS - 1)) begin
      iter_q <= cfg_loop_iter;
      loop_id_q <= cfg_loop_iter_loop_id;
      loop_level_q <= cfg_loop_iter_level; 
    end
  end


// We are assuming that first the loop instructions come and then the stride instructions come
    assign mws_ld_start = (ldmem_state_q == LDMEM_BUSY) && (ldmem_state_qq != LDMEM_BUSY);
    assign mws_ld_loop_iter_v = ld_stride_v && ld_iter_v_q && (loop_id_q == cfg_loop_stride_loop_id);
    assign mws_ld_loop_iter = iter_q;
    assign mws_ld_loop_iter_loop_id = loop_id_q;
    assign mws_ld_loop_iter_loop_level = loop_level_q;

// TODO: This can be later updated with a controller that supports groups and layer fusion, but we actually do not need it for here in IBUF!
  
  controller_fsm_group #(
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
    .LOOP_ITER_W                    ( LOOP_ITER_W                    ),
    .GROUP_ID_W                     ( INST_GROUP_ID_W                ),
    .GROUP_ENABLED                  ( GROUP_ENABLED                  )
  ) mws_ld_ctrl (
    .clk                            ( clk                            ), //input
    .reset                          ( reset                          ), //input
    
    .start                          ( mws_ld_start                   ), //input
    .block_done                     ( block_done                     ),
    .done                           ( mws_ld_done                    ), //output
    .stall                          ( mws_ld_stall                   ), //input
    
    .cfg_loop_iter_v                ( mws_ld_loop_iter_v             ), //input
    .cfg_loop_iter                  ( mws_ld_loop_iter               ), //input
    .cfg_loop_iter_loop_id          ( mws_ld_loop_iter_loop_id       ), //input
    
    .cfg_loop_group_id              ( inst_group_id     			 ), //input
    .loop_group_id                  ( inst_group_id                  ), //input

    .iter_done                      ( fsm_ld_iter_done               ),
    .current_iters                  (                                )
  );
//=============================================================

// We are assuming that first the loop instructions come and then the stride instructions come
    assign mws_st_start = ldmem_state_q == STMEM_DDR;
    assign mws_st_loop_iter_v = st_stride_v && st_iter_v_q && (loop_id_q == cfg_loop_stride_loop_id);
    assign mws_st_loop_iter = iter_q;
    assign mws_st_loop_iter_loop_id = loop_id_q;
    assign mws_st_loop_iter_loop_level = loop_level_q;

  
  controller_fsm_group #(
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
    .LOOP_ITER_W                    ( LOOP_ITER_W                    ),
    .GROUP_ID_W                     ( INST_GROUP_ID_W                ),
    .GROUP_ENABLED                  ( GROUP_ENABLED                  )
  ) mws_st_ctrl (
    .clk                            ( clk                            ), //input
    .reset                          ( reset                          ), //input
    
    .start                          ( mws_st_start                   ), //input
    .block_done                     ( block_done                     ),
    .done                           ( mws_st_done                    ), //output
    .stall                          ( mws_st_stall                   ), //input
    
    .cfg_loop_iter_v                ( mws_st_loop_iter_v             ), //input
    .cfg_loop_iter                  ( mws_st_loop_iter               ), //input
    .cfg_loop_iter_loop_id          ( mws_st_loop_iter_loop_id       ), //input
    
    .cfg_loop_group_id              ( inst_group_id     			 ), //input
    .loop_group_id                  ( inst_group_id                  ), //input

    .iter_done                      ( fsm_st_iter_done               ),
    .current_iters                  (                                )
  );
//=============================================================

//==============================================================================
// Memory Request generation
//==============================================================================
    assign ld_mem_req_v = ld_st_group_loop_v && cfg_mem_req_v && (cfg_mem_req_loop_id >  NUM_BASE_LOOPS - 1) && cfg_mem_req_type == MEM_LD && cfg_mem_req_id == MEM_ID;
  always @(posedge clk)
  begin
    if (reset) begin
      ld_req_size <= 0;
    end
    else if (ld_mem_req_v) begin
      ld_req_size <= cfg_mem_req_size;
    end
  end

    assign st_mem_req_v = ld_st_group_loop_v && cfg_mem_req_v && (cfg_mem_req_loop_id >  NUM_BASE_LOOPS - 1) && cfg_mem_req_type == MEM_ST && cfg_mem_req_id == MEM_ID;
  always @(posedge clk)
  begin
    if (reset) begin
      st_req_size <= 0;
    end
    else if (st_mem_req_v) begin
      st_req_size <= cfg_mem_req_size;
    end
  end

    // TODO: Make sure that the following is correct
    assign ld_req_valid_d = ld_addr_v;
    assign st_req_valid_d = st_addr_v;


  always @(posedge clk)
  begin
    if (reset) begin
      ld_req_valid_q <= 1'b0;
      ld_req_addr <= 0;
      st_req_valid_q <= 1'b0;
      st_req_addr <= 0;
      
    end
    else begin
      ld_req_valid_q <= ld_req_valid_d;
      ld_req_addr <= ld_addr;
      st_req_valid_q <= st_req_valid_d;
      st_req_addr <= st_addr;
    end
  end

// TODO: (controller): In the controller it needs to be managed in a way that the base address for LD/ST just be generated for the last group (fused layer)
// To make sure, we only accept the base address, when next tag is gonna be used for the last group
//  always @(posedge clk)
//  begin
//    if (tag_req && tag_ready && next_compute_last_group) begin
//      tag_ld_addr[tag] <= tag_base_ld_addr;
//    end
//  end
//
//  always @(posedge clk)
//  begin
//    if (tag_req && tag_ready && next_st_last_group) begin
//      tag_st_addr[tag] <= tag_base_st_addr;
//    end
//end
// TODO: Double check in the controller, it seems like that the base_addr for LD/ST comes at the same time, when there is a free tag
  always @(posedge clk)
  begin
//    if (tag_req && tag_ready) begin
    if (base_ld_addr_v) begin
      tag_ld_addr[tag] <= tag_base_ld_addr;
    end
  end

  always @(posedge clk)
  begin
//    if (tag_req && tag_ready) begin
    if (base_st_addr_v) begin
      tag_st_addr[tag] <= tag_base_st_addr;
    end
  end
//==============================================================================

//==============================================================================
// Logic for handling the LD/ST of multiple fused layers
//==============================================================================
  // Used for ST
  always @(posedge clk) begin
      if (reset) begin
         next_tag_compute_counter <= 0;
     end
     else if (compute_start) begin
         if (next_tag_compute_counter == NUM_TAGS - 1'b1)
            next_tag_compute_counter <= 0;
         else
            next_tag_compute_counter <= next_tag_compute_counter + 1'b1;   
     end
  end 

 always @(posedge clk) begin
     if (reset) begin
        next_group_compute_counter <= 0;
     end
    else if (compute_start && next_tag_compute_counter == NUM_TAGS - 1'b1) begin
        if (next_group_compute_counter == max_groups_counter - 1)
            next_group_compute_counter <= 0;
        else
            next_group_compute_counter <= next_group_compute_counter + 1'b1;     
    end
end

  // Used for LD
  always @(posedge clk) begin
      if (reset) begin
         next_tag_st_counter <= 0;
     end
     else if (st_buf_start) begin
         if (next_tag_st_counter == NUM_TAGS - 1'b1)
            next_tag_st_counter <= 0;
         else
            next_tag_st_counter <= next_tag_st_counter + 1'b1;   
     end
  end 

 always @(posedge clk) begin
     if (reset) begin
        next_group_st_counter <= 0;
     end
    else if (st_buf_start && next_tag_st_counter == NUM_TAGS - 1'b1) begin
        if (next_group_st_counter == max_groups_counter - 1)
            next_group_st_counter <= 0;
        else
            next_group_st_counter <= next_group_st_counter + 1'b1;     
    end
  end

  assign next_compute_last_group = next_group_compute_counter == max_groups_counter - 1;
  assign next_st_last_group = next_group_st_counter == max_groups_counter - 1;
//==============================================================================
// Tag-based synchronization for double buffering
//==============================================================================

 // Logic to emulate LD (LDMEM_DUMMY) when we do not need to LD anything from DDR   
  always @(posedge clk) begin
    if (reset)
        ldmem_dummy_q <= 1'b0;
    else if (ldmem_state_q == LDMEM_DUMMY)
        ldmem_dummy_q <= 1'b1;
    else if (ldmem_state_q == LDMEM_DONE)
        ldmem_dummy_q <= 1'b0;
  end  
  
  always @(posedge clk)  
    ldmem_state_qq <= ldmem_state_q;
    
  always @(posedge clk)
  begin
    if (reset)
      raw_stmem_tag_q <= 0;
    else
      raw_stmem_tag_q <= raw_stmem_tag_d;
  end

    assign raw_stmem_tag = raw_stmem_tag_q;
    assign raw_stmem_st_addr = tag_st_addr[raw_stmem_tag];

    assign stmem_state = stmem_state_q;
//============================================================== 
// LD FSM
//==============================================================    
  always @(*)
  begin
    ldmem_state_d = ldmem_state_q;
    raw_stmem_tag_d = raw_stmem_tag_q;
    case(ldmem_state_q)
      LDMEM_IDLE: begin
        if (ldmem_tag_ready && !next_compute_last_group || ldmem_tag_ready && next_compute_last_group && first_ic_outer_loop_ld)
            ldmem_state_d = LDMEM_DUMMY;
        else if (ldmem_tag_ready && next_compute_last_group && ~first_ic_outer_loop_ld) begin
            if (ldmem_tag == stmem_tag)
               ldmem_state_d = LDMEM_BUSY;
           else begin
               ldmem_state_d = LDMEM_CHECK_RAW;
               raw_stmem_tag_d = stmem_tag;              
           end
        end
      end
      LDMEM_CHECK_RAW: begin
        if (raw_stmem_st_addr != mws_ld_base_addr)
          ldmem_state_d = LDMEM_BUSY;
        else if (stmem_state_q == STMEM_DONE)
          ldmem_state_d = LDMEM_IDLE;
      end
      LDMEM_BUSY: begin
        if (mws_ld_done)
          ldmem_state_d = LDMEM_WAIT_0;
      end
      LDMEM_DUMMY: begin
          ldmem_state_d = LDMEM_WAIT_0; 
      end
      LDMEM_WAIT_0: begin
        ldmem_state_d = LDMEM_WAIT_1;
      end
      LDMEM_WAIT_1: begin
        ldmem_state_d = LDMEM_WAIT_2;
      end
      LDMEM_WAIT_2: begin
        ldmem_state_d = LDMEM_WAIT_3;
      end
      LDMEM_WAIT_3: begin
        if (axi_rd_done || ldmem_dummy_q)
          ldmem_state_d = LDMEM_DONE;
      end
      LDMEM_DONE: begin
        ldmem_state_d = LDMEM_IDLE;
      end
    endcase
  end

  always @(posedge clk)
  begin
    if (reset)
      ldmem_state_q <= LDMEM_IDLE;
    else
      ldmem_state_q <= ldmem_state_d;
  end
//================================================================
//================================================================
// ST FSM
//================================================================
  always @(posedge clk)
  begin
    if (reset)
      wait_cycles_q <= 0;
    else
      wait_cycles_q <= wait_cycles_d;
  end

// Logic to activate SIMD array
  always @(posedge clk) begin
     if (reset)
        simd_start_q <= 1'b0;
     else
        simd_start_q <= simd_start_d; 
  end
  
  assign simd_activate = simd_start_q;




  always @(*)
  begin     
    stmem_state_d = stmem_state_q;
    simd_start_d = 1'b0;
    wait_cycles_d = wait_cycles_q;   
    case(stmem_state_q)
      STMEM_IDLE: begin
        if (stmem_tag_ready) begin
          stmem_state_d = STMEM_COMPUTE_WAIT;
          wait_cycles_d = WAIT_CYCLES_COMPUTE_ST;
        end
      end
      STMEM_COMPUTE_WAIT: begin
        if (wait_cycles_q == 0) begin
            if ((~next_st_last_group && next_group_simd) || (next_st_last_group && stmem_ddr_pe_sw && next_group_simd)) begin
                // We need to store on SIMD
                if (simd_ready) begin
                   stmem_state_d = STMEM_SIMD; 
                   simd_start_d = 1'b1;
                end
            end
            else if ((next_st_last_group && ~stmem_ddr_pe_sw) || (next_st_last_group && stmem_ddr_pe_sw && ~next_group_simd)) begin
                // We need to store on DDR
                stmem_state_d = STMEM_DDR;
            end
        else
            wait_cycles_d = wait_cycles_q - 1'b1;
      end
      end
      STMEM_SIMD: begin
        if (st_buf_simd_done)
          stmem_state_d = STMEM_DONE;
      end
      STMEM_DDR: begin
        if (mws_st_done) begin
          stmem_state_d = STMEM_WAIT_0;
        end
      end
      STMEM_WAIT_0: begin
         stmem_state_d = STMEM_WAIT_1; 
      end
      STMEM_WAIT_1: begin
         stmem_state_d = STMEM_WAIT_2; 
      end
      STMEM_WAIT_2: begin
        stmem_state_d = STMEM_WAIT_3;
      end
      STMEM_WAIT_3: begin
         if (axi_wr_done)
             stmem_state_d = STMEM_DONE;
      end
      STMEM_DONE: begin
         stmem_state_d = STMEM_IDLE; 
      end
    endcase
  end

  always @(posedge clk)
  begin
    if (reset)
      stmem_state_q <= STMEM_IDLE;
    else
      stmem_state_q <= stmem_state_d;
  end

//=============================================================
// TAG Logic
//=============================================================
    wire                                        ldmem_ready;

    assign compute_tag_done = compute_done;
    assign compute_ready = compute_tag_ready;

    assign ldmem_tag_done = ldmem_state_q == LDMEM_DONE;
    assign ldmem_ready = ldmem_tag_ready;
  // assign ldmem_tag_done = mws_ld_done;

    assign stmem_tag_done = stmem_state_q == STMEM_DONE;

  tag_sync  #(
    .NUM_TAGS                       ( NUM_TAGS                       )
  )
  mws_tag (
    .clk                            ( clk                            ),
    .reset                          ( reset                          ),
    .block_done                     ( block_done                     ),
    .tag_req                        ( tag_req                        ),
    .tag_reuse                      ( tag_reuse                      ),
    .tag_bias_prev_sw               ( tag_bias_prev_sw               ),
    .tag_ddr_pe_sw                  ( tag_ddr_pe_sw                  ), //input
    .tag_ready                      ( tag_ready                      ),
    .tag                            ( tag                            ),
    .tag_done                       ( tag_done                       ),
    .raw_stmem_tag                  ( raw_stmem_tag_q                ),
    .raw_stmem_tag_ready            ( raw_stmem_tag_ready            ),
    .compute_tag_done               ( compute_tag_done               ),
    .compute_tag_ready              ( compute_tag_ready              ),
    .compute_bias_prev_sw           ( compute_bias_prev_sw           ),
    .compute_tag                    ( compute_tag                    ),
    .ldmem_tag_done                 ( ldmem_tag_done                 ),
    .ldmem_tag_ready                ( ldmem_tag_ready                ),
    .ldmem_tag                      ( ldmem_tag                      ),
    .stmem_ddr_pe_sw                ( stmem_ddr_pe_sw                ),
    .stmem_tag_done                 ( stmem_tag_done                 ),
    .stmem_tag_ready                ( stmem_tag_ready                ),
    .stmem_tag                      ( stmem_tag                      )
  );
//==============================================================================


//==============================================================================
// AXI4 Memory Mapped interface
//==============================================================================
    assign axi_rd_req = ld_req_valid_q;
    assign axi_rd_req_size = ld_req_size * (ARRAY_M * DATA_WIDTH) / AXI_DATA_WIDTH;
    assign axi_rd_addr = ld_req_addr;
    assign axi_rd_req_id = 1'b0;
    
    assign axi_wr_req = st_req_valid_q;
    assign axi_wr_req_id = 1'b0;
    assign axi_wr_req_size = st_req_size * (ARRAY_M * DATA_WIDTH) / AXI_DATA_WIDTH;
    assign axi_wr_addr = st_req_addr;
    
    assign mem_write_ready = 1'b1;
//    assign mem_read_ready = 1'b1;

  axi_master #(
    .TX_SIZE_WIDTH                  ( MEM_REQ_W                      ),
    .AXI_DATA_WIDTH                 ( AXI_DATA_WIDTH                 ),
    .AXI_ADDR_WIDTH                 ( AXI_ADDR_WIDTH                 ),
    .AXI_BURST_WIDTH                ( AXI_BURST_WIDTH                )
  ) u_axi_mm_master (
    .clk                            ( clk                            ),
    .reset                          ( reset                          ),
    .m_axi_awaddr                   ( mws_awaddr                     ),
    .m_axi_awlen                    ( mws_awlen                      ),
    .m_axi_awsize                   ( mws_awsize                     ),
    .m_axi_awburst                  ( mws_awburst                    ),
    .m_axi_awvalid                  ( mws_awvalid                    ),
    .m_axi_awready                  ( mws_awready                    ),
    .m_axi_wdata                    ( mws_wdata                      ),
    .m_axi_wstrb                    ( mws_wstrb                      ),
    .m_axi_wlast                    ( mws_wlast                      ),
    .m_axi_wvalid                   ( mws_wvalid                     ),
    .m_axi_wready                   ( mws_wready                     ),
    .m_axi_bresp                    ( mws_bresp                      ),
    .m_axi_bvalid                   ( mws_bvalid                     ),
    .m_axi_bready                   ( mws_bready                     ),
    .m_axi_araddr                   ( mws_araddr                     ),
    .m_axi_arid                     ( mws_arid                       ),
    .m_axi_arlen                    ( mws_arlen                      ),
    .m_axi_arsize                   ( mws_arsize                     ),
    .m_axi_arburst                  ( mws_arburst                    ),
    .m_axi_arvalid                  ( mws_arvalid                    ),
    .m_axi_arready                  ( mws_arready                    ),
    .m_axi_rdata                    ( mws_rdata                      ),
    .m_axi_rid                      ( mws_rid                        ),
    .m_axi_rresp                    ( mws_rresp                      ),
    .m_axi_rlast                    ( mws_rlast                      ),
    .m_axi_rvalid                   ( mws_rvalid                     ),
    .m_axi_rready                   ( mws_rready                     ),
    // Buffer
    .mem_write_id                   ( mem_write_id                   ),
    .mem_write_req                  ( mem_write_req                  ),
    .mem_write_data                 ( mem_write_data                 ),
    .mem_write_ready                ( mem_write_ready                ),
    .mem_read_data                  ( mem_read_data                  ),
    .mem_read_req                   ( axi_mem_read_req               ),
    .mem_read_ready                 ( axi_mem_read_ready             ),
    // AXI RD Req
    .rd_req                         ( axi_rd_req                     ),
    .rd_req_id                      ( axi_rd_req_id                  ),
    .rd_done                        ( axi_rd_done                    ),
    .rd_ready                       ( axi_rd_ready                   ),
    .rd_req_size                    ( axi_rd_req_size                ),
    .rd_addr                        ( axi_rd_addr                    ),
    // AXI WR Req
    .wr_req                         ( axi_wr_req                     ),
    .wr_req_id                      ( axi_wr_req_id                  ),
    .wr_ready                       ( axi_wr_ready                   ),
    .wr_req_size                    ( axi_wr_req_size                ),
    .wr_addr                        ( axi_wr_addr                    ),
    .wr_done                        ( axi_wr_done                    )
  );
//==============================================================================

//==============================================================================
// SIMD/DDR START/DONE Control
//==============================================================================

    
  assign st_buf_simd_start = simd_buf_read_req[0] && &(~simd_buf_read_req[ARRAY_M-1:1]);
  assign st_buf_simd_done = simd_buf_read_req[ARRAY_M-1] && &(~simd_buf_read_req[ARRAY_N-2:0]);

  assign st_buf_ddr_start = stmem_state_q == STMEM_DDR;
  assign st_buf_start = st_buf_ddr_start || st_buf_simd_start;
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
// Logic Interface for LD
//==============================================================================
  always @(posedge clk)
  begin
      if (reset)
          buf_write_ext_counter_group <= 0;
      else if (mem_write_req) begin
          if (buf_write_ext_counter_group == BUF_WRITE_NUM_GROUP_EXT - 1)
              buf_write_ext_counter_group <= 0;
          else
              buf_write_ext_counter_group <= buf_write_ext_counter_group + 1'b1;
      end
  end
 //

//
  always @(posedge clk)
  begin
      if (reset)
          _buf_ext_write_addr <= 0;
      else begin 
          if (mem_write_req && buf_write_ext_counter_group == BUF_WRITE_NUM_GROUP_EXT - 1)  
            _buf_ext_write_addr <= _buf_ext_write_addr + 1'b1;
          else if (ldmem_state_q == LDMEM_DONE)
            _buf_ext_write_addr <= 0;
      end
  end
//

// ASSIGNs
//Assign Data out
  assign buf_ext_write_data_out = {BUF_WRITE_NUM_GROUP_EXT{mem_write_data}};
   
//Assign Address out  
  assign  buf_ext_write_addr = _buf_ext_write_addr;
//  assign  tag_buf_ext_write_addr = {ldmem_tag, buf_ext_write_addr};   
  genvar i;
  generate
    for (i=0; i<BUF_WRITE_GROUP_SIZE_EXT; i=i+1) begin
        assign group_buf_ext_write_addr[(i+1)*BUF_ADDR_W-1: i*BUF_ADDR_W] = buf_ext_write_addr;     
    end
  endgenerate
  
  assign buf_ext_write_addr_out = {BUF_WRITE_NUM_GROUP_EXT{group_buf_ext_write_addr}};
  
  
//Assign Address Req Out
  assign buf_ext_write_req = mem_write_req;

  genvar j;
  generate
      for (j=0; j<BUF_WRITE_GROUP_SIZE_EXT; j=j+1) begin
          assign group_buf_ext_write_req[j] = buf_ext_write_req;
      end
  endgenerate
  
  genvar k;
  generate
      for (k=0; k<BUF_WRITE_NUM_GROUP_EXT; k=k+1) begin
            assign buf_ext_write_req_out[(k+1)*BUF_WRITE_GROUP_SIZE_EXT-1: (k)*BUF_WRITE_GROUP_SIZE_EXT] = (buf_write_ext_counter_group == k) ? group_buf_ext_write_req : 0;
      end
  endgenerate
//============================================================================== 
// Logic Interface for ST
//============================================================================== 
  always @(posedge clk)
  begin
      if (reset)
          buf_read_ext_counter_group <= 0;
      else if (axi_mem_read_req) begin
          if (buf_read_ext_counter_group == BUF_READ_NUM_GROUP_EXT - 1)
              buf_read_ext_counter_group <= 0;
          else
              buf_read_ext_counter_group <= buf_read_ext_counter_group + 1'b1;
      end
  end
 //
  wire [ COUNTER_BUF_READ_GROUP_W      -1 : 0 ] buf_read_ext_counter_group_delayed;
  register_sync #(COUNTER_BUF_READ_GROUP_W) buf_read_ext_group_counter_reg (clk, reset, buf_read_ext_counter_group, buf_read_ext_counter_group_delayed);
//
  always @(posedge clk)
  begin
      if (reset)
          _buf_ext_read_addr <= 0;
      else begin 
          if (axi_mem_read_req && buf_read_ext_counter_group == BUF_READ_NUM_GROUP_EXT - 1)  
            _buf_ext_read_addr <= _buf_ext_read_addr + 1'b1;
          else if (stmem_state_q == STMEM_DONE)
            _buf_ext_read_addr <= 0;
      end
  end
//

// ASSIGNs
//Assign Data out
  assign mem_read_data = {BUF_READ_NUM_GROUP_EXT{_buf_ext_read_data}};   
//Assign Address out  
  assign  buf_ext_read_addr = _buf_ext_read_addr;
//  assign  tag_buf_ext_read_addr = {stmem_tag, buf_ext_read_addr};
    

  generate
    for (i=0; i<BUF_READ_GROUP_SIZE_EXT; i=i+1) begin
        assign group_buf_ext_read_addr[(i+1)*BUF_ADDR_W-1: i*BUF_ADDR_W] = buf_ext_read_addr;     
    end
  endgenerate
  
  assign buf_ext_read_addr_out = {BUF_READ_NUM_GROUP_EXT{group_buf_ext_read_addr}};
  
  
//Assign Address Req Out
  assign buf_ext_read_req = axi_mem_read_req;

  generate
      for (j=0; j<BUF_READ_GROUP_SIZE_EXT; j=j+1) begin
          assign group_buf_ext_read_req[j] = buf_ext_read_req;
      end
  endgenerate
  

  generate
      for (k=0; k<BUF_READ_NUM_GROUP_EXT; k=k+1) begin
              assign buf_ext_read_req_out[(k+1)*BUF_READ_GROUP_SIZE_EXT-1: (k)*BUF_READ_GROUP_SIZE_EXT] = (buf_read_ext_counter_group == k) ? group_buf_ext_read_req : 0;
      end
  endgenerate
  
  assign axi_mem_read_ready = stmem_state_q != STMEM_SIMD;
//============================================================================== 
// Logic Interface for ST from SIMD 
//============================================================================== 
  assign buf_simd_read_req_out = simd_buf_read_req;
  assign buf_simd_read_addr_out = simd_buf_read_addr;
  
  wire  [ ARRAY_M           -1 : 0 ]        simd_read_state;
  genvar s;
  generate
      for (s=0; s<ARRAY_M; s=s+1) begin
          assign simd_read_state[s] = stmem_state_q == STMEM_SIMD;
       end
  endgenerate
  
  // Delay logic to generate the data_valid signals, assumption is that the data will be ready in the next cycle
  register_sync #(ARRAY_M) buf_data_valid_delay (clk, reset, simd_buf_read_req & simd_read_state, simd_data_valid);
  
  assign simd_buf_read_data = _simd_buf_read_data;
//==============================================================================
// Logic Interface for Systolic Array---OBUF
//==============================================================================
// Delaying the compute tag for (N-1) cycles for read_obuf and (N+1) cycles for write_obuf
  genvar r;
  generate
      for (r=0; r<ARRAY_N+2; r=r+1)
      begin: TAG_DELAY_LOOP
        wire [TAG_W-1:0] prev_tag, next_tag;
        if (r==0)
    assign prev_tag = compute_tag;
        else
    assign prev_tag = TAG_DELAY_LOOP[r-1].next_tag;
        register_sync #(TAG_W) tag_delay (clk, reset, prev_tag, next_tag);
      end
      // Increased compute tag delays
      // Might need a separate compute_tag_delayed for buf_read
    assign compute_tag_delayed_buf_write = TAG_DELAY_LOOP[ARRAY_N+1].next_tag;
    assign compute_tag_delayed_buf_read = TAG_DELAY_LOOP[ARRAY_N-1].next_tag;
 endgenerate  
 
  assign sys_buf_write_req_out = sys_buf_write_req_in;
  assign sys_buf_write_addr_out = sys_buf_write_addr_in;
  assign sys_buf_write_data_out = sys_buf_write_data_in;
  
  assign sys_buf_read_req_out = sys_buf_read_req_in;
  assign sys_buf_read_addr_out = sys_buf_read_addr_in;
  assign sys_buf_read_data_out = _sys_buf_read_data_in;

  // Logic to identify the compute state
  reg                           compute_state;
  always @(posedge clk) begin
     if (reset)
        compute_state <= 1'b0;
     else if (compute_start)
        compute_state <= 1'b1;
     else if (stmem_state_q == STMEM_COMPUTE_WAIT && wait_cycles_d == 0)
        compute_state <= 1'b0; 
  end
//==============================================================================
// Logic to Map SIMD/AXI/Systolic signals to the Buf signals
//============================================================================== 
 // Following three possible states are based on double buffering. It has to be updated if we want to do more!
 always @(*) begin
     // LD/ST
    if ((stmem_state_q == STMEM_DDR || stmem_state_q == STMEM_SIMD) && ldmem_state_q == LDMEM_BUSY) begin
        _ld_st_sys_buf_write_req_out[ldmem_tag] = buf_ext_write_req_out;
        _ld_st_sys_buf_write_addr_out[ldmem_tag] = buf_ext_write_addr_out;
        _ld_st_sys_buf_write_data_out[ldmem_tag] = buf_ext_write_data_out;
        if (stmem_state_q == STMEM_DDR) begin
            _ld_st_sys_buf_read_req_out[stmem_tag] = buf_ext_read_req_out;
            _ld_st_sys_buf_read_addr_out[stmem_tag] = buf_ext_read_addr_out;
            _buf_ext_read_data = _ld_st_sys_buf_read_data_in[stmem_tag];
        end
        else if (stmem_state_q == STMEM_SIMD) begin
            _ld_st_sys_buf_read_req_out[stmem_tag] = buf_simd_read_req_out;
            _ld_st_sys_buf_read_addr_out[stmem_tag] = buf_simd_read_addr_out;
            _simd_buf_read_data = _ld_st_sys_buf_read_data_in[stmem_tag];
        end
    end
    // ST/Compute
    else if ((stmem_state_q == STMEM_DDR || stmem_state_q == STMEM_SIMD) && compute_state) begin
        _ld_st_sys_buf_write_req_out[compute_tag_delayed_buf_write] = sys_buf_write_req_out;
        _ld_st_sys_buf_write_addr_out[compute_tag_delayed_buf_write] = sys_buf_write_addr_out;
        _ld_st_sys_buf_write_data_out[compute_tag_delayed_buf_write] = sys_buf_write_data_out;
        
        _ld_st_sys_buf_read_req_out[compute_tag_delayed_buf_read] = sys_buf_read_req_out;
        _ld_st_sys_buf_read_addr_out[compute_tag_delayed_buf_read] = sys_buf_read_addr_out;
        _sys_buf_read_data_in = _ld_st_sys_buf_read_data_in[compute_tag_delayed_buf_read];
        if (stmem_state_q == STMEM_DDR) begin
            _ld_st_sys_buf_read_req_out[stmem_tag] = buf_ext_read_req_out;
            _ld_st_sys_buf_read_addr_out[stmem_tag] = buf_ext_read_addr_out;
            _buf_ext_read_data = _ld_st_sys_buf_read_data_in[stmem_tag] ;
        end
        else if (stmem_state_q == STMEM_SIMD) begin
            _ld_st_sys_buf_read_req_out[stmem_tag] = buf_simd_read_req_out;
            _ld_st_sys_buf_read_addr_out[stmem_tag] = buf_simd_read_addr_out;
            _simd_buf_read_data = _ld_st_sys_buf_read_data_in[stmem_tag];
        end 
    end
    // LD/Compute
    else if (ldmem_state_q == LDMEM_BUSY && compute_state) begin
        _ld_st_sys_buf_write_req_out[compute_tag_delayed_buf_write] = sys_buf_write_req_out;
        _ld_st_sys_buf_write_addr_out[compute_tag_delayed_buf_write] = sys_buf_write_addr_out;
        _ld_st_sys_buf_write_data_out[compute_tag_delayed_buf_write] = sys_buf_write_data_out;
        
        _ld_st_sys_buf_read_req_out[compute_tag_delayed_buf_read] = sys_buf_read_req_out;
        _ld_st_sys_buf_read_addr_out[compute_tag_delayed_buf_read] = sys_buf_read_addr_out;
        _sys_buf_read_data_in = _ld_st_sys_buf_read_data_in[compute_tag_delayed_buf_read];
        
        _ld_st_sys_buf_write_req_out[ldmem_tag] = buf_ext_write_req_out;
        _ld_st_sys_buf_write_addr_out[ldmem_tag] = buf_ext_write_addr_out;
        _ld_st_sys_buf_write_data_out[ldmem_tag] = buf_ext_write_data_out;        
    end  
 end
//============================================================================== 
//==============================================================================
// Final Assignments
//==============================================================================
  generate
  for(genvar t = 0 ; t < NUM_TAGS ; t = t+1) begin
      assign  ld_st_sys_buf_write_req_out[(t+1)*ARRAY_M-1:t*ARRAY_M] = _ld_st_sys_buf_write_req_out[t];
      assign ld_st_sys_buf_write_addr_out[(t+1)*ARRAY_M*BUF_ADDR_W-1:t*ARRAY_M*BUF_ADDR_W] = _ld_st_sys_buf_write_addr_out[t];
      assign ld_st_sys_buf_write_data_out[(t+1)*ARRAY_M*DATA_WIDTH-1:t*ARRAY_M*DATA_WIDTH] = _ld_st_sys_buf_write_data_out[t];
      assign   ld_st_sys_buf_read_req_out[(t+1)*ARRAY_M-1:t*ARRAY_M] = _ld_st_sys_buf_read_req_out[t];
      assign  ld_st_sys_buf_read_addr_out[(t+1)*ARRAY_M*BUF_ADDR_W-1:t*ARRAY_M*BUF_ADDR_W] = _ld_st_sys_buf_read_addr_out[t];
      assign   _ld_st_sys_buf_read_data_in[t] = ld_st_sys_buf_read_data_in[(t+1)*ARRAY_M*DATA_WIDTH-1:t*ARRAY_M*DATA_WIDTH];
  end
  endgenerate
//==============================================================================
//`ifdef COCOTB_SIM
//  integer wr_req_count=0;
//  integer rd_req_count=0;
//  integer missed_rd_req_count=0;
//  integer req_count;
//
//  always @(posedge clk)
//    if (reset)
//      wr_req_count <= 0;
//    else
//      wr_req_count <= wr_req_count + axi_wr_req;
//
//  always @(posedge clk)
//    if (reset)
//      rd_req_count <= 0;
//    else
//      rd_req_count <= rd_req_count + axi_rd_req;
//
//  always @(posedge clk)
//    if (reset)
//      missed_rd_req_count <= 0;
//    else
//      missed_rd_req_count <= missed_rd_req_count + (axi_rd_req && ~axi_rd_ready);
//
//  always @(posedge clk)
//  begin
//    if (reset) req_count <= 0;
//    else req_count = req_count + (tag_req && tag_ready);
//  end
//`endif
////==============================================================================
//
////=============================================================
//// VCD
////=============================================================
//`ifdef COCOTB_TOPLEVEL_mem_wrapper
//initial begin
//  $dumpfile("mem_wrapper.vcd");
//  $dumpvars(0, mem_wrapper);
//end
//`endif
////=============================================================
endmodule
