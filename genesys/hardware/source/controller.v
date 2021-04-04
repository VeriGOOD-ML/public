//
// GeneSys main controller

`timescale 1ns/1ps
module controller #(
    parameter integer  NUM_TAGS                     = 2,
    parameter integer  TAG_W                        = $clog2(NUM_TAGS),
    parameter integer  ADDR_WIDTH                   = 42,
    parameter integer  IBUF_ADDR_WIDTH              = ADDR_WIDTH,
    parameter integer  WBUF_ADDR_WIDTH              = ADDR_WIDTH,
    parameter integer  OBUF_ADDR_WIDTH              = ADDR_WIDTH,
    parameter integer  BBUF_ADDR_WIDTH              = ADDR_WIDTH,
    parameter integer  NUM_BASE_LOOPS               = 7,
  // Instructions
    parameter integer  INST_DATA_WIDTH              = 32,
    parameter integer  INST_WSTRB_WIDTH             = INST_DATA_WIDTH/8,
    parameter integer  INST_BURST_WIDTH             = 8,
  // Decoder
    parameter integer  BUF_TYPE_W                   = 2,
    parameter integer  IMM_WIDTH                    = 16,
    parameter integer  LOOP_ITER_W                  = IMM_WIDTH,
    parameter integer  MEM_REQ_W                    = IMM_WIDTH,
    parameter integer  ADDR_STRIDE_W                = 2*IMM_WIDTH,
    parameter integer  GROUP_ENABLED                = 0,
    
    parameter integer  OP_CODE_W                    = 4,
    parameter integer  OP_SPEC_W                    = 6,
    parameter integer  LOOP_ID_W                    = 6,
    parameter integer  INST_GROUP_ID_W              = 4,
    parameter integer  NUM_MAX_LOOPS                = (1 << INST_GROUP_ID_W),
    parameter integer  EXEC_DONE_WAIT_CYCLES        = 128,
  // AXI-Lite
    parameter integer  CTRL_ADDR_WIDTH              = 32,
    parameter integer  CTRL_DATA_WIDTH              = 32,
    parameter integer  CTRL_WSTRB_WIDTH             = CTRL_DATA_WIDTH/8,
  // AXI
    parameter integer  AXI_BURST_WIDTH              = 8,
    parameter integer  AXI_DATA_WIDTH               = 64,
    parameter integer  AXI_ID_WIDTH                 = 1,
    parameter integer  WSTRB_W                      = AXI_DATA_WIDTH/8,
  // Instruction Mem
    parameter integer  IMEM_ADDR_WIDTH              = 12
) (
    input  wire                                         clk,
    input  wire                                         reset,
    
    output wire                                         done_block,

  // controller <-> compute handshakes
    output wire                                         tag_flush,
    output wire                                         tag_req,
    output wire                                         ibuf_tag_reuse,
    output wire                                         obuf_tag_reuse,
    output wire                                         wbuf_tag_reuse,
    output wire                                         bias_tag_reuse, 
    input  wire                                         tag_ready,
    input  wire                                         ibuf_tag_done,
    input  wire                                         wbuf_tag_done,
    input  wire                                         obuf_tag_done,
    input  wire                                         bias_tag_done,

    input  wire                                         compute_done,
    output wire [ INST_GROUP_ID_W      -1 : 0 ]         cfg_curr_group_id,
    output wire [ INST_GROUP_ID_W      -1 : 0 ]         next_group_id,

  // Load/Store addresses
    // Bias load address
    output wire  [ BBUF_ADDR_WIDTH      -1 : 0 ]        bbuf_ld_addr,
    output wire                                         bbuf_ld_addr_v,
    // IBUF load address
    output wire  [ IBUF_ADDR_WIDTH      -1 : 0 ]        ibuf_ld_addr,
    output wire                                         ibuf_ld_addr_v,
    // WBUF load address
    output wire  [ WBUF_ADDR_WIDTH      -1 : 0 ]        wbuf_ld_addr,
    output wire                                         wbuf_ld_addr_v,
    // OBUF load/store address
    output wire  [ OBUF_ADDR_WIDTH      -1 : 0 ]        obuf_ld_addr,
    output wire                                         obuf_ld_addr_v,
    output wire  [ OBUF_ADDR_WIDTH      -1 : 0 ]        obuf_st_addr,
    output wire                                         obuf_st_addr_v,

  // Load bias or obuf
    output wire                                         tag_first_ic_outer_loop_ld,
    output wire                                         tag_ddr_pe_sw,

 // Systolic Array
    output wire                                         sa_compute_req,
    
    input  wire                                         ibuf_compute_ready,
    input  wire                                         wbuf_compute_ready,
    input  wire                                         obuf_compute_ready,
    input  wire                                         bbuf_compute_ready,


  // PCIe -> CL_wrapper AXI4-Lite interface
    // Slave Write address
    input  wire                                         pci_cl_ctrl_awvalid,
    input  wire  [ CTRL_ADDR_WIDTH      -1 : 0 ]        pci_cl_ctrl_awaddr,
    output wire                                         pci_cl_ctrl_awready,
    // Slave Write data
    input  wire                                         pci_cl_ctrl_wvalid,
    input  wire  [ CTRL_DATA_WIDTH      -1 : 0 ]        pci_cl_ctrl_wdata,
    input  wire  [ CTRL_WSTRB_WIDTH     -1 : 0 ]        pci_cl_ctrl_wstrb,
    output wire                                         pci_cl_ctrl_wready,
    // Slave Write response
    output wire                                         pci_cl_ctrl_bvalid,
    output wire  [ 2                    -1 : 0 ]        pci_cl_ctrl_bresp,
    input  wire                                         pci_cl_ctrl_bready,
    // Slave Read address
    input  wire                                         pci_cl_ctrl_arvalid,
    input  wire  [ CTRL_ADDR_WIDTH      -1 : 0 ]        pci_cl_ctrl_araddr,
    output wire                                         pci_cl_ctrl_arready,
    // Slave Read data/response
    output wire                                         pci_cl_ctrl_rvalid,
    output wire  [ CTRL_DATA_WIDTH      -1 : 0 ]        pci_cl_ctrl_rdata,
    output wire  [ 2                    -1 : 0 ]        pci_cl_ctrl_rresp,
    input  wire                                         pci_cl_ctrl_rready,


  // CL_wrapper -> DDR AXI4 interface
    // Master Interface Write Address
    output wire  [ ADDR_WIDTH           -1 : 0 ]        imem_awaddr,
    output wire  [ AXI_BURST_WIDTH      -1 : 0 ]        imem_awlen,
    output wire  [ 3                    -1 : 0 ]        imem_awsize,
    output wire  [ 2                    -1 : 0 ]        imem_awburst,
    output wire                                         imem_awvalid,
    input  wire                                         imem_awready,
    // Master Interface Write Data
    output wire  [ AXI_DATA_WIDTH       -1 : 0 ]        imem_wdata,
    output wire  [ WSTRB_W              -1 : 0 ]        imem_wstrb,
    output wire                                         imem_wlast,
    output wire                                         imem_wvalid,
    input  wire                                         imem_wready,
    // Master Interface Write Response
    input  wire  [ 2                    -1 : 0 ]        imem_bresp,
    input  wire                                         imem_bvalid,
    output wire                                         imem_bready,
    // Master Interface Read Address
    output wire  [ ADDR_WIDTH           -1 : 0 ]        imem_araddr,
    output wire  [ AXI_BURST_WIDTH      -1 : 0 ]        imem_arlen,
    output wire  [ 3                    -1 : 0 ]        imem_arsize,
    output wire  [ 2                    -1 : 0 ]        imem_arburst,
    output wire                                         imem_arvalid,
    output wire  [ AXI_ID_WIDTH         -1 : 0 ]        imem_arid,
    input  wire                                         imem_arready,
    // Master Interface Read Data
    input  wire  [ AXI_DATA_WIDTH       -1 : 0 ]        imem_rdata,
    input  wire  [ 2                    -1 : 0 ]        imem_rresp,
    input  wire                                         imem_rlast,
    input  wire                                         imem_rvalid,
    input  wire  [ AXI_ID_WIDTH         -1 : 0 ]        imem_rid,
    output wire                                         imem_rready,

  // Programming interface
    // Loop iterations
    output wire  [ LOOP_ITER_W          -1 : 0 ]        cfg_loop_iter,
    output wire  [ LOOP_ID_W            -1 : 0 ]        cfg_loop_iter_loop_id,
    output wire                                         cfg_loop_iter_v,
    output wire                                         cfg_set_specific_loop_v,
    output wire  [ LOOP_ID_W            -1 : 0 ]        cfg_set_specific_loop_loop_id,
    output wire  								        cfg_set_specific_loop_loop_param,
    // Loop stride
    output wire  [ ADDR_STRIDE_W        -1 : 0 ]        cfg_loop_stride,
    output wire                                         cfg_loop_stride_v,
    output wire  [ BUF_TYPE_W           -1 : 0 ]        cfg_loop_stride_id,
    output wire  [ 2                    -1 : 0 ]        cfg_loop_stride_type,
    output wire  [ LOOP_ID_W            -1 : 0 ]        cfg_loop_stride_loop_id,
    // Memory request
    output wire  [ MEM_REQ_W            -1 : 0 ]        cfg_mem_req_size,
    output wire                                         cfg_mem_req_v,
    output wire  [ 2                    -1 : 0 ]        cfg_mem_req_type,
    output wire  [ BUF_TYPE_W           -1 : 0 ]        cfg_mem_req_id,
    output wire  [ LOOP_ID_W            -1 : 0 ]        cfg_mem_req_loop_id,
    
    output wire  [ INST_GROUP_ID_W      -1 : 0 ]        inst_group_id,
    output wire                                         inst_group_type,
    output wire                                         inst_group_s_e,
    output wire                                         inst_group_v,
    output wire                                         inst_group_last,
    
    output wire  [ INST_DATA_WIDTH      -1 : 0 ]        cfg_simd_inst, // instructions for SIMD
    output wire                                         cfg_simd_inst_v,  // instructions for SIMD   

    input  wire                                         simd_group_done,
    // For now, not used
    input  wire  [ INST_GROUP_ID_W      -1 : 0 ]        simd_group_done_id,
    input  wire                                         simd_tiles_done,
    output wire                                         ctrl_simd_start,
    output wire                                         fused_sa_simd,
    output wire                                         simd_ready
    
  );

//=============================================================
// Localparam
//=============================================================

  // GeneSys Instruction Groups and Sequence of Execution
  // For now we just support these three case: SA, SIMD, SA-SIMD (one group from each type)

  // GeneSys controller states
    localparam integer  IDLE                         = 0;
    localparam integer  GET_FIRST_INST_BLOCK         = 1;
    localparam integer  DECODE                       = 2;
    localparam integer  SA                           = 3;
    localparam integer  SA_DONE_WAIT                 = 4;
    localparam integer  SIMD                         = 5;
    localparam integer  SIMD_DONE_WAIT               = 6;
    localparam integer  BLOCK_DONE                   = 7;
    localparam integer  DONE                         = 8;

    localparam integer  TM_IDLE                      = 0;
    localparam integer  TM_REQUEST                   = 1;
    localparam integer  TM_CHECK                     = 2;
    localparam integer  TM_WAIT                      = 3;
    localparam integer  TM_FLUSH                     = 4;
    
    localparam integer SA_GROUP                      = 0;
    localparam integer SIMD_GROUP                    = 1;
    localparam integer GROUP_START                   = 0;
    localparam integer GROUP_END                     = 1;
//=============================================================

//=============================================================
// Wires/Regs
//=============================================================


  // GeneSys states
    reg  [ 4                    -1 : 0 ]        genesys_state_d;
    reg  [ 4                    -1 : 0 ]        genesys_state_q;
    reg  [ 4                    -1 : 0 ]        genesys_state_qq;
    wire [ 4                    -1 : 0 ]        genesys_state;
    
    reg                                         sa_group_v;
    reg                                         simd_group_v;
    wire                                        sa_simd_group_v;
    
    reg  [ INST_GROUP_ID_W      -1 : 0 ]        sa_group_id;
    reg  [ INST_GROUP_ID_W      -1 : 0 ]        simd_group_id;
    
    wire                                        chip_start;

    wire                                        genesys_done;


  // DECODER
    wire [ INST_DATA_WIDTH      -1 : 0 ]        imem_read_data;
    wire                                        imem_read_valid;
    wire [ IMEM_ADDR_WIDTH      -1 : 0 ]        imem_read_addr;
    wire                                        imem_read_req;
    wire                                        imem_base_addr_valid;
    wire                                        decoder_start;
    wire                                        decoder_done;
    wire                                        base_loop_ctrl_start;
    wire                                        _base_loop_ctrl_start;
    wire                                        base_loop_ctrl_done;
    wire                                        base_loop_ctrl_stall;
    wire                                        block_done;
    wire                                        last_block;
    wire                                        imem_block_ready;
    wire                                        imem_rd_block_done;
    wire                                        _cfg_set_specific_loop_v;
    wire  [ LOOP_ID_W            -1 : 0 ]       _cfg_set_specific_loop_loop_id;
    wire  								        _cfg_set_specific_loop_loop_param;
    wire                                        imem_ld_req_valid;
    wire [ IMM_WIDTH            -1 : 0 ]        imem_ld_req_size;
    wire [ ADDR_WIDTH           -1 : 0 ]        ibuf_base_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        wbuf_base_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        obuf_base_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        bbuf_base_addr;
    wire [ ADDR_WIDTH           -1 : 0 ]        imem_base_addr;  
    
// Instruction Memory
    wire [ ADDR_WIDTH           -1 : 0 ]        imem_slave_ld_addr;
    wire [ MEM_REQ_W            -1 : 0 ]        imem_slave_ld_req_size;
    wire                                        imem_slave_ld_req_in;
    
    wire                                        imem_start;


// BASE ADDR GEN
    wire [ INST_GROUP_ID_W      -1 : 0 ]        _cfg_curr_group_id;
    wire [ INST_GROUP_ID_W      -1 : 0 ]        _next_group_id;
    wire [ NUM_MAX_LOOPS        -1 : 0 ]        group_first_iter;


  // resetn for axi slave
    wire                                        resetn;

  // Slave registers
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg0_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg0_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg1_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg1_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg2_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg2_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg3_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg3_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg4_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg4_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg5_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg5_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg6_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg6_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg7_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg7_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg8_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg8_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg9_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg9_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg10_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg10_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg11_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg11_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg12_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg12_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg13_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg13_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg14_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg14_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg15_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg15_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg16_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg16_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg17_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg17_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg18_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg18_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg19_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg19_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg20_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg20_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg21_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg21_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg22_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg22_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg23_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg23_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg24_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg24_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg25_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg25_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg26_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg26_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg27_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg27_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg28_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg28_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg29_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg29_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg30_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg30_out;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg31_in;
    wire [ CTRL_DATA_WIDTH      -1 : 0 ]        slv_reg31_out;
  // Slave registers end

  // Accelerator start logic
    wire                                        start_bit_d;
    reg                                         start_bit_q;

  // TM State
    reg  [ 3                    -1 : 0 ]        tm_state;
    reg  [ 3                    -1 : 0 ]        tm_state_d;
    reg  [ 3                    -1 : 0 ]        tm_state_q;
    reg  [ 3                    -1 : 0 ]        tm_state_qq;

    reg                                         tm_ibuf_tag_reuse;
    reg                                         tm_obuf_tag_reuse;
    reg                                         tm_wbuf_tag_reuse;
    reg                                         tm_bbuf_tag_reuse;

    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_ibuf_tag_addr_d;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_ibuf_tag_addr_q;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_obuf_tag_addr_d;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_obuf_tag_addr_q;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_wbuf_tag_addr_d;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_wbuf_tag_addr_q;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_bias_tag_addr_d;
    reg  [ ADDR_WIDTH           -1 : 0 ]        tm_bias_tag_addr_q;

    wire                                        tm_start;
	wire 										base_ctrl_tag_ready;
  always @(posedge clk)
  begin
    if(reset)
      tm_state_q <= TM_IDLE;
    else
      tm_state_q <= tm_state_d;
  end

  always @(posedge clk)
  begin
    if(reset) begin
      tm_ibuf_tag_addr_q  <= 0;
      tm_obuf_tag_addr_q  <= 0;
      tm_wbuf_tag_addr_q  <= 0;
      tm_bias_tag_addr_q  <= 0;
    end else begin
      tm_ibuf_tag_addr_q  <= tm_ibuf_tag_addr_d;
      tm_obuf_tag_addr_q  <= tm_obuf_tag_addr_d;
      tm_wbuf_tag_addr_q  <= tm_wbuf_tag_addr_d;
      tm_bias_tag_addr_q  <= tm_bias_tag_addr_d;
    end
  end

  always @(*)
  begin
    tm_state_d = tm_state_q;
    
    tm_ibuf_tag_addr_d = tm_ibuf_tag_addr_q;
    tm_obuf_tag_addr_d = tm_obuf_tag_addr_q;
    tm_wbuf_tag_addr_d = tm_wbuf_tag_addr_q;
    tm_bias_tag_addr_d = tm_bias_tag_addr_q;
    
    case(tm_state_q)
      TM_IDLE: begin
        if (tm_start && tag_ready)
          tm_state_d = TM_REQUEST;
      end
      TM_REQUEST: begin
             tm_state_d = TM_CHECK;         
     end    
      TM_CHECK: begin
        if (tag_ready && ibuf_ld_addr_v && obuf_ld_addr_v && wbuf_ld_addr_v && bbuf_ld_addr_v) begin
          tm_ibuf_tag_reuse = tm_ibuf_tag_addr_q == ibuf_ld_addr;
          tm_obuf_tag_reuse = tm_obuf_tag_addr_q == obuf_ld_addr;
          tm_wbuf_tag_reuse = tm_wbuf_tag_addr_q == wbuf_ld_addr;
          tm_bbuf_tag_reuse = tm_bias_tag_addr_q == bbuf_ld_addr;
          
          tm_state_d = TM_WAIT;
        end
      end
      TM_WAIT: begin
          tm_ibuf_tag_addr_d = ibuf_ld_addr;
          tm_obuf_tag_addr_d = obuf_ld_addr;
          tm_wbuf_tag_addr_d = wbuf_ld_addr; 
          tm_bias_tag_addr_d = bbuf_ld_addr;
          if (genesys_state == SA_DONE_WAIT)
             tm_state_d = TM_FLUSH;
          else if (tag_ready)
             tm_state_d = TM_REQUEST;            
      end      
      TM_FLUSH: begin
        tm_state_d = TM_IDLE;
      end
    endcase
  end


    assign tag_req = tm_state_q == TM_CHECK;

    assign tag_flush = tm_state_q == TM_FLUSH;

    assign ibuf_tag_reuse = tm_ibuf_tag_reuse;
    assign obuf_tag_reuse = tm_obuf_tag_reuse;
    assign wbuf_tag_reuse = tm_wbuf_tag_reuse;
    assign bias_tag_reuse = tm_bbuf_tag_reuse;

    assign base_ctrl_tag_ready = tag_ready && tm_state_q == TM_REQUEST;
//=============================================================

//=============================================================
// Accelerator Start logic
//=============================================================
  always @(posedge clk)
  begin
    if (reset)
      start_bit_q <= 1'b0;
    else
      start_bit_q <= start_bit_d;
  end
//=============================================================

//=============================================================
// FSM
//=============================================================
  always @(posedge clk)
  begin
    if (reset) begin
      genesys_state_q <= IDLE;
    end
    else begin
      genesys_state_q <= genesys_state_d;
    end
  end

  always @(*)
  begin
    genesys_state_d = genesys_state_q;
    case(genesys_state_q)
      IDLE: begin
        if (chip_start) begin
          genesys_state_d = GET_FIRST_INST_BLOCK;
        end
      end
      GET_FIRST_INST_BLOCK: begin
        if (imem_block_ready) begin
           genesys_state_d = DECODE; 
        end 
      end    
      DECODE: begin
        if (decoder_done) begin
           if (sa_group_v) 
              genesys_state_d = SA;
           else if (simd_group_v)
              genesys_state_d = SIMD;   
        end
      end
      SA: begin
         if (base_loop_ctrl_done) 
             genesys_state_d = SA_DONE_WAIT;
      end
      SA_DONE_WAIT: begin
         if (ibuf_tag_done && wbuf_tag_done && obuf_tag_done && bias_tag_done)
             if (sa_simd_group_v)
                 genesys_state_d = SIMD_DONE_WAIT;
             else
                 genesys_state_d = BLOCK_DONE;
      end
      SIMD: begin
         if (simd_group_done)
             genesys_state_d = SIMD_DONE_WAIT;
      end
      SIMD_DONE_WAIT: begin
         if (sa_simd_group_v) begin
            if (simd_group_done)
                genesys_state_d = BLOCK_DONE; 
         end
         else begin
            if (simd_tiles_done)
                genesys_state_d = BLOCK_DONE;
            else
                genesys_state_d = SIMD;
         end
      end
      BLOCK_DONE: begin
         if (~last_block) 
             genesys_state_d = DECODE;
         else
             genesys_state_d = DONE;
      end
      DONE: begin
         genesys_state_d = IDLE; 
      end
  endcase
  end
      
      
    assign block_done = genesys_state == BLOCK_DONE;
    assign genesys_done = genesys_state == DONE;
//=============================================================
  // GROUPs and Sequence Configuration
//=============================================================
  wire                                      _sa_group_v;
  wire                                      _simd_group_v;
  
  reg  [ INST_GROUP_ID_W         -1 : 0]    _sa_group_counter;
  
  assign _sa_group_v = (inst_group_type == SA_GROUP && inst_group_s_e == GROUP_START && inst_group_v);
  assign _simd_group_v = (inst_group_type == SIMD_GROUP && inst_group_s_e == GROUP_START && inst_group_v);
  
  always @ (posedge clk) begin
    if (reset) begin
       sa_group_v <= 1'b0;
       simd_group_v <= 1'b0;
       sa_group_id <= 0;
       simd_group_id <= 0;  
    end
    else if (_sa_group_v) begin
       sa_group_v <= 1'b1;
       sa_group_id <= inst_group_id;
    end
    else if (_simd_group_v) begin
       simd_group_v <= 1'b1;
       simd_group_id <= inst_group_id;
    end
  end  
  
  assign sa_simd_group_v = sa_group_v && simd_group_v;
  assign fused_sa_simd = sa_simd_group_v;
  
  always @(posedge clk) begin
     if (reset) 
        _sa_group_counter <= 0; 
     else if (_sa_group_v)
        _sa_group_counter <= _sa_group_counter + 1'b1;
     else if (block_done)
        _sa_group_counter <= 0;
  end
  
//=============================================================
// Assigns
//============================================================
    // We use the first register to start the chip
    // We use the second and third register to get the infor about the instruction blocks
    assign resetn = ~reset;
    assign genesys_state = genesys_state_q;
    
    assign start_bit_d = slv_reg0_out[0];
    assign chip_start = (start_bit_q ^ start_bit_d) && genesys_state_q == IDLE;
    
    assign imem_slave_ld_addr = slv_reg1_out;
    assign imem_slave_ld_req_size = slv_reg2_out;
    
    assign imem_start = chip_start;
    
    assign imem_slave_ld_req_in = (genesys_state == GET_FIRST_INST_BLOCK) && (genesys_state_qq != GET_FIRST_INST_BLOCK);
    
    always @(posedge clk) begin
       if (reset) begin
          genesys_state_qq <= 0;
          tm_state_qq <= 0;
       end
       else begin
          genesys_state_qq <= genesys_state_q; 
          tm_state_qq <= tm_state_q; 
       end
    end
    
    assign decoder_start = (genesys_state == DECODE && genesys_state_qq == GET_FIRST_INST_BLOCK) || (genesys_state == DECODE && genesys_state_qq == BLOCK_DONE);
    
    assign tm_start = (genesys_state == SA) && (genesys_state_qq == DECODE);
    
    assign base_loop_ctrl_start = (tm_state_q == TM_REQUEST) && (tm_state_qq == TM_IDLE);
    assign base_loop_ctrl_stall = tm_state_q != TM_REQUEST;
    
    // The below logic is just for the case that we have one SA.
    //TODO: for later, that we support multiple groups we need to revisit this
    assign _next_group_id = _sa_group_counter;
    assign cfg_curr_group_id = _cfg_curr_group_id;
    assign next_group_id = _next_group_id;
    
    reg                                 _group_first_iter_v;
    always @(posedge clk) begin
       if  (reset)
           _group_first_iter_v <= 1'b0;
       else if (genesys_state == DECODE && sa_group_v)
           _group_first_iter_v <= 1'b1;
       else if (tm_state_d == TM_WAIT)
           _group_first_iter_v <= 1'b0;
    end
    assign group_first_iter[1] = _group_first_iter_v;
    //
    
    assign ctrl_simd_start = ((genesys_state == SIMD) && (genesys_state_qq == DECODE)) || ((genesys_state == SIMD) && genesys_state_qq == SIMD_DONE_WAIT) && simd_ready;
    
    assign sa_compute_req = ibuf_compute_ready && wbuf_compute_ready && obuf_compute_ready && bbuf_compute_ready;
    
    assign done_block = block_done;
//=============================================================


//=============================================================
// Status/Control AXI4-Lite
//=============================================================

  axi4lite_slave #(
    .AXIS_ADDR_WIDTH                ( CTRL_ADDR_WIDTH                ),
    .AXIS_DATA_WIDTH                ( CTRL_DATA_WIDTH                )
  ) status_ctrl_slv   (
    .clk                            ( clk                            ),
    .resetn                         ( resetn                         ),
  // Slave registers
    .slv_reg0_in                    ( slv_reg0_in                    ),
    .slv_reg0_out                   ( slv_reg0_out                   ),
    .slv_reg1_in                    ( slv_reg1_in                    ),
    .slv_reg1_out                   ( slv_reg1_out                   ),
    .slv_reg2_in                    ( slv_reg2_in                    ),
    .slv_reg2_out                   ( slv_reg2_out                   ),
    .slv_reg3_in                    ( slv_reg3_in                    ),
    .slv_reg3_out                   ( slv_reg3_out                   ),
    .slv_reg4_in                    ( slv_reg4_in                    ),
    .slv_reg4_out                   ( slv_reg4_out                   ),
    .slv_reg5_in                    ( slv_reg5_in                    ),
    .slv_reg5_out                   ( slv_reg5_out                   ),
    .slv_reg6_in                    ( slv_reg6_in                    ),
    .slv_reg6_out                   ( slv_reg6_out                   ),
    .slv_reg7_in                    ( slv_reg7_in                    ),
    .slv_reg7_out                   ( slv_reg7_out                   ),
    .slv_reg8_in                    ( slv_reg8_in                    ),
    .slv_reg8_out                   ( slv_reg8_out                   ),
    .slv_reg9_in                    ( slv_reg9_in                    ),
    .slv_reg9_out                   ( slv_reg9_out                   ),
    .slv_reg10_in                   ( slv_reg10_in                   ),
    .slv_reg10_out                  ( slv_reg10_out                  ),
    .slv_reg11_in                   ( slv_reg11_in                   ),
    .slv_reg11_out                  ( slv_reg11_out                  ),
    .slv_reg12_in                   ( slv_reg12_in                   ),
    .slv_reg12_out                  ( slv_reg12_out                  ),
    .slv_reg13_in                   ( slv_reg13_in                   ),
    .slv_reg13_out                  ( slv_reg13_out                  ),
    .slv_reg14_in                   ( slv_reg14_in                   ),
    .slv_reg14_out                  ( slv_reg14_out                  ),
    .slv_reg15_in                   ( slv_reg15_in                   ),
    .slv_reg15_out                  ( slv_reg15_out                  ),

    .slv_reg16_in                   ( slv_reg16_in                   ),
    .slv_reg16_out                  ( slv_reg16_out                  ),
    .slv_reg17_in                   ( slv_reg17_in                   ),
    .slv_reg17_out                  ( slv_reg17_out                  ),
    .slv_reg18_in                   ( slv_reg18_in                   ),
    .slv_reg18_out                  ( slv_reg18_out                  ),
    .slv_reg19_in                   ( slv_reg19_in                   ),
    .slv_reg19_out                  ( slv_reg19_out                  ),
    .slv_reg20_in                   ( slv_reg20_in                   ),
    .slv_reg20_out                  ( slv_reg20_out                  ),
    .slv_reg21_in                   ( slv_reg21_in                   ),
    .slv_reg21_out                  ( slv_reg21_out                  ),
    .slv_reg22_in                   ( slv_reg22_in                   ),
    .slv_reg22_out                  ( slv_reg22_out                  ),
    .slv_reg23_in                   ( slv_reg23_in                   ),
    .slv_reg23_out                  ( slv_reg23_out                  ),
    .slv_reg24_in                   ( slv_reg24_in                   ),
    .slv_reg24_out                  ( slv_reg24_out                  ),
    .slv_reg25_in                   ( slv_reg25_in                   ),
    .slv_reg25_out                  ( slv_reg25_out                  ),
    .slv_reg26_in                   ( slv_reg26_in                   ),
    .slv_reg26_out                  ( slv_reg26_out                  ),
    .slv_reg27_in                   ( slv_reg27_in                   ),
    .slv_reg27_out                  ( slv_reg27_out                  ),
    .slv_reg28_in                   ( slv_reg28_in                   ),
    .slv_reg28_out                  ( slv_reg28_out                  ),
    .slv_reg29_in                   ( slv_reg29_in                   ),
    .slv_reg29_out                  ( slv_reg29_out                  ),
    .slv_reg30_in                   ( slv_reg30_in                   ),
    .slv_reg30_out                  ( slv_reg30_out                  ),
    .slv_reg31_in                   ( slv_reg31_in                   ),
    .slv_reg31_out                  ( slv_reg31_out                  ),

    .s_axi_awaddr                   ( pci_cl_ctrl_awaddr             ),
    .s_axi_awvalid                  ( pci_cl_ctrl_awvalid            ),
    .s_axi_awready                  ( pci_cl_ctrl_awready            ),
    .s_axi_wdata                    ( pci_cl_ctrl_wdata              ),
    .s_axi_wstrb                    ( pci_cl_ctrl_wstrb              ),
    .s_axi_wvalid                   ( pci_cl_ctrl_wvalid             ),
    .s_axi_wready                   ( pci_cl_ctrl_wready             ),
    .s_axi_bresp                    ( pci_cl_ctrl_bresp              ),
    .s_axi_bvalid                   ( pci_cl_ctrl_bvalid             ),
    .s_axi_bready                   ( pci_cl_ctrl_bready             ),
    .s_axi_araddr                   ( pci_cl_ctrl_araddr             ),
    .s_axi_arvalid                  ( pci_cl_ctrl_arvalid            ),
    .s_axi_arready                  ( pci_cl_ctrl_arready            ),
    .s_axi_rdata                    ( pci_cl_ctrl_rdata              ),
    .s_axi_rresp                    ( pci_cl_ctrl_rresp              ),
    .s_axi_rvalid                   ( pci_cl_ctrl_rvalid             ),
    .s_axi_rready                   ( pci_cl_ctrl_rready             )
  );
//=============================================================

//=============================================================
// Instruction Memory
//=============================================================
  instruction_memory_wrapper #(
    .MEM_REQ_W                      ( MEM_REQ_W                      ),
    .AXI_ADDR_WIDTH                 ( ADDR_WIDTH                     ),
    .AXI_DATA_WIDTH                 ( AXI_DATA_WIDTH                 ),
    .AXI_BURST_WIDTH                ( AXI_BURST_WIDTH                ),
    .INST_DATA_WIDTH                ( INST_DATA_WIDTH                ),
    .INST_ADDR_WIDTH                ( IMEM_ADDR_WIDTH                )    
  ) inst_memory (
    .clk                            ( clk                            ),
    .reset                          ( reset                          ),
    .start                          ( imem_start                     ),
    .imem_rd_req                    ( imem_read_req                  ),
    .imem_rd_addr                   ( imem_read_addr                 ),
    .imem_rd_block_done             ( imem_rd_block_done             ),
    .imem_rd_data                   ( imem_read_data                 ),
    .imem_rd_valid                  ( imem_read_valid                ),
    .imem_block_ready               ( imem_block_ready               ),
    .imem_awaddr                    ( imem_awaddr                    ),
    .imem_awlen                     ( imem_awlen                     ),
    .imem_awsize                    ( imem_awsize                    ),
    .imem_awburst                   ( imem_awburst                   ),
    .imem_awvalid                   ( imem_awvalid                   ),
    .imem_awready                   ( imem_awready                   ),
    .imem_wdata                     ( imem_wdata                     ),
    .imem_wstrb                     ( imem_wstrb                     ),
    .imem_wlast                     ( imem_wlast                     ),
    .imem_wvalid                    ( imem_wvalid                    ),
    .imem_wready                    ( imem_wready                    ),
    .imem_bresp                     ( imem_bresp                     ),
    .imem_bvalid                    ( imem_bvalid                    ),
    .imem_bready                    ( imem_bready                    ),
    .imem_araddr                    ( imem_araddr                    ),
    .imem_arlen                     ( imem_arlen                     ),
    .imem_arsize                    ( imem_arsize                    ),
    .imem_arburst                   ( imem_arburst                   ),
    .imem_arvalid                   ( imem_arvalid                   ),
    .imem_arid                      ( imem_arid                      ),
    .imem_arready                   ( imem_arready                   ),
    .imem_rdata                     ( imem_rdata                     ),
    .imem_rresp                     ( imem_rresp                     ),
    .imem_rlast                     ( imem_rlast                     ),
    .imem_rvalid                    ( imem_rvalid                    ),
    .imem_rid                       ( imem_rid                       ),
    .imem_rready                    ( imem_rready                    ),
    .slave_ld_addr                  ( imem_slave_ld_addr             ),
    .slave_ld_req_size              ( imem_slave_ld_req_size         ),
    .slave_ld_req_in                ( imem_slave_ld_req_in           ),
    .decoder_ld_addr                ( imem_base_addr                 ),
    .decoder_ld_req_size            ( imem_ld_req_size               ),
    .decoder_ld_req_in              ( imem_ld_req_valid              )
  );
//=============================================================


//=============================================================
// Decoder
//=============================================================
  decoder #(
    .IMEM_ADDR_W                    ( IMEM_ADDR_WIDTH                ),
    .DDR_ADDR_W                     ( ADDR_WIDTH                     ),
    .INST_W                         ( INST_DATA_WIDTH                ),
    .BUF_TYPE_W                     ( BUF_TYPE_W                     ),
    .IMM_WIDTH                      ( IMM_WIDTH                      ),
    .OP_CODE_W                      ( OP_CODE_W                      ),
    .OP_SPEC_W                      ( OP_SPEC_W                      ),
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
    .INST_GROUP_ID_W                ( INST_GROUP_ID_W                ),
    .WAIT_CYCLES                    ( EXEC_DONE_WAIT_CYCLES          )             
    
  ) instruction_decoder (
    .clk                            ( clk                            ), 
    .reset                          ( reset                          ), 
    .imem_read_data                 ( imem_read_data                 ),
    .imem_read_valid                ( imem_read_valid                ),
    .imem_read_addr                 ( imem_read_addr                 ),
    .imem_read_req                  ( imem_read_req                  ),
    .imem_base_addr_valid           ( imem_base_addr_valid           ),
    .start                          ( decoder_start                  ),
    .done                           ( decoder_done                   ),
    .loop_ctrl_start                ( _base_loop_ctrl_start          ),
    .block_done                     ( block_done                     ),
    .last_block                     ( last_block                     ),
    .next_block_ready               ( imem_block_ready               ),
    .imem_rd_block_done             ( imem_rd_block_done             ),
    .cfg_loop_iter_v                ( cfg_loop_iter_v                ),
    .cfg_loop_iter                  ( cfg_loop_iter                  ),
    .cfg_loop_iter_loop_id          ( cfg_loop_iter_loop_id          ),
    .cfg_set_specific_loop_v        ( cfg_set_specific_loop_v        ),
    .cfg_set_specific_loop_loop_id  ( cfg_set_specific_loop_loop_id  ),
    .cfg_set_specific_loop_loop_param (cfg_set_specific_loop_loop_param ),
    .cfg_loop_stride_v              ( cfg_loop_stride_v              ),
    .cfg_loop_stride_type           ( cfg_loop_stride_type           ),
    .cfg_loop_stride                ( cfg_loop_stride                ),
    .cfg_loop_stride_loop_id        ( cfg_loop_stride_loop_id        ),
    .cfg_loop_stride_id             ( cfg_loop_stride_id             ),
    .cfg_mem_req_v                  ( cfg_mem_req_v                  ),
    .cfg_mem_req_size               ( cfg_mem_req_size               ),
    .cfg_mem_req_type               ( cfg_mem_req_type               ),
    .cfg_mem_req_loop_id            ( cfg_mem_req_loop_id            ),
    .cfg_mem_req_id                 ( cfg_mem_req_id                 ),
    .imem_ld_req_valid              ( imem_ld_req_valid              ),
    .imem_ld_req_size               ( imem_ld_req_size               ),
    .ibuf_base_addr                 ( ibuf_base_addr                 ),
    .wbuf_base_addr                 ( wbuf_base_addr                 ),
    .obuf_base_addr                 ( obuf_base_addr                 ),
    .bias_base_addr                 ( bbuf_base_addr                 ),
    .imem_base_addr                 ( imem_base_addr                 ),
    .inst_group_id                  ( inst_group_id                  ),
    .inst_group_type                ( inst_group_type                ),
    .inst_group_s_e                 ( inst_group_s_e                 ),
    .inst_group_v                   ( inst_group_v                   ),
    .inst_group_last                ( inst_group_last                ),
    .cfg_simd_inst                  ( cfg_simd_inst                  ),
    .cfg_simd_inst_v                ( cfg_simd_inst_v                )
  );
//=============================================================

 assign _cfg_set_specific_loop_v = cfg_set_specific_loop_v;
 assign _cfg_set_specific_loop_loop_id = cfg_set_specific_loop_loop_id;
 assign _cfg_set_specific_loop_loop_param  = cfg_set_specific_loop_loop_param;

//=============================================================
// Base address generator
//    This module is in charge of the outer loops (base loops for tiles)
//=============================================================
  base_addr_gen #(
    .ADDR_WIDTH                     ( ADDR_WIDTH                     ),
    .LOOP_ITER_W                    ( LOOP_ITER_W                    ),
    .ADDR_STRIDE_W                  ( ADDR_STRIDE_W                  ),
    .LOOP_ID_W                      ( LOOP_ID_W                      ),
    .INST_GROUP_ID_W                ( INST_GROUP_ID_W                ),
    .BUF_TYPE_W                     ( BUF_TYPE_W                     ),
    .GROUP_ENABLED                  ( GROUP_ENABLED                  )
  ) base_ctrl (
    .clk                            ( clk                            ), //input
    .reset                          ( reset                          ), //input

    .start                          ( base_loop_ctrl_start           ), //input
    .done                           ( base_loop_ctrl_done            ), //output
    .block_done                     ( block_done                     ),
    
    .cfg_curr_group_id              ( _cfg_curr_group_id              ),
    .next_group_id                  ( _next_group_id                  ),
    .group_first_iter               ( group_first_iter               ),
    
    .stall                          ( base_loop_ctrl_stall           ),
    
    .cfg_loop_iter_v                ( cfg_loop_iter_v                ),
    .cfg_loop_iter                  ( cfg_loop_iter                  ),
    .cfg_loop_iter_loop_id          ( cfg_loop_iter_loop_id          ),
    .cfg_set_specific_loop_v        ( _cfg_set_specific_loop_v        ),
    .cfg_set_specific_loop_loop_id  (_cfg_set_specific_loop_loop_id   ),
    .cfg_set_specific_loop_loop_param (_cfg_set_specific_loop_loop_param),
    .cfg_loop_stride_v              ( cfg_loop_stride_v              ),
    .cfg_loop_stride                ( cfg_loop_stride                ),
    .cfg_loop_stride_loop_id        ( cfg_loop_stride_loop_id        ),
    .cfg_loop_stride_id             ( cfg_loop_stride_id             ),
    .cfg_loop_stride_type           ( cfg_loop_stride_type           ),
    .inst_group_id                  ( inst_group_id                  ),
    .inst_group_type                ( inst_group_type                ),
    .inst_group_s_e                 ( inst_group_s_e                 ),
    .inst_group_v                   ( inst_group_v                   ),
    .inst_group_last                ( inst_group_last                ),
    
    .obuf_base_addr                 ( obuf_base_addr                 ),
    .obuf_ld_addr                   ( obuf_ld_addr                   ),
    .obuf_ld_addr_v                 ( obuf_ld_addr_v                 ),
    .obuf_st_addr                   ( obuf_st_addr                   ),
    .obuf_st_addr_v                 ( obuf_st_addr_v                 ),  
    .ibuf_base_addr                 ( ibuf_base_addr                 ),
    .ibuf_ld_addr                   ( ibuf_ld_addr                   ),
    .ibuf_ld_addr_v                 ( ibuf_ld_addr_v                 ),  
    .bbuf_base_addr                 ( bbuf_base_addr                 ),
    .bbuf_ld_addr                   ( bbuf_ld_addr                   ),
    .bbuf_ld_addr_v                 ( bbuf_ld_addr_v                 ),  
    .wbuf_base_addr                 ( wbuf_base_addr                 ),
    .wbuf_ld_addr                   ( wbuf_ld_addr                   ),
    .wbuf_ld_addr_v                 ( wbuf_ld_addr_v                 ),     

    .first_ic_outer_loop_ld         ( tag_first_ic_outer_loop_ld     ), //output
    .ddr_pe_sw                      ( tag_ddr_pe_sw                  )  //output
  );
//=============================================================


endmodule
