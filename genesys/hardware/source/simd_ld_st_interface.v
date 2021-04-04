


module simd_ld_st_interface #(

parameter   VMEM1_MEM_ID                        = 0,
parameter   VMEM2_MEM_ID                        = 1,

//LD/ST
parameter   MEM_REQ_W                           = 16,
parameter   LOOP_ITER_W                         = 16,
parameter   IMM_WIDTH                           = 16,
parameter   ADDR_STRIDE_W                       = 16, 
parameter   SIMD_LOOP_ID_W                      = 5,
parameter   NUM_TAGS                            = 1,
parameter   TAG_W                               = $clog2(NUM_TAGS),
parameter   SIMD_DATA_WIDTH                     = 32,
parameter   LD_ST_HIGH_DATA_WIDTH               = SIMD_DATA_WIDTH,
parameter   LD_ST_LOW_DATA_WIDTH                = 8,
// AXI
parameter   AXI_ADDR_WIDTH                      = 42,
parameter   AXI_ID_WIDTH                        = 1,
parameter   AXI_DATA_WIDTH                      = 64,
parameter   AXI_BURST_WIDTH                     = 8,
parameter   WSTRB_W                             = AXI_DATA_WIDTH/8,  

parameter   NUM_SIMD_LANES                      = 16,
parameter   VMEM_BUF_ADDR_W                     = 16,
parameter   VMEM_TAG_BUF_ADDR_W                 = VMEM_BUF_ADDR_W + TAG_W,

parameter   GROUP_ID_W                          = 4,
parameter   MAX_NUM_GROUPS                      = (1<<GROUP_ID_W),
parameter   NS_ID_BITS                          = 3,
parameter   NS_INDEX_ID_BITS                    = 5,
parameter   OPCODE_BITS                         = 4,
parameter   FUNCTION_BITS                       = 4,
parameter   INSTRUCTION_WIDTH                   = OPCODE_BITS + FUNCTION_BITS + 3*(NS_ID_BITS + NS_INDEX_ID_BITS),

parameter   LD_ST_DATA_WIDTH                    = NS_INDEX_ID_BITS + 1,
parameter   LOOP_ID_W                           = 5,
parameter   BASE_ADDR_SEGMENT_W                 = 16,
parameter   ADDR_WIDTH                          = 32,

parameter   SIMD_LD_ST_HIGH_BW_GROUP_SIZE       = AXI_DATA_WIDTH / LD_ST_HIGH_DATA_WIDTH,
parameter   SIMD_LD_ST_HIGH_BW_NUM_GROUPS       = NUM_SIMD_LANES / SIMD_LD_ST_HIGH_BW_GROUP_SIZE,
parameter   SIMD_LD_ST_LOW_BW_GROUP_SIZE        = AXI_DATA_WIDTH / LD_ST_LOW_DATA_WIDTH,
parameter   SIMD_LD_ST_LOW_BW_NUM_GROUPS        = NUM_SIMD_LANES / SIMD_LD_ST_LOW_BW_GROUP_SIZE,
parameter integer  GROUP_ENABLED                = 0

)(
input  wire                                         clk,
input  wire                                         reset,
input  wire                                         block_done,
// Extracted filed instruction
input  wire  [OPCODE_BITS           -1:0]           opcode,
input  wire  [FUNCTION_BITS         -1:0]           fn,
input  wire  [NS_ID_BITS            -1:0]           dest_ns_id,
input  wire  [NS_INDEX_ID_BITS      -1:0]           dest_ns_index_id,  
input  wire  [NS_ID_BITS            -1:0]           src1_ns_id,
input  wire  [NS_INDEX_ID_BITS      -1:0]           src1_ns_index_id,    
input  wire  [NS_ID_BITS            -1:0]           src2_ns_id,
input  wire  [NS_INDEX_ID_BITS      -1:0]           src2_ns_index_id,

//
input  wire  [MAX_NUM_GROUPS        -1:0]           ld_config_done,
input  wire  [MAX_NUM_GROUPS        -1:0]           st_config_done,
input  wire  [GROUP_ID_W            -1:0]           ld_st_group_id,

output wire                                         ld_mem_simd_done,
output wire                                         st_mem_simd_done,

// VMEM1
output wire  [NUM_SIMD_LANES        -1:0]               vmem1_write_req,
output wire  [NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W-1:0]   vmem1_write_addr,
output wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH -1:0]      vmem1_write_data,
output wire  [NUM_SIMD_LANES        -1:0]               vmem1_read_req,
output wire  [NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W-1:0]   vmem1_read_addr,
input  wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH -1:0]      vmem1_read_data,
// VMEM2
output wire  [NUM_SIMD_LANES        -1:0]               vmem2_write_req,
output wire  [NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W-1:0]   vmem2_write_addr,
output wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH -1:0]      vmem2_write_data,
output wire  [NUM_SIMD_LANES        -1:0]               vmem2_read_req,
output wire  [NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W-1:0]   vmem2_read_addr,
input  wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH -1:0]      vmem2_read_data,

// Controller
output wire                                             simd_tiles_done,

// AXI
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
    localparam integer  LDMEM_GEN_BASE_ADDR          = 1;
    localparam integer  LDMEM_TILE_BUSY_NS_VMEM1     = 2;
    localparam integer  LDMEM_TILE_BUSY_NS_VMEM2     = 3;
    localparam integer  LDMEM_WAIT_0                 = 4;
    localparam integer  LDMEM_WAIT_1                 = 5;
    localparam integer  LDMEM_WAIT_2                 = 6;
    localparam integer  LDMEM_WAIT_3                 = 7;
    localparam integer  LDMEM_DONE                   = 78;

    localparam integer  STMEM_IDLE                   = 0;
    localparam integer  STMEM_GEN_BASE_ADDR          = 1;
    localparam integer  STMEM_TILE_BUSY_NS_VMEM1     = 2;
    localparam integer  STMEM_TILE_BUSY_NS_VMEM2     = 3;
    localparam integer  STMEM_WAIT_0                 = 4;
    localparam integer  STMEM_WAIT_1                 = 5;
    localparam integer  STMEM_WAIT_2                 = 6;
    localparam integer  STMEM_WAIT_3                 = 7;
    localparam integer  STMEM_DONE                   = 8;

    localparam integer  MEM_LD                       = 0;
    localparam integer  MEM_ST                       = 1;
    
    localparam integer  NS_VMEM_1                    = 1;
    localparam integer  NS_VMEM_2                    = 2;
    
    localparam integer  LD_CONFIG_BASE_ADDR          = 0;
    localparam integer  LD_CONFIG_BASE_LOOP_ITER     = 1;
    localparam integer  LD_CONFIG_BASE_LOOP_STRIDE   = 2;
    localparam integer  LD_CONFIG_TILE_LOOP_ITER     = 3;
    localparam integer  LD_CONFIG_TILE_LOOP_STRIDE   = 4;
    localparam integer  LD_START                     = 5;

    localparam integer  ST_CONFIG_BASE_ADDR          = 8;
    localparam integer  ST_CONFIG_BASE_LOOP_ITER     = 9;
    localparam integer  ST_CONFIG_BASE_LOOP_STRIDE   = 10;
    localparam integer  ST_CONFIG_TILE_LOOP_ITER     = 11;
    localparam integer  ST_CONFIG_TILE_LOOP_STRIDE   = 12;
    localparam integer  ST_START                     = 13;
    
    // This is for the maximum range of the number of groups
    localparam integer  GROUP_COUNTER_LD_ST_WIDTH = $clog2(SIMD_LD_ST_HIGH_BW_NUM_GROUPS);
    localparam integer  NUM_MAX_LOOPS = (1<< LOOP_ID_W);
//==============================================================================
// WIRE & REG
//==============================================================================
  // Signals for Programming     
    wire                                         ld_cfg_base_addr_v;
    wire                                         ld_cfg_base_loop_iter_v;
    wire                                         ld_cfg_base_loop_stride_v;
    wire                                         ld_cfg_tile_loop_iter_v;
    wire                                         ld_cfg_tile_loop_stride_v;
    wire                                         ld_start;
                                      
    wire                                         st_cfg_base_addr_v;
    wire                                         st_cfg_base_loop_iter_v;
    wire                                         st_cfg_base_loop_stride_v;
    wire                                         st_cfg_tile_loop_iter_v;
    wire                                         st_cfg_tile_loop_stride_v;
    wire                                         st_start;  
    
    wire                                         cfg_base_addr_segment;
    wire                                         cfg_stride_segment;
    wire  [ 2                    -1: 0 ]         cfg_ns_id;
    wire  [ LOOP_ID_W            -1: 0 ]         cfg_loop_id;
    
    wire  [ BASE_ADDR_SEGMENT_W  -1: 0 ]         cfg_base_addr;
    wire  [ ADDR_STRIDE_W        -1: 0 ]         cfg_loop_stride;
    wire  [ LOOP_ITER_W          -1: 0 ]         cfg_loop_iter;
    wire  [ MEM_REQ_W            -1: 0 ]         cfg_mem_req_size;
    wire  [ LD_ST_DATA_WIDTH     -1: 0 ]         cfg_ld_st_data_width;
    
    wire  [ GROUP_ID_W           -1: 0 ]         cfg_group_id;
    
    // FSM
    reg   [ 4                     -1: 0 ]        ldmem_state_d;
    reg   [ 4                     -1: 0 ]        ldmem_state_q;
    reg   [ 4                     -1: 0 ]        stmem_state_d;
    reg   [ 4                     -1: 0 ]        stmem_state_q;
    
    // Memory Request
    wire                                         ld_req_valid_d;
    reg                                          ld_req_valid_q;
    reg   [ MEM_REQ_W             -1: 0 ]        ld_req_size;
    reg   [ ADDR_WIDTH            -1: 0 ]        ld_req_addr;
    wire                                         st_req_valid_d;
    reg                                          st_req_valid_q;
    reg   [ MEM_REQ_W             -1: 0 ]        st_req_size;
    reg   [ ADDR_WIDTH            -1: 0 ]        st_req_addr;
    reg   [ LD_ST_DATA_WIDTH      -1: 0 ]        ld_data_width;
    reg   [ LD_ST_DATA_WIDTH      -1: 0 ]        st_data_width;    
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_ld_base_vmem1_iter_done;
    wire                                         mws_ld_base_vmem1_start; 
    wire                                         mws_ld_base_vmem1_stall;
    wire                                         mws_ld_base_vmem1_done; 
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_ld_base_vmem1_lsb;
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_ld_base_vmem1_msb;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_cfg_base_addr_vmem1;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_base_addr_vmem1;
    wire                                         mws_ld_cfg_base_addr_vmem1_v;
    wire                                         mws_ld_base_addr_out_vmem1_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_ld_base_vmem2_iter_done;
    wire                                         mws_ld_base_vmem2_start; 
    wire                                         mws_ld_base_vmem2_stall;
    wire                                         mws_ld_base_vmem2_done; 
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_ld_base_vmem2_lsb;
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_ld_base_vmem2_msb;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_cfg_base_addr_vmem2;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_base_addr_vmem2;
    wire                                         mws_ld_cfg_base_addr_vmem2_v;
    wire                                         mws_ld_base_addr_out_vmem2_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_st_base_vmem1_iter_done;
    wire                                         mws_st_base_vmem1_start; 
    wire                                         mws_st_base_vmem1_stall;
    wire                                         mws_st_base_vmem1_done; 
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_st_base_vmem1_lsb;
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_st_base_vmem1_msb;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_cfg_base_addr_vmem1;
    wire                                         mws_st_cfg_base_addr_vmem1_v;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_base_addr_vmem1;
    wire                                         mws_st_base_addr_out_vmem1_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_st_base_vmem2_iter_done;
    wire                                         mws_st_base_vmem2_start; 
    wire                                         mws_st_base_vmem2_stall;
    wire                                         mws_st_base_vmem2_done; 
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_st_base_vmem2_lsb;
    reg   [ BASE_ADDR_SEGMENT_W  -1: 0 ]         _mws_st_base_vmem2_msb;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_cfg_base_addr_vmem2;
    wire                                         mws_st_cfg_base_addr_vmem2_v;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_base_addr_vmem2;
    wire                                         mws_st_base_addr_out_vmem2_v;



    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_ld_tile_vmem1_iter_done;
    wire                                         mws_ld_tile_vmem1_start; 
    wire                                         mws_ld_tile_vmem1_stall;
    wire                                         mws_ld_tile_vmem1_done; 
    reg   [ ADDR_WIDTH           -1: 0 ]         _mws_ld_tile_base_addr_vmem1;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_tile_base_addr_vmem1;
    wire                                         mws_ld_tile_base_addr_vmem1_v;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_tile_addr_vmem1;
    wire                                         mws_ld_tile_addr_out_vmem1_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_ld_tile_vmem2_iter_done;
    wire                                         mws_ld_tile_vmem2_start; 
    wire                                         mws_ld_tile_vmem2_stall;
    wire                                         mws_ld_tile_vmem2_done; 
    reg   [ ADDR_WIDTH           -1: 0 ]         _mws_ld_tile_base_addr_vmem2;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_tile_base_addr_vmem2;
    wire                                         mws_ld_tile_base_addr_vmem2_v;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_ld_tile_addr_vmem2;
    wire                                         mws_ld_tile_addr_out_vmem2_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_st_tile_vmem1_iter_done;
    wire                                         mws_st_tile_vmem1_start; 
    wire                                         mws_st_tile_vmem1_stall;
    wire                                         mws_st_tile_vmem1_done; 
    reg   [ ADDR_WIDTH           -1: 0 ]         _mws_st_tile_base_addr_vmem1;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_tile_base_addr_vmem1;
    wire                                         mws_st_tile_base_addr_vmem1_v;

    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_tile_addr_vmem1;
    wire                                         mws_st_tile_addr_out_vmem1_v;
    
    //MWS
    wire  [NUM_MAX_LOOPS : 0]                    mws_st_tile_vmem2_iter_done;
    wire                                         mws_st_tile_vmem2_start; 
    wire                                         mws_st_tile_vmem2_stall;
    wire                                         mws_st_tile_vmem2_done; 
    reg   [ ADDR_WIDTH           -1: 0 ]         _mws_st_tile_base_addr_vmem2;
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_tile_base_addr_vmem2;
    wire                                         mws_st_tile_base_addr_vmem2_v;
    
    wire  [ ADDR_WIDTH           -1: 0 ]         mws_st_tile_addr_vmem2;
    wire                                         mws_st_tile_addr_out_vmem2_v;    
 
 // AXI Interface   
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

    wire                                        curr_group_ld_config_done;
    wire                                        curr_group_st_config_done;
    // reg                                         _curr_group_ld_config_done;
    // reg                                         _curr_group_st_config_done;   
//==============================================================================
  
//==============================================================================
// ASSIGNS CONFIG SIGNALS
//==============================================================================

    // genvar k;
    // generate
        // for (k=0; k<MAX_NUM_GROUPS; k=k+1) begin
           // always @(*) begin
               // if (k == cfg_group_id)
                  // _curr_group_ld_config_done =  ld_config_done[k];
           // end 
        // end
    // endgenerate

    // generate
        // for (k=0; k<MAX_NUM_GROUPS; k=k+1) begin
           // always @(*) begin
               // if (k == cfg_group_id)
                  // _curr_group_st_config_done =  st_config_done[k];
           // end 
        // end
    // endgenerate
    assign curr_group_ld_config_done = ld_config_done[cfg_group_id];
    assign curr_group_st_config_done = st_config_done[cfg_group_id];

    assign ld_cfg_base_addr_v = (opcode == LD_CONFIG_BASE_ADDR) && (~curr_group_ld_config_done);
    assign ld_cfg_base_loop_iter_v = (opcode == LD_CONFIG_BASE_LOOP_ITER) && (~curr_group_ld_config_done);
    assign ld_cfg_base_loop_stride_v = (opcode == LD_CONFIG_BASE_LOOP_STRIDE) && (~curr_group_ld_config_done);
      
    assign ld_cfg_tile_loop_iter_v = opcode == LD_CONFIG_TILE_LOOP_ITER && (~curr_group_ld_config_done);
    assign ld_cfg_tile_loop_stride_v = opcode == LD_CONFIG_TILE_LOOP_STRIDE && (~curr_group_ld_config_done);
    
    assign ld_start = opcode == LD_START;
    
  
    
    assign st_cfg_base_addr_v = opcode == ST_CONFIG_BASE_ADDR && (~curr_group_st_config_done);
    assign st_cfg_base_loop_iter_v = opcode == ST_CONFIG_BASE_LOOP_ITER && (~curr_group_st_config_done);
    assign st_cfg_base_loop_stride_v = opcode == ST_CONFIG_BASE_LOOP_STRIDE && (~curr_group_st_config_done);
    
    assign st_cfg_tile_loop_iter_v = opcode == ST_CONFIG_TILE_LOOP_ITER && (~curr_group_st_config_done);
    assign st_cfg_tile_loop_stride_v = opcode == ST_CONFIG_TILE_LOOP_STRIDE && (~curr_group_st_config_done);
    
    assign st_start = opcode == ST_START;    



    assign cfg_group_id = ld_st_group_id;
   
    assign cfg_base_addr_segment = dest_ns_id[2];
    assign cfg_ns_id = dest_ns_id[1:0];
    assign cfg_loop_id = dest_ns_index_id;
    assign cfg_stride_segment = dest_ns_id[2];
    assign cfg_base_addr = {src1_ns_id, src1_ns_index_id, src2_ns_id, src2_ns_index_id};
    assign cfg_loop_stride = {src1_ns_id, src1_ns_index_id, src2_ns_id, src2_ns_index_id};
    assign cfg_loop_iter = {src1_ns_id, src1_ns_index_id, src2_ns_id, src2_ns_index_id};
    assign cfg_mem_req_size = {src1_ns_id, src1_ns_index_id, src2_ns_id, src2_ns_index_id};
    assign cfg_ld_st_data_width = {1'b0, dest_ns_id} + 1;
//==============================================================================
    assign simd_tiles_done = mws_ld_base_vmem1_done || mws_ld_base_vmem2_done;
    assign ld_mem_simd_done = mws_ld_tile_vmem1_done || mws_ld_tile_vmem2_done;
    assign st_mem_simd_done = mws_st_tile_vmem1_done || mws_st_tile_vmem2_done;

//==============================================================================    
// mem_walker_stride and controller_fsm for BASE_ADDR/LD/VMEM1    
//==============================================================================
    always @(posedge clk) begin
        if (reset) begin
            _mws_ld_base_vmem1_lsb <= 0;
            _mws_ld_base_vmem1_msb <= 0;
        end
        else if (ld_cfg_base_addr_v && cfg_base_addr_segment == 0 && cfg_ns_id == NS_VMEM_1)
            _mws_ld_base_vmem1_lsb <= cfg_base_addr;
        else if (ld_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_1)
            _mws_ld_base_vmem1_msb <= cfg_base_addr;
    end
    assign mws_ld_cfg_base_addr_vmem1[BASE_ADDR_SEGMENT_W-1:0] = _mws_ld_base_vmem1_lsb;
    assign mws_ld_cfg_base_addr_vmem1[2*BASE_ADDR_SEGMENT_W-1:BASE_ADDR_SEGMENT_W] = _mws_ld_base_vmem1_msb;
    
    register_sync #(1) mws_ld_cfg_base_vmem1_delay (clk, reset, (ld_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_1), mws_ld_cfg_base_addr_vmem1_v);
 
    
    wire                              ld_cfg_stride_base_addr_vmem1_v;
    wire                              ld_cfg_base_loop_iter_vmem1_v;

 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_ld_cfg_stride_base_addr_vmem1;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_vmem1_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_vmem1_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_ld_cfg_stride_vmem1_lsb <= 0;
           _mws_ld_cfg_stride_vmem1_msb <= 0;
        end
        else if (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 0 )
            _mws_ld_cfg_stride_vmem1_lsb <= cfg_loop_stride;
        else if (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 )
            _mws_ld_cfg_stride_vmem1_msb <= cfg_loop_stride;
    end    
    assign mws_ld_cfg_stride_base_addr_vmem1[ADDR_STRIDE_W-1:0] = _mws_ld_cfg_stride_vmem1_lsb;
    assign mws_ld_cfg_stride_base_addr_vmem1[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_ld_cfg_stride_vmem1_msb;
    
    register_sync #(1) mws_ld_cfg_stride_base_vmem1_delay (clk, reset, (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 ), ld_cfg_stride_base_addr_vmem1_v);
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_base_ld_vmem1 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_ld_cfg_base_addr_vmem1 ),
      .iter_done                    ( mws_ld_base_vmem1_iter_done ),
      .start                        ( mws_ld_base_vmem1_start ),
      .stall                        ( mws_ld_base_vmem1_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_ld_cfg_base_addr_vmem1_v ),
      
      .cfg_loop_id                  ( cfg_loop_id ),
      .cfg_addr_stride_v            ( ld_cfg_stride_base_addr_vmem1_v ),
      .cfg_addr_stride              ( mws_ld_cfg_stride_base_addr_vmem1 ),
      
      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_ld_base_addr_vmem1 ),
      .addr_out_valid               ( mws_ld_base_addr_out_vmem1_v )
    );

    assign ld_cfg_base_loop_iter_vmem1_v = ld_cfg_base_loop_iter_v && cfg_ns_id == NS_VMEM_1;

    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_base_ld_vmem1  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_ld_base_vmem1_start ),
      .block_done                   ( block_done              ),
      .done                         ( mws_ld_base_vmem1_done ),
      .stall                        ( mws_ld_base_vmem1_stall ),
      
      .cfg_loop_iter_v              ( ld_cfg_base_loop_iter_vmem1_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_ld_base_vmem1_iter_done ),
      .current_iters                (                            )
    );



//==============================================================================    
// mem_walker_stride and controller_fsm for BASE_ADDR/LD/VMEM2 
//==============================================================================
    always @(posedge clk) begin
        if (reset) begin
            _mws_ld_base_vmem2_lsb <= 0;
            _mws_ld_base_vmem2_msb <= 0;
        end
        else if (ld_cfg_base_addr_v && cfg_base_addr_segment == 0 && cfg_ns_id == NS_VMEM_2)
            _mws_ld_base_vmem2_lsb <= cfg_base_addr;
        else if (ld_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_2)
            _mws_ld_base_vmem2_msb <= cfg_base_addr;
    end
    
    assign mws_ld_cfg_base_addr_vmem2[BASE_ADDR_SEGMENT_W-1:0] = _mws_ld_base_vmem2_lsb;
    assign mws_ld_cfg_base_addr_vmem2[2*BASE_ADDR_SEGMENT_W-1:BASE_ADDR_SEGMENT_W] = _mws_ld_base_vmem2_msb;

    register_sync #(1) mws_ld_cfg_base_vmem2_delay (clk, reset, (ld_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_2), mws_ld_cfg_base_addr_vmem2_v);
    
    wire                              ld_cfg_stride_base_addr_vmem2_v;
    wire                              ld_cfg_base_loop_iter_vmem2_v;

 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_ld_cfg_stride_base_addr_vmem2;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_vmem2_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_vmem2_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_ld_cfg_stride_vmem2_lsb <= 0;
           _mws_ld_cfg_stride_vmem2_msb <= 0;
        end
        else if (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 0 )
            _mws_ld_cfg_stride_vmem2_lsb <= cfg_loop_stride;
        else if (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 )
            _mws_ld_cfg_stride_vmem2_msb <= cfg_loop_stride;
    end    
    assign mws_ld_cfg_stride_base_addr_vmem2[ADDR_STRIDE_W-1:0] = _mws_ld_cfg_stride_vmem2_lsb;
    assign mws_ld_cfg_stride_base_addr_vmem2[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_ld_cfg_stride_vmem2_msb;
    
    register_sync #(1) mws_ld_cfg_stride_base_vmem2_delay (clk, reset, (ld_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 ), ld_cfg_stride_base_addr_vmem2_v);
    
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ) ,
      .GROUP_ENABLED                ( GROUP_ENABLED  ) 
    ) mws_base_ld_vmem2 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_ld_cfg_base_addr_vmem2 ),
      .iter_done                    ( mws_ld_base_vmem2_iter_done ),
      .start                        ( mws_ld_base_vmem2_start ),
      .stall                        ( mws_ld_base_vmem2_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_ld_cfg_base_addr_vmem2_v ),
      
      .cfg_loop_id                  ( cfg_loop_id ),
      .cfg_addr_stride_v            ( ld_cfg_stride_base_addr_vmem2_v ),
      .cfg_addr_stride              ( mws_ld_cfg_stride_base_addr_vmem2 ),
      
      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_ld_base_addr_vmem2 ),
      .addr_out_valid               ( mws_ld_base_addr_out_vmem2_v )
    );
    assign ld_cfg_base_loop_iter_vmem2_v = ld_cfg_base_loop_iter_v && cfg_ns_id == NS_VMEM_2;

    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_base_ld_vmem2  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_ld_base_vmem2_start ),
      .block_done                   ( block_done              ),
      .done                         ( mws_ld_base_vmem2_done ),
      .stall                        ( mws_ld_base_vmem2_stall ),
      
      .cfg_loop_iter_v              ( ld_cfg_base_loop_iter_vmem2_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_ld_base_vmem2_iter_done ),
      .current_iters                (                            )
    );

//==============================================================================    
// mem_walker_stride and controller_fsm for TILE/LD/VMEM1    
//==============================================================================
    always @(posedge clk) begin
       if (reset)
           _mws_ld_tile_base_addr_vmem1 <= 0;
       else if (mws_ld_base_addr_out_vmem1_v)
           _mws_ld_tile_base_addr_vmem1 <= mws_ld_base_addr_vmem1;
    end
    
    assign mws_ld_tile_base_addr_vmem1 = _mws_ld_tile_base_addr_vmem1;

    register_sync #(1) mws_ld_cfg_tile_vmem1_delay (clk, reset, mws_ld_base_addr_out_vmem1_v, mws_ld_tile_base_addr_vmem1_v);
 
 
 
 
    wire                              ld_cfg_stride_tile_addr_vmem1_v;


 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_ld_cfg_stride_tile_addr_vmem1;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_tile_vmem1_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_tile_vmem1_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_ld_cfg_stride_tile_vmem1_lsb <= 0;
           _mws_ld_cfg_stride_tile_vmem1_msb <= 0;
        end
        else if (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 0 )
            _mws_ld_cfg_stride_tile_vmem1_lsb <= cfg_loop_stride;
        else if (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 )
            _mws_ld_cfg_stride_tile_vmem1_msb <= cfg_loop_stride;
    end    
    assign mws_ld_cfg_stride_tile_addr_vmem1[ADDR_STRIDE_W-1:0] = _mws_ld_cfg_stride_tile_vmem1_lsb;
    assign mws_ld_cfg_stride_tile_addr_vmem1[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_ld_cfg_stride_tile_vmem1_msb;
    
    register_sync #(1) mws_ld_cfg_stride_tile_vmem1_delay (clk, reset, (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 ), ld_cfg_stride_tile_addr_vmem1_v);
       
    
    wire                              ld_cfg_tile_loop_iter_vmem1_v;
    assign ld_cfg_tile_loop_iter_vmem1_v = ld_cfg_tile_loop_iter_v && cfg_ns_id == NS_VMEM_1;
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_tile_ld_vmem1 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_ld_tile_base_addr_vmem1 ),
      .iter_done                    ( mws_ld_tile_vmem1_iter_done ),
      .start                        ( mws_ld_tile_vmem1_start ),
      .stall                        ( mws_ld_tile_vmem1_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_ld_tile_base_addr_vmem1_v ),
      
      .cfg_loop_id                  ( cfg_loop_id                   ),
      .cfg_addr_stride_v            ( ld_cfg_stride_tile_addr_vmem1_v ),
      .cfg_addr_stride              ( mws_ld_cfg_stride_tile_addr_vmem1 ),

      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_ld_tile_addr_vmem1 ),
      .addr_out_valid               ( mws_ld_tile_addr_out_vmem1_v )
    );



    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_tile_ld_vmem1  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_ld_tile_vmem1_start ),
      .done                         ( mws_ld_tile_vmem1_done ),
      .stall                        ( mws_ld_tile_vmem1_stall ),
      .block_done                   ( block_done              ),
      
      .cfg_loop_iter_v              ( ld_cfg_tile_loop_iter_vmem1_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_ld_tile_vmem1_iter_done ),
      .current_iters                (                           )
    );


//==============================================================================    
// mem_walker_stride and controller_fsm for TILE/LD/VMEM2    
//==============================================================================
    always @(posedge clk) begin
       if (reset)
           _mws_ld_tile_base_addr_vmem2 <= 0;
       else if (mws_ld_base_addr_out_vmem2_v)
           _mws_ld_tile_base_addr_vmem2 <= mws_ld_base_addr_vmem2;
    end
    
    assign mws_ld_tile_base_addr_vmem2 = _mws_ld_tile_base_addr_vmem2;

    register_sync #(1) mws_ld_cfg_tile_vmem2_delay (clk, reset, mws_ld_base_addr_out_vmem2_v, mws_ld_tile_base_addr_vmem2_v);
 
 
 
 
    wire                              ld_cfg_stride_tile_addr_vmem2_v;


 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_ld_cfg_stride_tile_addr_vmem2;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_tile_vmem2_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_ld_cfg_stride_tile_vmem2_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_ld_cfg_stride_tile_vmem2_lsb <= 0;
           _mws_ld_cfg_stride_tile_vmem2_msb <= 0;
        end
        else if (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 0 )
            _mws_ld_cfg_stride_tile_vmem2_lsb <= cfg_loop_stride;
        else if (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 )
            _mws_ld_cfg_stride_tile_vmem2_msb <= cfg_loop_stride;
    end    
    assign mws_ld_cfg_stride_tile_addr_vmem2[ADDR_STRIDE_W-1:0] = _mws_ld_cfg_stride_tile_vmem2_lsb;
    assign mws_ld_cfg_stride_tile_addr_vmem2[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_ld_cfg_stride_tile_vmem2_msb;
    
    register_sync #(1) mws_ld_cfg_stride_tile_vmem2_delay (clk, reset, (ld_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 ), ld_cfg_stride_tile_addr_vmem2_v);
       
    
    wire                              ld_cfg_tile_loop_iter_vmem2_v;
    assign ld_cfg_tile_loop_iter_vmem2_v = ld_cfg_tile_loop_iter_v && cfg_ns_id == NS_VMEM_2;
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_tile_ld_vmem2 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_ld_tile_base_addr_vmem2 ),
      .iter_done                    ( mws_ld_tile_vmem2_iter_done ),
      .start                        ( mws_ld_tile_vmem2_start ),
      .stall                        ( mws_ld_tile_vmem2_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_ld_tile_base_addr_vmem2_v ),
      
      .cfg_loop_id                  ( cfg_loop_id                   ),
      .cfg_addr_stride_v            ( ld_cfg_stride_tile_addr_vmem2_v ),
      .cfg_addr_stride              ( mws_ld_cfg_stride_tile_addr_vmem2 ),

      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_ld_tile_addr_vmem2 ),
      .addr_out_valid               ( mws_ld_tile_addr_out_vmem2_v )
    );



    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_tile_ld_vmem2  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_ld_tile_vmem2_start ),
      .done                         ( mws_ld_tile_vmem2_done ),
      .stall                        ( mws_ld_tile_vmem2_stall ),
      .block_done                   ( block_done              ),
      
      .cfg_loop_iter_v              ( ld_cfg_tile_loop_iter_vmem2_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_ld_tile_vmem2_iter_done ),
      .current_iters                (                           )
    );


//==============================================================================    
// mem_walker_stride and controller_fsm for BASE_ADDR/ST/VMEM1    
//==============================================================================
   always @(posedge clk) begin
        if (reset) begin
            _mws_st_base_vmem1_lsb <= 0;
            _mws_st_base_vmem1_msb <= 0;
        end
        else if (st_cfg_base_addr_v && cfg_base_addr_segment == 0 && cfg_ns_id == NS_VMEM_1)
            _mws_st_base_vmem1_lsb <= cfg_base_addr;
        else if (st_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_1)
            _mws_st_base_vmem1_msb <= cfg_base_addr;
    end
    assign mws_st_cfg_base_addr_vmem1[BASE_ADDR_SEGMENT_W-1:0] = _mws_st_base_vmem1_lsb;
    assign mws_st_cfg_base_addr_vmem1[2*BASE_ADDR_SEGMENT_W-1:BASE_ADDR_SEGMENT_W] = _mws_st_base_vmem1_msb;
    
    register_sync #(1) mws_st_cfg_base_vmem1_delay (clk, reset, (st_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_1), mws_st_cfg_base_addr_vmem1_v);
 
    
    wire                              st_cfg_stride_base_addr_vmem1_v;
    wire                              st_cfg_base_loop_iter_vmem1_v;

 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_st_cfg_stride_base_addr_vmem1;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_vmem1_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_vmem1_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_st_cfg_stride_vmem1_lsb <= 0;
           _mws_st_cfg_stride_vmem1_msb <= 0;
        end
        else if (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 0 )
            _mws_st_cfg_stride_vmem1_lsb <= cfg_loop_stride;
        else if (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 )
            _mws_st_cfg_stride_vmem1_msb <= cfg_loop_stride;
    end    
    assign mws_st_cfg_stride_base_addr_vmem1[ADDR_STRIDE_W-1:0] = _mws_st_cfg_stride_vmem1_lsb;
    assign mws_st_cfg_stride_base_addr_vmem1[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_st_cfg_stride_vmem1_msb;
    
    register_sync #(1) mws_st_cfg_stride_base_vmem1_delay (clk, reset, (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 ), st_cfg_stride_base_addr_vmem1_v);
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_base_st_vmem1 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_st_cfg_base_addr_vmem1 ),
      .iter_done                    ( mws_st_base_vmem1_iter_done ),
      .start                        ( mws_st_base_vmem1_start ),
      .stall                        ( mws_st_base_vmem1_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_st_cfg_base_addr_vmem1_v ),
      
      .cfg_loop_id                  ( cfg_loop_id ),
      .cfg_addr_stride_v            ( st_cfg_stride_base_addr_vmem1_v ),
      .cfg_addr_stride              ( mws_st_cfg_stride_base_addr_vmem1 ),
      
      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_st_base_addr_vmem1 ),
      .addr_out_valid               ( mws_st_base_addr_out_vmem1_v )
    );

    assign st_cfg_base_loop_iter_vmem1_v = st_cfg_base_loop_iter_v && cfg_ns_id == NS_VMEM_1;

    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_base_st_vmem1  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_st_base_vmem1_start ),
      .block_done                   ( block_done              ),
      .done                         ( mws_st_base_vmem1_done ),
      .stall                        ( mws_st_base_vmem1_stall ),
      
      .cfg_loop_iter_v              ( st_cfg_base_loop_iter_vmem1_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_st_base_vmem1_iter_done ),
      .current_iters                (                            )
    );


//==============================================================================    
// mem_walker_stride and controller_fsm for BASE_ADDR/ST/VMEM2    
//==============================================================================
    always @(posedge clk) begin
        if (reset) begin
            _mws_st_base_vmem2_lsb <= 0;
            _mws_st_base_vmem2_msb <= 0;
        end
        else if (st_cfg_base_addr_v && cfg_base_addr_segment == 0 && cfg_ns_id == NS_VMEM_2)
            _mws_st_base_vmem2_lsb <= cfg_base_addr;
        else if (st_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_2)
            _mws_st_base_vmem2_msb <= cfg_base_addr;
    end
    assign mws_st_cfg_base_addr_vmem2[BASE_ADDR_SEGMENT_W-1:0] = _mws_st_base_vmem2_lsb;
    assign mws_st_cfg_base_addr_vmem2[2*BASE_ADDR_SEGMENT_W-1:BASE_ADDR_SEGMENT_W] = _mws_st_base_vmem2_msb;
    
    register_sync #(1) mws_st_cfg_base_vmem2_delay (clk, reset, (st_cfg_base_addr_v && cfg_base_addr_segment == 1 && cfg_ns_id == NS_VMEM_2), mws_st_cfg_base_addr_vmem2_v);
 
    
    wire                              st_cfg_stride_base_addr_vmem2_v;
    wire                              st_cfg_base_loop_iter_vmem2_v;

 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_st_cfg_stride_base_addr_vmem2;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_vmem2_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_vmem2_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_st_cfg_stride_vmem2_lsb <= 0;
           _mws_st_cfg_stride_vmem2_msb <= 0;
        end
        else if (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 0 )
            _mws_st_cfg_stride_vmem2_lsb <= cfg_loop_stride;
        else if (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 )
            _mws_st_cfg_stride_vmem2_msb <= cfg_loop_stride;
    end    
    assign mws_st_cfg_stride_base_addr_vmem2[ADDR_STRIDE_W-1:0] = _mws_st_cfg_stride_vmem2_lsb;
    assign mws_st_cfg_stride_base_addr_vmem2[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_st_cfg_stride_vmem2_msb;
    
    register_sync #(1) mws_st_cfg_stride_base_vmem2_delay (clk, reset, (st_cfg_base_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 ), st_cfg_stride_base_addr_vmem2_v);
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_base_st_vmem2 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_st_cfg_base_addr_vmem2 ),
      .iter_done                    ( mws_st_base_vmem2_iter_done ),
      .start                        ( mws_st_base_vmem2_start ),
      .stall                        ( mws_st_base_vmem2_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_st_cfg_base_addr_vmem2_v ),
      
      .cfg_loop_id                  ( cfg_loop_id ),
      .cfg_addr_stride_v            ( st_cfg_stride_base_addr_vmem2_v ),
      .cfg_addr_stride              ( mws_st_cfg_stride_base_addr_vmem2 ),
      
      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_st_base_addr_vmem2 ),
      .addr_out_valid               ( mws_st_base_addr_out_vmem2_v )
    );

    assign st_cfg_base_loop_iter_vmem2_v = st_cfg_base_loop_iter_v && cfg_ns_id == NS_VMEM_2;

    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_base_st_vmem2  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_st_base_vmem2_start ),
      .block_done                   ( block_done              ),
      .done                         ( mws_st_base_vmem2_done ),
      .stall                        ( mws_st_base_vmem2_stall ),
      
      .cfg_loop_iter_v              ( st_cfg_base_loop_iter_vmem2_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_st_base_vmem2_iter_done ),
      .current_iters                (                            )
    );

//==============================================================================    
// mem_walker_stride and controller_fsm for TILE/ST/VMEM1    
//==============================================================================
    always @(posedge clk) begin
       if (reset)
           _mws_st_tile_base_addr_vmem1 <= 0;
       else if (mws_st_base_addr_out_vmem1_v)
           _mws_st_tile_base_addr_vmem1 <= mws_st_base_addr_vmem1;
    end
    
    assign mws_st_tile_base_addr_vmem1 = _mws_st_tile_base_addr_vmem1;

    register_sync #(1) mws_st_cfg_tile_vmem1_delay (clk, reset, mws_st_base_addr_out_vmem1_v, mws_st_tile_base_addr_vmem1_v);
 
 
 
 
    wire                              st_cfg_stride_tile_addr_vmem1_v;


 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_st_cfg_stride_tile_addr_vmem1;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_tile_vmem1_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_tile_vmem1_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_st_cfg_stride_tile_vmem1_lsb <= 0;
           _mws_st_cfg_stride_tile_vmem1_msb <= 0;
        end
        else if (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 0 )
            _mws_st_cfg_stride_tile_vmem1_lsb <= cfg_loop_stride;
        else if (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 )
            _mws_st_cfg_stride_tile_vmem1_msb <= cfg_loop_stride;
    end    
    assign mws_st_cfg_stride_tile_addr_vmem1[ADDR_STRIDE_W-1:0] = _mws_st_cfg_stride_tile_vmem1_lsb;
    assign mws_st_cfg_stride_tile_addr_vmem1[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_st_cfg_stride_tile_vmem1_msb;
    
    register_sync #(1) mws_st_cfg_stride_tile_vmem1_delay (clk, reset, (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_1 && cfg_stride_segment == 1 ), st_cfg_stride_tile_addr_vmem1_v);
       
    
    wire                              st_cfg_tile_loop_iter_vmem1_v;
    assign st_cfg_tile_loop_iter_vmem1_v = st_cfg_tile_loop_iter_v && cfg_ns_id == NS_VMEM_1;
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_tile_st_vmem1 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_st_tile_base_addr_vmem1 ),
      .iter_done                    ( mws_st_tile_vmem1_iter_done ),
      .start                        ( mws_st_tile_vmem1_start ),
      .stall                        ( mws_st_tile_vmem1_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_st_tile_base_addr_vmem1_v ),
      
      .cfg_loop_id                  ( cfg_loop_id                   ),
      .cfg_addr_stride_v            ( st_cfg_stride_tile_addr_vmem1_v ),
      .cfg_addr_stride              ( mws_st_cfg_stride_tile_addr_vmem1 ),

      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_st_tile_addr_vmem1 ),
      .addr_out_valid               ( mws_st_tile_addr_out_vmem1_v )
    );



    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_tile_st_vmem1  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_st_tile_vmem1_start ),
      .done                         ( mws_st_tile_vmem1_done ),
      .stall                        ( mws_st_tile_vmem1_stall ),
      .block_done                   ( block_done              ),
      
      .cfg_loop_iter_v              ( st_cfg_tile_loop_iter_vmem1_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_st_tile_vmem1_iter_done ),
      .current_iters                (                           )
    );


//==============================================================================    
// mem_walker_stride and controller_fsm for TILE/ST/VMEM2    
//==============================================================================
      always @(posedge clk) begin
       if (reset)
           _mws_st_tile_base_addr_vmem2 <= 0;
       else if (mws_st_base_addr_out_vmem2_v)
           _mws_st_tile_base_addr_vmem2 <= mws_st_base_addr_vmem2;
    end
    
    assign mws_st_tile_base_addr_vmem2 = _mws_st_tile_base_addr_vmem2;

    register_sync #(1) mws_st_cfg_tile_vmem2_delay (clk, reset, mws_st_base_addr_out_vmem2_v, mws_st_tile_base_addr_vmem2_v);
 
 
 
 
    wire                              st_cfg_stride_tile_addr_vmem2_v;


 
    wire [ 2*ADDR_STRIDE_W          -1: 0 ]              mws_st_cfg_stride_tile_addr_vmem2;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_tile_vmem2_lsb;
    reg  [ ADDR_STRIDE_W            -1: 0 ]              _mws_st_cfg_stride_tile_vmem2_msb;
    
    always @(posedge clk) begin
        if (reset) begin
           _mws_st_cfg_stride_tile_vmem2_lsb <= 0;
           _mws_st_cfg_stride_tile_vmem2_msb <= 0;
        end
        else if (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 0 )
            _mws_st_cfg_stride_tile_vmem2_lsb <= cfg_loop_stride;
        else if (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 )
            _mws_st_cfg_stride_tile_vmem2_msb <= cfg_loop_stride;
    end    
    assign mws_st_cfg_stride_tile_addr_vmem2[ADDR_STRIDE_W-1:0] = _mws_st_cfg_stride_tile_vmem2_lsb;
    assign mws_st_cfg_stride_tile_addr_vmem2[2*ADDR_STRIDE_W-1:ADDR_STRIDE_W] = _mws_st_cfg_stride_tile_vmem2_msb;
    
    register_sync #(1) mws_st_cfg_stride_tile_vmem2_delay (clk, reset, (st_cfg_tile_loop_stride_v && cfg_ns_id == NS_VMEM_2 && cfg_stride_segment == 1 ), st_cfg_stride_tile_addr_vmem2_v);
       
    
    wire                              st_cfg_tile_loop_iter_vmem2_v;
    assign st_cfg_tile_loop_iter_vmem2_v = st_cfg_tile_loop_iter_v && cfg_ns_id == NS_VMEM_2;
    
    mem_walker_stride_group #(
      .ADDR_WIDTH                   ( ADDR_WIDTH ),
      .ADDR_STRIDE_W                ( 2*ADDR_STRIDE_W ),
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )  
    ) mws_tile_st_vmem2 (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .base_addr                    ( mws_st_tile_base_addr_vmem2 ),
      .iter_done                    ( mws_st_tile_vmem2_iter_done ),
      .start                        ( mws_st_tile_vmem2_start ),
      .stall                        ( mws_st_tile_vmem2_stall ),
      .block_done                   ( block_done              ),
      .base_addr_v                  ( mws_st_tile_base_addr_vmem2_v ),
      
      .cfg_loop_id                  ( cfg_loop_id                   ),
      .cfg_addr_stride_v            ( st_cfg_stride_tile_addr_vmem2_v ),
      .cfg_addr_stride              ( mws_st_cfg_stride_tile_addr_vmem2 ),

      .cfg_loop_group_id            ( cfg_group_id ),
      .loop_group_id                ( ld_st_group_id ),
      
      .addr_out                     ( mws_st_tile_addr_vmem2 ),
      .addr_out_valid               ( mws_st_tile_addr_out_vmem2_v )
    );



    controller_fsm_group #(
      .LOOP_ID_W                    ( LOOP_ID_W ),
      .GROUP_ID_W                   ( GROUP_ID_W ),
      .LOOP_ITER_W                  ( LOOP_ITER_W ),
      .GROUP_ENABLED                ( GROUP_ENABLED  )
    ) controller_fsm_tile_st_vmem2  (
      .clk                          ( clk ),
      .reset                        ( reset ),
      
      .start                        ( mws_st_tile_vmem2_start ),
      .done                         ( mws_st_tile_vmem2_done ),
      .stall                        ( mws_st_tile_vmem2_stall ),
      .block_done                   ( block_done              ),
      
      .cfg_loop_iter_v              ( st_cfg_tile_loop_iter_vmem2_v ),
      .cfg_loop_iter                ( cfg_loop_iter ),
      .cfg_loop_iter_loop_id        ( cfg_loop_id ),   
      .cfg_loop_group_id            ( cfg_group_id ),
      
      .loop_group_id                ( ld_st_group_id ),
      .iter_done                    (mws_st_tile_vmem2_iter_done ),
      .current_iters                (                           )
    );

//==============================================================================
//==============================================================================

//==============================================================================
// Memory Request Generation
//==============================================================================

// LD
  wire   [ ADDR_WIDTH           -1:0 ] ld_addr;
  reg    [ ADDR_WIDTH           -1:0 ] _ld_addr;
  wire                                 ld_addr_v;
  
  assign ld_addr_v = mws_ld_tile_addr_out_vmem1_v || mws_ld_tile_addr_out_vmem2_v;
  
  always @(*) begin
     if (mws_ld_tile_addr_out_vmem1_v) 
         _ld_addr = mws_ld_tile_addr_vmem1;
     else if (mws_ld_tile_addr_out_vmem1_v)
         _ld_addr = mws_ld_tile_addr_vmem2;
  end
  assign ld_addr = _ld_addr;

  always @(posedge clk)
  begin
    if (reset) begin
      ld_req_size <= 0;
      ld_data_width <= 0;
    end
    else if (ld_start) begin
      ld_req_size <= cfg_mem_req_size;
      ld_data_width <= cfg_ld_st_data_width;
    end
  end

    assign ld_req_valid_d = ld_addr_v;

  always @(posedge clk)
  begin
    if (reset) begin
      ld_req_valid_q <= 1'b0;
      ld_req_addr <= 0;
    end
    else begin
      ld_req_valid_q <= ld_req_valid_d;
      ld_req_addr <= ld_addr;
    end
  end

// ST
  wire   [ ADDR_WIDTH           -1:0 ] st_addr;
  reg    [ ADDR_WIDTH           -1:0 ] _st_addr;
  wire                                 st_addr_v;
  
  assign st_addr_v = mws_st_tile_addr_out_vmem1_v || mws_st_tile_addr_out_vmem2_v;
  
  always @(*) begin
     if (mws_st_tile_addr_out_vmem1_v) 
         _st_addr = mws_st_tile_addr_vmem1;
     else if (mws_st_tile_addr_out_vmem1_v)
         _st_addr = mws_st_tile_addr_vmem2;
  end
  assign st_addr = _st_addr;

  always @(posedge clk)
  begin
    if (reset) begin
      st_req_size <= 0;
      st_data_width <= 0;
    end
    else if (st_start) begin
      st_req_size <= cfg_mem_req_size;
      st_data_width <= cfg_ld_st_data_width;
    end
  end

    assign st_req_valid_d = st_addr_v;

  always @(posedge clk)
  begin
    if (reset) begin
      st_req_valid_q <= 1'b0;
      st_req_addr <= 0;
    end
    else begin
      st_req_valid_q <= st_req_valid_d;
      st_req_addr <= st_addr;
    end
  end
//==============================================================================


//==============================================================================
// FSM
//==============================================================================
  reg   [ 2                 -1: 0]          _ldmem_ns_id;
  reg   [ 2                 -1: 0]          _stmem_ns_id;
  
  always @(posedge clk) begin
     if (reset) begin
        _ldmem_ns_id <= 0;
        _stmem_ns_id <= 0;
     end
     else if (ld_start)
        _ldmem_ns_id <= cfg_ns_id;
     else if (st_start)
        _stmem_ns_id <= cfg_ns_id;
  end

// LD FSM
  assign mws_ld_base_vmem1_start = (ldmem_state_q == LDMEM_IDLE) && ld_start && (cfg_ns_id == NS_VMEM_1);
  assign mws_ld_base_vmem2_start = (ldmem_state_q == LDMEM_IDLE) && ld_start && (cfg_ns_id == NS_VMEM_2);
  
  assign mws_ld_tile_vmem1_start = (ldmem_state_q == LDMEM_TILE_BUSY_NS_VMEM1);
  assign mws_ld_tile_vmem2_start = (ldmem_state_q == LDMEM_TILE_BUSY_NS_VMEM2);
  
  
  assign mws_ld_base_vmem1_stall = ldmem_state_q != LDMEM_GEN_BASE_ADDR || (ldmem_state_q == LDMEM_GEN_BASE_ADDR) && _ldmem_ns_id != NS_VMEM_1;
  assign mws_ld_base_vmem2_stall = ldmem_state_q != LDMEM_GEN_BASE_ADDR || (ldmem_state_q == LDMEM_GEN_BASE_ADDR) && _ldmem_ns_id != NS_VMEM_2;
  
  assign mws_ld_tile_vmem1_stall = ~axi_rd_ready;
  assign mws_ld_tile_vmem2_stall = ~axi_rd_ready;
  
    
  always @(*) begin
    ldmem_state_d = ldmem_state_q;
    case(ldmem_state_q)
        LDMEM_IDLE: begin
           if (ld_start) 
              ldmem_state_d = LDMEM_GEN_BASE_ADDR; 
        end
        LDMEM_GEN_BASE_ADDR: begin
           if (mws_ld_base_addr_out_vmem1_v)
              ldmem_state_d = LDMEM_TILE_BUSY_NS_VMEM1;
           else if (mws_ld_base_addr_out_vmem2_v)
              ldmem_state_d = LDMEM_TILE_BUSY_NS_VMEM2; 
        end     
        LDMEM_TILE_BUSY_NS_VMEM1: begin
            if (mws_ld_tile_vmem1_done)
                ldmem_state_d = LDMEM_WAIT_0;
        end
        LDMEM_TILE_BUSY_NS_VMEM2: begin
            if (mws_ld_tile_vmem2_done)
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
           if (axi_rd_done)
               ldmem_state_d = LDMEM_DONE; 
        end
        LDMEM_DONE: begin
           ldmem_state_d = LDMEM_IDLE;
        end
    endcase   
  end  
  
  always @(posedge clk) begin
     if (reset) 
        ldmem_state_q <= LDMEM_IDLE; 
     else
        ldmem_state_q <= ldmem_state_d;
  end
      

// ST FSM
  assign mws_st_base_vmem1_start = (stmem_state_q == STMEM_IDLE) && st_start && (cfg_ns_id == NS_VMEM_1);
  assign mws_st_base_vmem2_start = (stmem_state_q == STMEM_IDLE) && st_start && (cfg_ns_id == NS_VMEM_2);
  
  assign mws_st_tile_vmem1_start = (stmem_state_q == STMEM_TILE_BUSY_NS_VMEM1);
  assign mws_st_tile_vmem2_start = (stmem_state_q == STMEM_TILE_BUSY_NS_VMEM2);
  
  
  assign mws_st_base_vmem1_stall = stmem_state_q != STMEM_GEN_BASE_ADDR || (stmem_state_q == STMEM_GEN_BASE_ADDR) && _stmem_ns_id != NS_VMEM_1;
  assign mws_st_base_vmem2_stall = stmem_state_q != STMEM_GEN_BASE_ADDR || (stmem_state_q == STMEM_GEN_BASE_ADDR) && _stmem_ns_id != NS_VMEM_2;
  
  assign mws_st_tile_vmem1_stall = ~axi_wr_ready;
  assign mws_st_tile_vmem2_stall = ~axi_wr_ready;
  
    
  always @(*) begin
    stmem_state_d = stmem_state_q;
    case(stmem_state_q)
        STMEM_IDLE: begin
           if (st_start) 
              stmem_state_d = STMEM_GEN_BASE_ADDR; 
        end
        STMEM_GEN_BASE_ADDR: begin
           if (mws_st_base_addr_out_vmem1_v)
              stmem_state_d = STMEM_TILE_BUSY_NS_VMEM1;
           else if (mws_st_base_addr_out_vmem2_v)
              stmem_state_d = STMEM_TILE_BUSY_NS_VMEM2; 
        end     
        STMEM_TILE_BUSY_NS_VMEM1: begin
            if (mws_st_tile_vmem1_done)
                stmem_state_d = STMEM_WAIT_0;
        end
        STMEM_TILE_BUSY_NS_VMEM2: begin
            if (mws_st_tile_vmem2_done)
                stmem_state_d = STMEM_WAIT_0;           
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
  
  always @(posedge clk) begin
     if (reset) 
        stmem_state_q <= STMEM_IDLE; 
     else
        stmem_state_q <= stmem_state_d;
  end
//==============================================================================    


//==============================================================================
// AXI4 Memory Mapped interface
//==============================================================================
    assign axi_rd_req = ld_req_valid_q;
    assign axi_rd_req_size = ld_req_size * ld_data_width * (NUM_SIMD_LANES / AXI_DATA_WIDTH);
    assign axi_rd_addr = ld_req_addr;
    assign axi_rd_req_id = 1'b0;
    
    assign axi_wr_req = st_req_valid_q;
    assign axi_wr_req_id = 1'b0;
    assign axi_wr_req_size = st_req_size * st_data_width * (NUM_SIMD_LANES / AXI_DATA_WIDTH);
    assign axi_wr_addr = st_req_addr;
    
    assign mem_write_ready = 1'b1;
    assign axi_mem_read_ready = 1'b1;

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
// LD/ST VMEM <--> offchip interface
//==============================================================================
  reg  [ GROUP_COUNTER_LD_ST_WIDTH     -1 : 0 ]   simd_ld_group_counter;
  reg  [ VMEM_BUF_ADDR_W               -1 : 0 ]   simd_ld_addr_counter;
  
  reg  [ GROUP_COUNTER_LD_ST_WIDTH     -1 : 0 ]   simd_st_group_counter;
  reg  [ VMEM_BUF_ADDR_W               -1 : 0 ]   simd_st_addr_counter;  
  
  wire [ VMEM_BUF_ADDR_W               -1 : 0 ]   _vmem_write_addr;
  wire [ VMEM_BUF_ADDR_W               -1 : 0 ]   _vmem_read_addr;   
  
  wire [ VMEM_TAG_BUF_ADDR_W           -1 : 0 ]   tag_vmem_write_addr;
  wire [ VMEM_TAG_BUF_ADDR_W           -1 : 0 ]   tag_vmem_read_addr; 
   
  
  wire [ SIMD_LD_ST_HIGH_BW_GROUP_SIZE                    -1: 0]  group_vmem_high_bw_write_req;
  wire [ SIMD_LD_ST_HIGH_BW_GROUP_SIZE                    -1: 0]  group_vmem_high_bw_read_req;
  wire [ SIMD_LD_ST_LOW_BW_GROUP_SIZE                    -1: 0]   group_vmem_low_bw_write_req;
  wire [ SIMD_LD_ST_LOW_BW_GROUP_SIZE                    -1: 0]   group_vmem_low_bw_read_req;
  
  wire                                            vmem_read_req;
  wire                                            vmem_write_req; 
  
  wire                                            simd_ld_high_bw_v;
  wire                                            simd_st_high_bw_v; 
  
  wire [ AXI_DATA_WIDTH                 -1 : 0 ]                  simd_ld_high_bw_data;
  wire [ AXI_DATA_WIDTH                 -1 : 0 ]                  simd_ld_low_bw_data_packed;
  wire [ SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:0]        simd_ld_low_bw_data_unpacked;

  wire [ AXI_DATA_WIDTH                 -1 : 0 ]                  simd_st_high_bw_data;
  reg  [ AXI_DATA_WIDTH                 -1 : 0 ]                  _simd_st_high_bw_data;
  wire [ AXI_DATA_WIDTH                 -1 : 0 ]                  simd_st_low_bw_data_packed;
  reg [ SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:0]        simd_st_low_bw_data_unpacked;
  
  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem_write_data_out;
  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem1_write_data_out;
  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem2_write_data_out;  

  wire  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]   _vmem_write_addr_out;
  reg  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]    _vmem1_write_addr_out;
  reg  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]    _vmem2_write_addr_out; 
  
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_write_req_out_high_bw;
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_write_req_out_low_bw;
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_write_req_out;  
  
  reg  [ NUM_SIMD_LANES                                  -1:0]    _vmem1_write_req_out;
  reg  [ NUM_SIMD_LANES                                  -1:0]    _vmem2_write_req_out;  



  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem_read_data_in;
  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem1_read_data_in;
  reg  [ NUM_SIMD_LANES*SIMD_DATA_WIDTH              -1:0]        _vmem2_read_data_in;  

  wire  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]   _vmem_read_addr_out;
  reg  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]    _vmem1_read_addr_out;
  reg  [ NUM_SIMD_LANES*VMEM_TAG_BUF_ADDR_W              -1:0]    _vmem2_read_addr_out; 
  
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_read_req_out_high_bw;
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_read_req_out_low_bw;
  wire  [ NUM_SIMD_LANES                                  -1:0]    _vmem_read_req_out;  
  
  reg  [ NUM_SIMD_LANES                                  -1:0]    _vmem1_read_req_out;
  reg  [ NUM_SIMD_LANES                                  -1:0]    _vmem2_read_req_out;  
//==============================================================================

  assign simd_ld_high_bw_v = ld_data_width == SIMD_DATA_WIDTH;
  assign simd_st_high_bw_v = st_data_width == SIMD_DATA_WIDTH;

// LD to VMEM
// Counters for write groups, write addresses
  always @(posedge clk)
  begin
      if (reset)
          simd_ld_group_counter <= 0;
      else if (mem_write_req) begin
          if (simd_ld_high_bw_v && simd_ld_group_counter == SIMD_LD_ST_HIGH_BW_NUM_GROUPS)
              simd_ld_group_counter <= 0;
          else if (~simd_ld_high_bw_v && simd_ld_group_counter == SIMD_LD_ST_LOW_BW_NUM_GROUPS)
              simd_ld_group_counter <= 0;
          else
              simd_ld_group_counter <= simd_ld_group_counter + 1'b1;
      end
  end
 //

//
  always @(posedge clk)
  begin
      if (reset)
          simd_ld_addr_counter <= 0;
      else begin 
          if (mem_write_req) begin
             if (simd_ld_high_bw_v && simd_ld_group_counter == SIMD_LD_ST_HIGH_BW_NUM_GROUPS)
                 simd_ld_addr_counter <= simd_ld_addr_counter + 1'b1;
             else if (~simd_ld_high_bw_v && simd_ld_group_counter == SIMD_LD_ST_LOW_BW_NUM_GROUPS)
                 simd_ld_addr_counter <= simd_ld_addr_counter + 1'b1;
          end  
          else if (ldmem_state_q == LDMEM_DONE)
            simd_ld_addr_counter <= 0;
      end
  end
//

// ************* ASSIGNs for LD ************

// Assign Write data out
  assign simd_ld_high_bw_data = mem_write_data;
  assign simd_ld_low_bw_data_packed = mem_write_data;
  
  genvar i;
  generate
      for (i=0; i<SIMD_LD_ST_LOW_BW_GROUP_SIZE; i=i+1) begin
          wire [LD_ST_LOW_DATA_WIDTH   -1:0]   local_data;
          wire [SIMD_DATA_WIDTH        -1:0]   local_sign_ext_data;
          
          assign local_data = simd_ld_low_bw_data_packed[(i+1)*LD_ST_LOW_DATA_WIDTH-1:i*LD_ST_LOW_DATA_WIDTH];
          assign local_sign_ext_data = {{SIMD_DATA_WIDTH-LD_ST_LOW_DATA_WIDTH{local_data[LD_ST_DATA_WIDTH-1]}},local_data}; 
          assign simd_ld_low_bw_data_unpacked[(i+1)*SIMD_DATA_WIDTH-1:i*SIMD_DATA_WIDTH] = local_sign_ext_data;
      end     
  endgenerate
  
  
  wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH     -1:0]  _vmem_write_data_out_low_bw;
  wire  [NUM_SIMD_LANES*SIMD_DATA_WIDTH     -1:0]  _vmem_write_data_out_high_bw;  

  generate
     for (i=0; i<SIMD_LD_ST_LOW_BW_NUM_GROUPS; i=i+1) begin
         assign _vmem_write_data_out_low_bw[(i+1)*SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH] = simd_ld_low_bw_data_unpacked;
     end 
 endgenerate
  
 generate
     for (i=0; i<SIMD_LD_ST_HIGH_BW_NUM_GROUPS; i=i+1) begin
         assign _vmem_write_data_out_high_bw[(i+1)*SIMD_LD_ST_HIGH_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:SIMD_LD_ST_HIGH_BW_GROUP_SIZE*SIMD_DATA_WIDTH] = simd_ld_high_bw_data;
     end 
 endgenerate 
 
 always @(*) begin
    if (simd_ld_high_bw_v) 
        _vmem_write_data_out = _vmem_write_data_out_high_bw;
    else
        _vmem_write_data_out = _vmem_write_data_out_low_bw;
 end 

 
  always @(*) begin
     if (_ldmem_ns_id == NS_VMEM_1) 
        _vmem1_write_data_out = _vmem_write_data_out;
     else if (_ldmem_ns_id == NS_VMEM_2)
        _vmem2_write_data_out = _vmem_write_data_out; 
  end
  
  assign vmem1_write_data = _vmem1_write_data_out;
  assign vmem2_write_data = _vmem2_write_data_out;
//

// Assign Write addr out
  assign _vmem_write_addr = simd_ld_addr_counter;
  // For now, there is no tag
  assign tag_vmem_write_addr = _vmem_write_addr;
  

  genvar j;
  generate
      for (j=0; j<NUM_SIMD_LANES; j=j+1) begin
          assign _vmem_write_addr_out[(j+1)*VMEM_TAG_BUF_ADDR_W-1: j*VMEM_TAG_BUF_ADDR_W] = tag_vmem_write_addr;         
      end
  endgenerate



  always @(*) begin
     if (_ldmem_ns_id == NS_VMEM_1) 
        _vmem1_write_addr_out = _vmem_write_addr_out;
     else if (_ldmem_ns_id == NS_VMEM_2)
        _vmem2_write_addr_out = _vmem_write_addr_out; 
  end

  assign vmem1_write_addr = _vmem1_write_addr_out;
  assign vmem2_write_addr = _vmem2_write_addr_out ;
//

// Assign Write Req 
  assign vmem_write_req = mem_write_req;
 
  genvar l;
  generate
      for (l=0; l<SIMD_LD_ST_LOW_BW_GROUP_SIZE; l=l+1) begin
          assign group_vmem_low_bw_write_req[l] = vmem_write_req;         
      end
  endgenerate

  genvar m;
  generate
      for (m=0; m<SIMD_LD_ST_HIGH_BW_GROUP_SIZE; m=m+1) begin
          assign group_vmem_high_bw_write_req[m] = vmem_write_req;         
      end
  endgenerate
  
  genvar n;
  generate
      for (n=0; n<SIMD_LD_ST_HIGH_BW_NUM_GROUPS; n=n+1) begin
         assign _vmem_write_req_out_high_bw[(n+1)*SIMD_LD_ST_HIGH_BW_GROUP_SIZE-1:n*SIMD_LD_ST_HIGH_BW_GROUP_SIZE] = (simd_ld_group_counter == n) ? group_vmem_high_bw_write_req : 0;
      end      
  endgenerate
  
  generate
      for (n=0; n<SIMD_LD_ST_LOW_BW_NUM_GROUPS; n=n+1) begin
         assign _vmem_write_req_out_low_bw[(n+1)*SIMD_LD_ST_LOW_BW_GROUP_SIZE-1:n*SIMD_LD_ST_LOW_BW_GROUP_SIZE] = (simd_ld_group_counter == n) ? group_vmem_low_bw_write_req : 0;
      end      
  endgenerate

  assign _vmem_write_req_out = simd_ld_high_bw_v ? _vmem_write_req_out_high_bw : _vmem_write_req_out_low_bw;
  
  always @(*) begin
     if (_ldmem_ns_id == NS_VMEM_1) begin
         _vmem1_write_req_out = _vmem_write_req_out;
         _vmem2_write_req_out = 0;
     end
     else if (_ldmem_ns_id == NS_VMEM_2) begin
         _vmem1_write_req_out = 0;
         _vmem2_write_req_out = _vmem_write_req_out;
     end
  end

  assign vmem1_write_req = _vmem1_write_req_out;
  assign vmem2_write_req = _vmem2_write_req_out;
  
// ************************************************************ //
// ************************************************************ //

// ST from VMEM
// Counters for write groups, write addresses
  always @(posedge clk)
  begin
      if (reset)
          simd_st_group_counter <= 0;
      else if (axi_mem_read_req) begin
          if (simd_st_high_bw_v && simd_st_group_counter == SIMD_LD_ST_HIGH_BW_NUM_GROUPS)
              simd_st_group_counter <= 0;
          else if (~simd_st_high_bw_v && simd_st_group_counter == SIMD_LD_ST_LOW_BW_NUM_GROUPS)
              simd_st_group_counter <= 0;
          else
              simd_st_group_counter <= simd_st_group_counter + 1'b1;
      end
  end
  
  wire [ GROUP_COUNTER_LD_ST_WIDTH      -1 : 0 ] simd_st_group_counter_delayed;
  register_sync #(GROUP_COUNTER_LD_ST_WIDTH) simd_st_group_counter_reg (clk, reset, simd_st_group_counter, simd_st_group_counter_delayed);
  
 //

//
  always @(posedge clk)
  begin
      if (reset)
          simd_st_addr_counter <= 0;
      else begin 
          if (axi_mem_read_req) begin
             if (simd_st_high_bw_v && simd_st_group_counter == SIMD_LD_ST_HIGH_BW_NUM_GROUPS)
                 simd_st_addr_counter <= simd_st_addr_counter + 1'b1;
             else if (~simd_st_high_bw_v && simd_st_group_counter == SIMD_LD_ST_LOW_BW_NUM_GROUPS)
                 simd_st_addr_counter <= simd_st_addr_counter + 1'b1;
          end  
          else if (stmem_state_q == STMEM_DONE)
            simd_st_addr_counter <= 0;
      end
  end
//

// ************* ASSIGNs for LD ************

// Assign read data in
  assign mem_read_data = simd_st_high_bw_v ? simd_st_high_bw_data : simd_st_low_bw_data_packed;
  
  generate
      for (i=0; i<SIMD_LD_ST_LOW_BW_GROUP_SIZE; i=i+1) begin
          wire [LD_ST_LOW_DATA_WIDTH   -1:0]   local_data_low_bw;
          wire [SIMD_DATA_WIDTH        -1:0]   local_data_high_bw;

          assign local_data_high_bw = simd_st_low_bw_data_unpacked[(i+1)*SIMD_DATA_WIDTH-1:i*SIMD_DATA_WIDTH];
          assign local_data_low_bw = local_data_high_bw[LD_ST_LOW_DATA_WIDTH-1:0];
          assign simd_st_low_bw_data_packed[(i+1)*LD_ST_LOW_DATA_WIDTH-1:i*LD_ST_LOW_DATA_WIDTH] = local_data_low_bw;
      end     
  endgenerate
 
  
  generate
      for (i=0; i<SIMD_LD_ST_HIGH_BW_NUM_GROUPS; i=i+1) begin
         always @(*) begin
            if (i == simd_st_group_counter_delayed) 
                _simd_st_high_bw_data = _vmem_read_data_in[(i+1)*SIMD_LD_ST_HIGH_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:i*SIMD_LD_ST_HIGH_BW_GROUP_SIZE*SIMD_DATA_WIDTH];
         end 
      end   
  endgenerate
  
  assign simd_st_high_bw_data = _simd_st_high_bw_data;
 
  generate
      for (i=0; i<SIMD_LD_ST_LOW_BW_NUM_GROUPS; i=i+1) begin
         always @(*) begin
            if (i == simd_st_group_counter_delayed) 
                simd_st_low_bw_data_unpacked = _vmem_read_data_in[(i+1)*SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH-1:i*SIMD_LD_ST_LOW_BW_GROUP_SIZE*SIMD_DATA_WIDTH];
         end 
      end   
  endgenerate 
 

  always @(*) begin
     if (_stmem_ns_id == NS_VMEM_1) 
        _vmem_read_data_in = _vmem1_read_data_in;
     else
        _vmem_read_data_in = _vmem2_read_data_in ; 
  end

  
  
  assign _vmem1_read_data_in = vmem1_read_data ;
  assign _vmem2_read_data_in = vmem2_read_data ;
//

// Assign Read addr out
  assign _vmem_read_addr = simd_st_addr_counter;
  // For now, there is no tag
  assign tag_vmem_read_addr = _vmem_read_addr;
  
  generate
      for (i=0; i<NUM_SIMD_LANES; i=i+1) begin
          assign _vmem_read_addr_out[(i+1)*VMEM_TAG_BUF_ADDR_W-1:i*VMEM_TAG_BUF_ADDR_W] = tag_vmem_read_addr;
       end
  endgenerate

  always @(*) begin
     if (_stmem_ns_id == NS_VMEM_1) 
        _vmem1_read_addr_out = _vmem_read_addr_out;
     else if (_stmem_ns_id == NS_VMEM_2)
        _vmem2_read_addr_out = _vmem_read_addr_out; 
  end

  assign vmem1_read_addr = _vmem1_read_addr_out;
  assign vmem2_read_addr = _vmem2_read_addr_out ;
//

// Assign Read Req 
  assign vmem_read_req = axi_mem_read_req;
 
  generate
      for (l=0; l<SIMD_LD_ST_LOW_BW_GROUP_SIZE; l=l+1) begin
          assign group_vmem_low_bw_read_req[l] = vmem_read_req;         
      end
  endgenerate

  generate
      for (m=0; m<SIMD_LD_ST_HIGH_BW_GROUP_SIZE; m=m+1) begin
          assign group_vmem_high_bw_read_req[m] = vmem_read_req;         
      end
  endgenerate

  generate
      for (n=0; n<SIMD_LD_ST_HIGH_BW_NUM_GROUPS; n=n+1) begin
         assign _vmem_read_req_out_high_bw[(n+1)*SIMD_LD_ST_HIGH_BW_GROUP_SIZE-1:n*SIMD_LD_ST_HIGH_BW_GROUP_SIZE] = (simd_st_group_counter == n) ? group_vmem_high_bw_read_req : 0;
      end      
  endgenerate
  
  generate
      for (n=0; n<SIMD_LD_ST_LOW_BW_NUM_GROUPS; n=n+1) begin
         assign _vmem_read_req_out_low_bw[(n+1)*SIMD_LD_ST_LOW_BW_GROUP_SIZE-1:n*SIMD_LD_ST_LOW_BW_GROUP_SIZE] = (simd_st_group_counter == n) ? group_vmem_low_bw_read_req : 0;
      end      
  endgenerate

  assign _vmem_read_req_out = simd_st_high_bw_v ? _vmem_read_req_out_high_bw : _vmem_read_req_out_low_bw;
  
  always @(*) begin
     if (_stmem_ns_id == NS_VMEM_1) begin
         _vmem1_read_req_out = _vmem_read_req_out;
         _vmem2_read_req_out = 0;
     end
     else if (_stmem_ns_id == NS_VMEM_2) begin
         _vmem1_read_req_out = 0;
         _vmem2_read_req_out = _vmem_read_req_out;
     end
  end

  assign vmem1_read_req = _vmem1_read_req_out;
  assign vmem2_read_req = _vmem2_read_req_out;



//==============================================================================

endmodule