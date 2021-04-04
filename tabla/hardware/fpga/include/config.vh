`timescale 1ns / 100ps

/////////////////////////////////////
// `define FPGA 1
//`define USE_TRI_STATE 0
`define NUM_AXI 4
`define CONFIG_NO 88
/////////////////////////////////////

/////////////////////////////////////
`define NUM_PU `CONFIGURATION_SELECT(`CONFIG_NO,0)
`define NUM_PE `CONFIGURATION_SELECT(`CONFIG_NO,1)
`define BIT_WIDTH `CONFIGURATION_SELECT(`CONFIG_NO,2)
`define INTERNAL_BIT_WIDTH `CONFIGURATION_SELECT(`CONFIG_NO,3)
`define BENCHMARK `CONFIGURATION_SELECT(`CONFIG_NO,4)
/////////////////////////////////////

/////////////////INDEXES FOR EACH NAMESPACE /////
`define DATA_MEM_SIZE 		(`MEMORY_SIZE(`BENCHMARK,0)*64)/(`NUM_PU*`NUM_PE)
`define WEIGHT_MEM_SIZE 	(`MEMORY_SIZE(`BENCHMARK,1)*64)/(`NUM_PU*`NUM_PE)
`define INTERIM_MEM_SIZE   	(`MEMORY_SIZE(`BENCHMARK,2)*64)/(`NUM_PU*`NUM_PE)
`define INDEX_IN_INST		`MEMORY_SIZE(`BENCHMARK,3)+(64/(`NUM_PU*`NUM_PE)-1)
`define INDEX_INST 			`MEMORY_SIZE(`BENCHMARK,4)
/////////////////////////////////////////////////

/////////////////// MEMORY SIZES ///////////////
`define BUS_READ_DEPTH  (`MEMORY_SIZE(`BENCHMARK,5)*64)/(`NUM_PU*`NUM_PE)
`define BUS_FIFO_DEPTH  (`MEMORY_SIZE(`BENCHMARK,6)*64)/(`NUM_PU*`NUM_PE)
`define NEIGH_FIFO_DEPTH  (`MEMORY_SIZE(`BENCHMARK,7)*64)/(`NUM_PU*`NUM_PE)
///////////////////////////////////////////////

///////////// PIPELINES ////////////////////////
`define AXI_PIPELINE_STAGES 0

`define MEM_PIPELINE_STAGES 1
`define MEM_PIPELINE_STAGES_COMMON 1
`define MEM_PIPELINE_STAGES_OUTPUTS 2

`define BUS_PIPELINE_STAGES_PE 0
`define BUS_PIPELINE_STAGES_PU 2
`define NEIGH_PIPELINE_STAGES_PU 0
`define NEIGH_PIPELINE_STAGES_PE 0
//////////////////////////////////////////////


//NUMBER OF VALID PEs
`define LOG_NUM_PU 		3
`define LOG_NUM_PE 		3
`define NUM_PE_VALID 	`NUM_PU*`NUM_PE

`define LOG_MEM_NAME_SPACES 2
//COMPUTE ELEMENTS
//`define GAUSSIAN
//`define DIV
//`define SQRT
`define SIGMOID
`define INDEX_META 		4	

`define BUS_READ_DEPTH_PU  `BUS_READ_DEPTH
`define BUS_READ_DEPTH_HEAD  `BUS_READ_DEPTH
`define BUS_READ_DEPTH_HEAD_PU  `BUS_READ_DEPTH

`define FIFO_DEPTH_MACRO(pe) (\
(pe == 0) ?  `BUS_READ_DEPTH_HEAD: \
`BUS_READ_DEPTH)
`define FIFO_DEPTH_MACRO_PU(pe) (\
(pe == 0) ?  `BUS_READ_DEPTH_HEAD_PU: \
`BUS_READ_DEPTH_PU)


//MEM INSTRUCTION FILE 
`define MEM_INST_INIT  "hw-imp/include/instructions/linear-3/mem-inst/memInst_init.v"
`define MEM_INST_INIT_FPGA  		"hw-imp/include/instructions/linear-3/mem-inst/memInst.txt"

//COMPUTE INSTRUCTION FILE 
`define COMPUTE_INST_INIT	"hw-imp/include/instructions/linear-3/compute-inst/"

//META PARAMETER FILE 
`define META_DATA_FILE		"hw-imp/include/instructions/linear-3/meta.txt"

//PE COUNTER FOR WEIGHTS
//unused in asic flow as of now
`define WEIGHT_CTRL_INIT "hw-imp/include/instructions/linear-3/mem-inst/weightInst.txt" 
