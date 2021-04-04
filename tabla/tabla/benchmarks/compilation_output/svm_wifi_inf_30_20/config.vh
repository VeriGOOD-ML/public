
`timescale 1ns / 100ps
`define FPGA 1
`define NS_SIZE 8192
//INDEXES FOR EACH NAMESPACE
`define INDEX_INST 			10
`define INDEX_DATA 			`NS_SIZE
`define INDEX_WEIGHT 		`NS_SIZE
`define INDEX_META 			`NS_SIZE
`define INDEX_INTERIM   	`NS_SIZE

//INDEX INSIDE THE INSTRUCTION
`define INDEX_IN_INST		`NS_SIZE

`define BUS_READ_DEPTH  512
`define BUS_FIFO_DEPTH  512
`define NEIGH_FIFO_DEPTH  256

//NUMBER OF VALID PEs
`define LOG_NUM_PU 		3
`define LOG_NUM_PE 		6
`define NUM_PE_VALID 		64

`define LOG_MEM_NAME_SPACES 2
//COMPUTE ELEMENTS
//`define GAUSSIAN
//`define DIV
//`define SQRT
//`define SIGMOID

//MEM INSTRUCTION FILE 
`define MEM_INST_INIT  "svm_wifi_inf_30_20/mem-inst/memInst_init.v"
`define MEM_INST_INIT_FPGA  		"svm_wifi_inf_30_20/mem-inst/memInst.txt"

//COMPUTE INSTRUCTION FILE 
`define COMPUTE_INST_INIT	"svm_wifi_inf_30_20/compute-inst/"

//META PARAMETER FILE 
`define META_DATA_FILE		"svm_wifi_inf_30_20/meta.txt"

//PE COUNTER FOR WEIGHTS
//unused in asic flow as of now
`define WEIGHT_CTRL_INIT "svm_wifi_inf_30_20/mem-inst/weightInst.txt"
