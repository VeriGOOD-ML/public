`define NAMESPACE_NULL        	0
`define NAMESPACE_ALU_OUT       1
`define NAMESPACE_WEIGHT      	2 // NAMESPACE_DATA[0]         // NAMESPACE_WEIGHT[0]	
`define NAMESPACE_DATA        	3 // NAMESPACE_DATA[2 * len(w)]
//`define NAMESPACE_GRADIENT    	4 // NAMESPACE_DATA[len(W)]    // NAMESPACE_WEIGHT[len(W)]
`define NAMESPACE_META     	  	4
`define NAMESPACE_INTERIM     	5 
`define NAMESPACE_NEIGHBOR    	6 //NAMESPACE_NEIGHBOR[0] = NAMESPACE_PE_NEIGHBOR, NAMESPACE_NEIGHBOR[1] = NAMESPACE_PU_NEIGHBOR
`define NAMESPACE_BUS      	  	7 //NAMESPACE_BUS[0] = NAMESPACE_PE_BUS, NAMESPACE_BUS[1] = NAMESPACE_GLOBAL_BUS

`define FN_PASS 0
`define FN_ADD 1
`define FN_SUB 2
`define FN_MUL 3
`define FN_COM 4
`define FN_DIV 5
`define FN_SQR 6
`define FN_SIG 7
`define FN_GAU 8


//MEM_INTERFACE_UNIT
`define NAMESPACE_MEM_INST 		0
`define NAMESPACE_MEM_DATA		1
`define NAMESPACE_MEM_WEIGHT	2
`define NAMESPACE_MEM_META		3

//MEM_INTERFACE STATE MACHINE
`define NUM_STATES            8
`define STATE_WIDTH           3
`define STATE_IDLE            0
`define WEIGHT_READ           1
`define WEIGHT_READ_WAIT      2
`define DATA_READ             3
`define DATA_READ_WAIT        4
`define STATE_COMPUTE         5
`define WEIGHT_WRITE          6
`define WEIGHT_WRITE_WAIT     7
