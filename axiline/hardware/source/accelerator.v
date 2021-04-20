//
//      verilog template for pipeline stages in Axiline
//
`include "../include/config.vh"
`include "accelerator_unit.v"

module accelerator#(
    parameter inputBitwidth=`INPUT_BITWIDTH,
	parameter bitwidth=`BITWIDTH,
    parameter selBitwidth=1,
    parameter logNumCycle=`LOG_NUM_CYCLE,
    parameter numCycle=`NUM_CYCLE,
    parameter size= `SIZE,
	parameter numUnit=`NUM_UNIT
)(
    input [inputBitwidth*size*numUnit-1:0]data_in_w,
    input [inputBitwidth*size*numUnit-1:0]data_in_x,
    input [inputBitwidth*numUnit-1:0]bias,
	input [inputBitwidth*numUnit-1:0]rate,
    input clk,
    input start,
    input rst,
	input [inputBitwidth-1:0]mu,
    output [bitwidth*size*numUnit-1:0] data_out_r
);

	`ifdef RECO
		genvar i;
		generate
			for (i=0;i<numUnit;i=i+1)begin : NUM
				accelerator_unit #(
					.inputBitwidth(inputBitwidth),
					.bitwidth(bitwidth),
					.selBitwidth(selBitwidth),
					.logNumCycle(logNumCycle),
					.numCycle(numCycle),
					.size(size)
				)unit(
					.data_in_w(data_in_w[inputBitwidth*size*(i+1)-1:inputBitwidth*size*i]),
					.data_in_x(data_in_x[inputBitwidth*size*(i+1)-1:inputBitwidth*size*i]),
					.bias(bias[inputBitwidth*(i+1)-1:inputBitwidth*i]),
					.rate(rate[inputBitwidth*(i+1)-1:inputBitwidth*i]),
					.clk(clk),
					.start(start),
					.rst(rst),
					.mu(mu),
					.data_out_r(data_out_r[bitwidth*size*(i+1)-1:bitwidth*size*i])		
				);    
			end
    	endgenerate
	`else
		genvar i;
		generate
			for (i=0;i<1;i=i+1)begin : NUM
				accelerator_unit #(
					.inputBitwidth(inputBitwidth),
					.bitwidth(bitwidth),
					.selBitwidth(selBitwidth),
					.logNumCycle(logNumCycle),
					.numCycle(numCycle),
					.size(size)
				)unit(
					.data_in_w(data_in_w[inputBitwidth*size*(i+1)-1:inputBitwidth*size*i]),
					.data_in_x(data_in_x[inputBitwidth*size*(i+1)-1:inputBitwidth*size*i]),
					.bias(bias[inputBitwidth*(i+1)-1:inputBitwidth*i]),
					.rate(),
					.clk(clk),
					.start(start),
					.rst(rst),
					.mu(mu),
					.data_out_r(data_out_r[bitwidth*size*(i+1)-1:bitwidth*size*i])		
				);    
			end
    	endgenerate
	`endif

endmodule



