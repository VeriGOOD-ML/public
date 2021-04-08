`include "config.vh"

module $algo$_$asic$ #(
	//
	parameter	WEIGHT_N			= `NUM_WEIGHT,
	parameter	input_bitwidth	    = `INPUT_BITWIDTH,
	parameter	bitwidth		= `INTERNAL_BITWIDTH
	//
)(
	input clk,
	input rst_n,
	input  [input_bitwidth -1: 0] $%Activation%$,
	input  [input_bitwidth     -1: 0] $%Weight%$,
	input  [input_bitwidth       -1: 0] $%Bias%$,
	input  [input_bitwidth-1:0] $%rate%$,
	output reg [bitwidth	   -1: 0] $%Output%$,
	);


	wire [bitwidth	   -1: 0] _$%Output%$;
//*Implementation of the logic*//


	always@(posedge clk) begin
		if(~rst_n)begin
			$%Output%$ <= _$%Output%$;
			_$%Activation%$ <= $%Activation%$;
			_$%Weight%$ <= $%Weight%$;
			_$%Bias%$ <= $%Bias%$;
		end else begin
			$%Output%$ <= 0;
			_$%Activation%$ <= 0;
			_$%Weight%$ <= 0;
			_$%Bias%$ <= 0;
		end
	end

endmodule
