
module $algo$_asic #(
	//
	parameter	ACTIVATION_N				= $ActivationN$,
	parameter	ACTIVATION_BITWIDTH			= $ActivationBitwidth$,
	parameter	WEIGHT_N					= $WeightN$,
	parameter	WEIGHT_BITWIDTH				= $WeightBitwidth$,
	parameter	BIAS_N						= $BiasN$,
	parameter	BIAS_BITWIDTH				= $BiasBitwidth$,
	parameter	OUTPUT_N					= $OutputN$,
	parameter	OUTPUT_BITWIDTH				= $OutputBitwidth$
	//
)(
	input	clk,
	input	reset,
	input [ACTIVATION_BITWIDTH -1: 0] activation_in [0: ACTIVATION_N - 1],
	output[OUTPUT_BITWIDTH	   -1: 0] out [0: OUTPUT_N - 1]
);

	wire[WEIGHT_BITWIDTH - 1: 0] weight[0: WEIGHT_N - 1];
    wire[BIAS_BITWIDTH	 - 1: 0] bias[0: BIAS_N - 1];

// assigning the embedded weights and biases
	assign weight[%i%] = $Wi$;
	assign bias[%i%]   = $bi$;

	wire  [ACTIVATION_BITWIDTH -1: 0] _activation_in [0: ACTIVATION_N - 1];
	wire  [WEIGHT_BITWIDTH     -1: 0] _weight_in     [0: WEIGHT_N - 1];
	wire  [BIAS_BITWIDTH       -1: 0] _bias_in       [0: BIAS_N - 1];
	wire  [OUTPUT_BITWIDTH	   -1: 0] _out [0: OUTPUT_N - 1] ;
	assign _bias_in = bias;
	assign _activation_in = activation_in;
	assign _weight_in = weight;
	assign out = _out;

//*Implementation of the logic*//

endmodule