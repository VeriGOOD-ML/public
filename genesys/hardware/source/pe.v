//
// Processing Element, the input pipelining can be activated for both inputs and weights
//
// Soroush Ghodrati
// (soghodra@eng.ucsd.edu)

`timescale 1ns/1ps

module pe #(
	parameter					WMEM_ADDR_BITWIDTH				= 8,
	parameter					ACT_BITWIDTH					= 8,
	parameter					WGT_BITWIDTH					= 8,
	parameter					SUM_IN_BITWIDTH					= 16,
	parameter					INTER_BITWIDTH					= 17,
	parameter					TRUNCATION_MODE					= "MSB",
	parameter					ACT_PIPELINE					= "True",
	parameter                   PE_M                            = 0,
	parameter                   PE_N                            = 0,
	parameter					SUM_OUT_BITWIDTH				= SUM_IN_BITWIDTH
)(
	input														clk,
	input														reset,
	input						[ACT_BITWIDTH      -1: 0]		act_in,
	input						[SUM_IN_BITWIDTH   -1: 0]		sum_in,
	input														read_req_w_mem,
	input						[WMEM_ADDR_BITWIDTH-1: 0]		r_addr_w_mem,
	input														write_req_w_mem,
	input						[WMEM_ADDR_BITWIDTH-1: 0]		w_addr_w_mem,
	input						[WGT_BITWIDTH      -1: 0]		w_data_w_mem,
	output														read_req_w_mem_frwrd,
	output						[WMEM_ADDR_BITWIDTH-1: 0]		r_addr_w_mem_frwrd,
	output						[ACT_BITWIDTH	   -1: 0]		act_out,
	output						[SUM_OUT_BITWIDTH  -1: 0]		sum_out
);
	
	wire						[WGT_BITWIDTH	  -1: 0]		wgt_read;
	
	scratchpad #(
		.DATA_BITWIDTH											(WGT_BITWIDTH),
		.ADDR_BITWIDTH											(WMEM_ADDR_BITWIDTH)
	) weight_scratchpad (
		.clk													(clk),
		.reset													(reset),
		.read_req												(read_req_w_mem),
		.write_req   											(write_req_w_mem),
		.r_addr													(r_addr_w_mem),
		.w_addr													(w_addr_w_mem),
		.w_data													(w_data_w_mem),
		.r_data													(wgt_read)
	);
//	
// weight-stationary logic	

//	wire						[WGT_BITWIDTH	  -1: 0]		_wgt_read_reg;
//	reg						    [WGT_BITWIDTH	  -1: 0]		_wgt_read;
//	
//	register #(
//		.BIT_WIDTH										(WGT_BITWIDTH)
//	) register_ws(
//		.clk											(clk),
//		.reset											(reset_ws_reg),
//		.wrt_en											(ws_en),
//		.data_in										(_wgt_read_),
//		.data_out 										(_wgt_read_reg)	
//		);
//	always @ (*) begin	
//		if (ws_en == 0) begin
//			_wgt_read	=	_wgt_read_;
//		end
//		// if ws, at the first cycle the data directs to the macc logic,
//		//but for the rest cycles will be read from the register
//
//		if (ws_en == 1 && ws_mux == 1) begin			
//			_wgt_read	=	_wgt_read_;
//		end
//		
//		if (ws_en == 1 && ws_mux == 0) begin
//			_wgt_read	=	_wgt_read_reg;
//		end 
//	end
//	
//	
	
	
	wire						[INTER_BITWIDTH   -1: 0]		_macc_out;
	
	macc #(
		.ACT_BITWIDTH											(ACT_BITWIDTH),
		.WGT_BITWIDTH											(WGT_BITWIDTH),
		.SUM_IN_BITWIDTH										(SUM_IN_BITWIDTH),
		.INTER_BITWIDTH											(INTER_BITWIDTH)
	) macc_inst (
		.a_in													(act_in),
		.w_in													(wgt_read),
		.sum_in													(sum_in),
		.out													(_macc_out)
	);
		
	wire						[SUM_OUT_BITWIDTH -1: 0]		_truncator_out;
	
	truncator #(
		.TRUNCATION_MODE										(TRUNCATION_MODE),
		.DATA_IN_BITWIDTH										(INTER_BITWIDTH),
		.DATA_OUT_BITWIDTH										(SUM_OUT_BITWIDTH)
	) truncator_inst (
		.data_in												(_macc_out),
		.data_out												(_truncator_out)
	);
	


	if (ACT_PIPELINE == "True")
	begin	
	register_sync #(
		.WIDTH 													(ACT_BITWIDTH)
	) register_act_out(
		.clk													(clk),
		.reset 													(reset),
		.in 													(act_in),
		.out 													(act_out)
		);
		
	register_sync #(
		.WIDTH 													(1)
	) register_wmem_rd_req_frwrd(
		.clk													(clk),
		.reset 													(reset),
		.in 													(read_req_w_mem),
		.out 													(read_req_w_mem_frwrd)
		);
	
	register_sync #(
		.WIDTH 													(WMEM_ADDR_BITWIDTH)
	) register_wmem_rd_addr_frwrd(
		.clk													(clk),
		.reset 													(reset),
		.in 													(r_addr_w_mem),
		.out 													(r_addr_w_mem_frwrd)
		);	
			
	end
	else //if (ACT_PIPELINE == "False")
	begin
		assign act_out = act_in;
		
		if(PE_M == 0) begin
		    register_sync #(
                .WIDTH 													(1)
            ) register_wmem_rd_req_frwrd(
                .clk													(clk),
                .reset 													(reset),
                .in 													(read_req_w_mem),
                .out 													(read_req_w_mem_frwrd)
                );
            
            register_sync #(
                .WIDTH 													(WMEM_ADDR_BITWIDTH)
            ) register_wmem_rd_addr_frwrd(
                .clk													(clk),
                .reset 													(reset),
                .in 													(r_addr_w_mem),
                .out 													(r_addr_w_mem_frwrd)
                );	
		end
		else begin
            assign read_req_w_mem_frwrd = read_req_w_mem;
            assign r_addr_w_mem_frwrd = r_addr_w_mem;
        end
	end
	register_sync #(
		.WIDTH													(SUM_OUT_BITWIDTH)
	) register_out(
		.clk													(clk),
		.reset													(reset),
		.in														(_truncator_out),
		.out													(sum_out)	
	);

	
endmodule
