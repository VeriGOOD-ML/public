`timescale 1ns / 1ps

module compute_unit
#(
    parameter OPCODE_BITS 			=	4,
	parameter FUNCTION_BITS 		=	4,
	
	parameter DATA_WIDTH            =   32
)(
    input clk,
    input reset,
    
    input [DATA_WIDTH-1 : 0] data_in0,
    input [DATA_WIDTH-1 : 0] data_in1,
    
    input [15:0]             integer_bits,
    
    output reg [DATA_WIDTH-1 : 0] data_out,
    
    input [OPCODE_BITS-1 : 0] opcode,
    input [FUNCTION_BITS-1 : 0] fn
    );
    
    wire [DATA_WIDTH-1 : 0] data_out_arith,data_out_calc,data_out_comp,data_out_cast;
    reg [DATA_WIDTH-1 :0] data_out_mux;
    
    arithmetic_unit arithmetic_inst (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .fn             (   fn              ),
        .data_in0       (   data_in0        ),
        .data_in1       (   data_in1        ),
        .data_acc       (   data_out        ),
        
        .mult_out_shift (   integer_bits[4:0] ),
        
        .data_out       (   data_out_arith  )
    );
    
    comparison_unit comparison_inst (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .fn             (   fn              ),
        .data_in0       (   data_in0        ),
        .data_in1       (   data_in1        ),
        .data_out       (   data_out_comp   )
    );
    
    calculus_unit calculus_inst (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .fn             (   fn              ),
        .data_in0       (   data_in0        ),
        .data_in1       (   data_in1        ),
        .data_out       (   data_out_calc   )
    );
    
    datatype_cast cast_inst (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .fn             (   fn              ),
        .immediate      (   data_in1[31:0]  ),
        .data_in        (   data_in0        ),
        .data_out       (   data_out_cast   )
    );
    
    
    always @(*) begin
        case(opcode)
            4'b0000: data_out_mux = data_out_arith;
            4'b0001: data_out_mux = data_out_calc;
            4'b0010: data_out_mux = data_out_comp;
            4'b0011: data_out_mux = data_out_cast;
            default: data_out_mux = 'd0;
        endcase
    end
    
    always @(posedge clk ) begin
        if(reset) begin
            data_out <= 'b0;
        end
        else begin
            data_out <=  data_out_mux;
        end
    end
    
endmodule
