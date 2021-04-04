`timescale 1ns / 1ps

module calculus_unit#(
    parameter FUNCTION_BITS 		=	4,
    parameter BIT_WIDTH      		=	32
)(
    input       clk,
    input       reset,
    
    input [FUNCTION_BITS-1 : 0] fn,
    
    input signed [BIT_WIDTH-1 : 0] data_in0,
    input signed [BIT_WIDTH-1 : 0] data_in1,
    
    output reg signed [BIT_WIDTH-1 : 0] data_out
    
    );
    
    wire gtz;
    wire signed [2*BIT_WIDTH-1:0] mult_out;
    
    wire [BIT_WIDTH-1 : 0]  sigmoid_out;
    wire [BIT_WIDTH-1 : 0]  tanh_out;
    
    assign gtz = ~data_in0[BIT_WIDTH-1];
    assign mult_out = data_in0 * $unsigned(data_in1);
    
    
    sigmoid #( .BIT_WIDTH(BIT_WIDTH)
    ) sigmoid_unit (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .data_in0       (   data_in0        ),
        .immediate      (   data_in1        ),
        .data_out       (   sigmoid_out     )
    );
    
    tanh #( .BIT_WIDTH(BIT_WIDTH)
    ) tanh_unit (
        .clk            (   clk             ),
        .reset          (   reset           ),
        
        .data_in0       (   data_in0        ),
        .immediate      (   data_in1        ),
        .data_out       (   tanh_out        )
    );
    
    always @(*) begin
      case(fn)
            4'b0000:    data_out = gtz ? data_in0 : 'd0;                        // ReLU
            4'b0001:    data_out = gtz ? data_in0 : mult_out[2*BIT_WIDTH-1:BIT_WIDTH] ;  // Leaky ReLU
            4'b0010:    data_out = sigmoid_out;  // Sigmoid
            4'b0011:    data_out = tanh_out;  // Tanh
            default:    data_out = 'd0;
            endcase
        end
endmodule