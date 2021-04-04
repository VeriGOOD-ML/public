`timescale 1ns / 1ps

module arithmetic_unit#(
    parameter FUNCTION_BITS 		=	4,
    parameter BIT_WIDTH      		=	32
)(
    input       clk,
    input       reset,
    
    input [FUNCTION_BITS-1 : 0] fn,
    
    input signed [BIT_WIDTH-1 : 0] data_in0,
    input signed [BIT_WIDTH-1 : 0] data_in1,
    input signed [BIT_WIDTH-1 : 0] data_acc,
    
    input [4 : 0] mult_out_shift,
    
    output reg signed [BIT_WIDTH-1 : 0] data_out
    
    );
    
    wire signed [BIT_WIDTH:0] sum_out,sub_out,acc_out;
    reg signed [BIT_WIDTH-1:0] sum_final,sub_final,acc_final;
    wire signed [2*BIT_WIDTH-1:0] mult_out_shifted,mult_out_temp;
    reg [BIT_WIDTH-1:0] mult_out;
    wire [BIT_WIDTH-1:0] div_out;
    
    assign sum_out = data_in0 + data_in1;
    assign sub_out = data_in0 - data_in1;
 
    always @(*) begin
        case( sum_out[BIT_WIDTH:BIT_WIDTH-1])
            2'b01 : sum_final = {1'b0,{BIT_WIDTH-'d1{1'b1}}};
            2'b10 : sum_final = {1'b1,{BIT_WIDTH-'d1{1'b0}}};
            default : sum_final = sum_out[BIT_WIDTH-1 : 0];
        endcase
    end
    
    always @(*) begin
        case( sub_out[BIT_WIDTH:BIT_WIDTH-1])
            2'b01 : sub_final = {1'b0,{BIT_WIDTH-'d1{1'b1}}};
            2'b10 : sub_final = {1'b1,{BIT_WIDTH-'d1{1'b0}}};
            default : sub_final = sub_out[BIT_WIDTH-1 : 0];
        endcase
    end
    
    assign mult_out_temp = data_in0 * data_in1;
    assign mult_out_shifted = mult_out_temp >>> mult_out_shift;
    
    wire zeros,ones;
    assign zeros = |mult_out_shifted[2*BIT_WIDTH-2:BIT_WIDTH-1];
    assign ones = &mult_out_shifted[2*BIT_WIDTH-2:BIT_WIDTH-1];
    
    always @(*) begin
        case({mult_out_shifted[2*BIT_WIDTH-1],ones,zeros} )
            3'b001,3'b011 : mult_out = {1'b0,{BIT_WIDTH-1{1'b1}}};
            3'b100,3'b101 : mult_out = {1'b1,{BIT_WIDTH-1{1'b0}}};
            default : mult_out = mult_out_shifted[BIT_WIDTH-1 : 0];
        endcase
    end
    
    assign acc_out = mult_out + data_acc;
    always @(*) begin
        case( acc_out[BIT_WIDTH:BIT_WIDTH-1])
            2'b01 : acc_final = {1'b0,{BIT_WIDTH-1{1'b1}}};
            2'b10 : acc_final = {1'b1,{BIT_WIDTH-1{1'b0}}};
            default : acc_final = acc_out[BIT_WIDTH-1 : 0];
        endcase
    end
    always @(*) begin
        case(fn)
            4'b0000:    data_out = sum_final;
            4'b0001:    data_out = sub_final;
            4'b0010:    data_out = mult_out ;
            4'b0011:    data_out = acc_final;
            4'b0100:    data_out = div_out;
            4'b0101,4'b110:  begin
                if ( data_in0 > data_in1)
                    data_out = fn[0] ? data_in0 : data_in1;
                else
                    data_out = fn[0] ? data_in1 : data_in0;
            end
            4'b0111:    data_out = data_in0 >> $unsigned(data_in1[4:0]);
            4'b1000:    data_out = data_in0 << $unsigned(data_in1[4:0]);
            4'b1001:    data_out = data_in0 ;
            4'b1010:    data_out = data_in0 ;
            4'b1011:    data_out = data_in0 ;
            4'b1100:    data_out = ~data_in0 ;
            4'b1101:    data_out = data_in0 & data_in1 ;
            4'b1110:    data_out = data_in0 | data_in1 ;
            default:    data_out = data_in0;
        endcase        
    end
    
//    DW_div_pipe #(.a_width(BIT_WIDTH),
//                  .b_width(BIT_WIDTH),
//                  .tc_mode(1),
//                  .rem_mode(1),
//                  .num_stages(4))
//      pipelined_divider (
//          .clk          (   clk         ),
//          .rst_n        (   ~reset      ),
//          .a            (   data_in0    ),
//          .b            (   data_in1    ),
//          .en            (   1'b1        ),
          
//          .quotient     (   div_out     ),
//          .remainder    (               ),
//          .divide_by_0  (               )
//	  );
	  assign div_out = 'b0;
	  
endmodule