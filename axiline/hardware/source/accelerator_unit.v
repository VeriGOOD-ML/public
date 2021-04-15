//
//      verilog template for pipeline stages in Axiline
//
`include "../include/config.vh"
`include "ip.v"
`include "comb.v"
`include "comb_reco.v"
`include "sgd.v"
`include "pipeline_reg.v"
`include "controller.v"

module accelerator_unit#(
    parameter inputBitwidth=`INPUT_BITWIDTH,
	parameter bitwidth=`BITWIDTH,
    parameter selBitwidth=1,
    parameter logNumCycle=`LOG_NUM_CYCLE,
    parameter numCycle=`NUM_CYCLE,
    parameter size= `SIZE
)(
    
    input [inputBitwidth*size-1:0]data_in_w,
    input [inputBitwidth*size-1:0]data_in_x,
    input [inputBitwidth-1:0]bias,
    input [inputBitwidth-1:0]rate,
    input clk,
    input start, 
    input rst,
	input [inputBitwidth-1:0]mu,
    output reg [bitwidth*size-1:0] data_out_r
);
    // controller
    wire [selBitwidth-1:0]sel;

    controller #(
        .logNumCycle(logNumCycle),
        .NumCycle(numCycle)
    )
    bm_controller(
        .clk(clk),
        .start(start),
        .rst(rst),
        .sel(sel)
    );

    //********************************//
    // pipeline stage #1
    //********************************//
    wire [bitwidth-1:0]sum;
    reg [bitwidth-1:0]sum_r;
    //block 1
    ip #(
        .bitwidth(bitwidth),
		.inputBitwidth(inputBitwidth),
        .size(size)
    )
    ip1(
        .x(data_in_x),
        .w(data_in_w),
        .psum(sum_r),
        .sel(sel[0]),
        .sum(sum)
    );
    always @(posedge clk) begin
        if (rst) sum_r<=0;
        else sum_r<=sum;
    end

    // pipeline registers for stage # 2

    // pipeline reg for bias
    wire [inputBitwidth-1:0] bias_r;
    pipe_reg #(
        .bitwidth(inputBitwidth),
        .pipe(numCycle)
    )
    pipe_reg_y(
        .clk(clk),
        .rst(rst),
        .data_in(bias),
        .data_out(bias_r)
    );

    // pipeline reg for rate
    `ifdef RECO
    wire [inputBitwidth-1:0] rate_r;
    pipe_reg #(
        .bitwidth(inputBitwidth),
        .pipe(numCycle)
    )
    pipe_reg_r(
        .clk(clk),
        .rst(rst),
        .data_in(rate),
        .data_out(rate_r)
    );
    `endif


    //********************************//
    // pipeline stage #2
    //********************************//
    wire valid;
    wire [bitwidth-1:0]grad;
    reg [bitwidth-1:0]grad_r;
    assign valid=~sel[0];

    //reco
    `ifdef RECO
        comb_reco #(
            .bitwidth(bitwidth),
            .inputBitwidth(inputBitwidth)
        )
        comb1(
            .data_in(sum),
            .bias(bias_r),
            .rate(rate_r)
            .valid(valid),
            .data_out(grad)
        );
    //SVM, linear regression and logistic regression
    `else
        comb #(
            .bitwidth(bitwidth),
            .inputBitwidth(inputBitwidth)
        )
        comb1(
            .data_in(sum),
            .bias(bias_r),
            .valid(valid),
            .data_out(grad)
        );
    `endif

    always @(posedge clk) begin
        if (rst) grad_r<=0;
        else if (valid)grad_r<=grad;
        else grad_r<=grad_r;
    end

    // pipeline registers for stage 3

    // pipeline reg for weight
    wire [inputBitwidth*size-1:0]data_w,data_w_r;
    assign data_w=data_in_w;

    pipe_reg #(
        .bitwidth(inputBitwidth*size),
        .pipe(numCycle+1)
    )
    pipe_reg_w(
        .clk(clk),
        .rst(rst),
        .data_in(data_w),
        .data_out(data_w_r)
    );

    // pipeline reg for activation
    wire [inputBitwidth*size-1:0]data_x,data_x_r;
    assign data_x=data_in_x;

    pipe_reg #(
        .bitwidth(inputBitwidth*size),
        .pipe(numCycle+1)
    )
    pipe_reg_x(
        .clk(clk),
        .rst(rst),
        .data_in(data_x),
        .data_out(data_x_r)
    );


    //********************************//
    // pipeline stage #3
    //********************************//
    wire [bitwidth-1:0]data_out[size-1:0];
    genvar i;
    generate
        for (i=0;i<size;i=i+1)begin: SGD
            sgd_unit #(
                .bitwidth(bitwidth),
				.inputBitwidth(inputBitwidth)
            )
            unit(
                .x(data_x_r[inputBitwidth*i+inputBitwidth-1:inputBitwidth*i]),
                .w(data_w_r[inputBitwidth*i+inputBitwidth-1:inputBitwidth*i]),
                .data_in(grad_r),
                .mu(mu),
                .data_out(data_out[i])
            );
        end
    endgenerate


    genvar i;
    generate
        for (i=0;i<size;i=i+1)begin: OUTPUT
            always @(posedge clk) begin
                if(~rst)begin
                 	data_out_r[i*bitwidth+bitwidth-1:i*bitwidth]<=data_out[i];
                end else begin
            		data_out_r[i*bitwidth+bitwidth-1:i*bitwidth]<=0;
        		end
            end
        end
    endgenerate

endmodule



