
module nested_loop #(
    parameter NUM_MAX_LOOPS = 7,
    parameter LOG_NUM_MAX_LOOPS = 3,
    parameter BASE_WIDTH = 32,
    parameter STRIDE_WIDTH = BASE_WIDTH,
    parameter ADDRESS_WIDTH = BASE_WIDTH,
    parameter NUM_ITER_WIDTH = 32
)(
    
    input                                       clk,
    input                                       reset,
    
    input [BASE_WIDTH-1 : 0]                    base,
    input [STRIDE_WIDTH*NUM_MAX_LOOPS-1 : 0]    stride,
    input [NUM_ITER_WIDTH*NUM_MAX_LOOPS-1 : 0]  num_iter,
    input                                       start_loop,
    
    output [ADDRESS_WIDTH-1:0]                  address_out,
    output reg                                  address_valid,
    output reg                                  loop_done_out
    );
    
    reg start_loop_d,loop_done;
    wire [STRIDE_WIDTH-1:0] loop_stride[0:NUM_MAX_LOOPS-1];
    wire [NUM_ITER_WIDTH-1:0] max_iter[0:NUM_MAX_LOOPS-1];
    reg [NUM_ITER_WIDTH-1:0] iters[0:NUM_MAX_LOOPS-1];
    reg [ADDRESS_WIDTH-1:0] loop_address[0:NUM_MAX_LOOPS-1];
    reg [ADDRESS_WIDTH-1:0] loop_address_d[0:NUM_MAX_LOOPS-1];
    
    wire [NUM_MAX_LOOPS :0] iter_done;
    reg [NUM_MAX_LOOPS :0] iter_done_d;
    assign iter_done[NUM_MAX_LOOPS] = 1'b1;
    
    generate
    for(genvar i =0 ; i < NUM_MAX_LOOPS ; i = i +1) begin
        
        assign max_iter[i] = num_iter[i*NUM_ITER_WIDTH+:NUM_ITER_WIDTH];
        assign loop_stride[i] = stride[i*STRIDE_WIDTH+:STRIDE_WIDTH];
        
    end
    endgenerate
    
    always @(posedge clk) begin
        start_loop_d <= start_loop;
        if(reset)
            address_valid <= 1'b0;
        else if(start_loop)
            address_valid <= 1'b1;
        else if(iter_done[0])
            address_valid <= 1'b0;
    end
    
    always @(posedge clk) begin
        if(reset)
            loop_done <= 1'b0;
        else if(start_loop)
            loop_done <= 1'b0;
        else if(iter_done[0])
            loop_done <= 1'b1;
    end
    
    always @(posedge clk) begin
        loop_done_out <= iter_done[0];
        iter_done_d <= iter_done;
    end
    generate
    for(genvar i =0 ; i < NUM_MAX_LOOPS ; i = i +1) begin
        
        assign iter_done[i] = (iters[i] == max_iter[i]) && iter_done[i+1];
        
        always @( * ) begin
            if(start_loop) begin
                loop_address[i] = base;
            end
            else if(iter_done[i]) begin
                loop_address[i] = (i == 0) ? 'd0 : loop_address[i-1];
            end
            else if(iter_done[i+1]) begin
                loop_address[i] = loop_address_d[i] + loop_stride[i];
            end
            else begin
                loop_address[i] = loop_address_d[i];
            end
        end
        
        always @(posedge clk) begin
            if ( ~loop_done)
                loop_address_d[i] <= loop_address[i];
        end
        
        always @(posedge clk) begin
            if(start_loop_d || loop_done) begin
                iters[i] <= 'd0;
            end
            else if(iter_done[i]) begin
                iters[i] <= 'd0;
            end
            else if(iter_done[i+1]) begin
                iters[i] <= iters[i] + 'd1;
            end
        end
        
    end
    endgenerate
    
    assign address_out = loop_address[NUM_MAX_LOOPS-1];
endmodule
