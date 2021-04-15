//
//      verilog template for inner product in Axiline
//

module ip#(
    parameter bitwidth =16,
	parameter inputBitwidth=8,
    parameter size=8
)(
    input[inputBitwidth*size-1:0] x,
    input[inputBitwidth*size-1:0] w,
    input[bitwidth-1:0] psum,
    input sel,
    output [bitwidth-1:0]sum
);

    wire [bitwidth-1:0]psum_sel;
    assign psum_sel=sel?psum:0;

    genvar i;
    generate
        for (i=0;i<size;i=i+1)begin: LOOP
            wire [bitwidth-1:0]isum;
            if (i==0)begin
                assign isum=w[inputBitwidth-1:0]*x[inputBitwidth-1:0];
            end else begin
                assign isum=LOOP[i-1].isum+w[inputBitwidth*i+inputBitwidth-1:inputBitwidth*i]*x[inputBitwidth*i+inputBitwidth-1:inputBitwidth*i];
            end
        end
    endgenerate
    assign sum=LOOP[size-1].isum + psum_sel;

endmodule
