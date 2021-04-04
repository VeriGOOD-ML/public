`timescale 1ns/1ps
module bus_arbitrer_tb;


integer k;

reg clk,rstn;

always #5 clk =  ~clk;
	initial
begin
    $dumpfile("hw-imp/bin/waveform/arbitrer_tb.vcd");
    $dumpvars(0,bus_arbitrer_tb);
    for (k=0; k < 4; k = k+1) $dumpvars(0, bus_arbitrer_tb.PE[0].slave_inst.buffer_rd.reg_bank[k]);
end
initial begin
	clk = 1'b0;
	rstn = 1'b0;
	
	#50 rstn = 1'b1;
end
initial begin
#3000;
$finish;
end

parameter NUM_PE = 8;
parameter DATA_LEN = 16;
parameter BUS_ADDR_LEN = 3;

wire [DATA_LEN-1:0] bus_data;
wire [BUS_ADDR_LEN-1:0] addr_bus;

wire [BUS_ADDR_LEN-1:0] addr_to_bus[NUM_PE-1:0];
wire [NUM_PE-1:0] valid_to_bus,rd_buffer_full,wr_to_bus,rd_from_bus;

reg [BUS_ADDR_LEN-1:0] pe_dest_addr[NUM_PE-1:0];
reg [NUM_PE-1:0] pe_dest_valid,src_rq;
wire [NUM_PE-1:0] src_valid,fifo_full;
reg [DATA_LEN-1:0] pe_dest_data[NUM_PE-1:0];
reg [BUS_ADDR_LEN-1:0] pe_src_addr[NUM_PE-1:0];
wire [DATA_LEN-1:0] src_data[NUM_PE-1:0];

master_controller
controller_inst
(
	.clk		(clk		),
	.rstn		(rstn		),
	.addr_to_bus	(addr_to_bus	),
	.valid_to_bus	(valid_to_bus	),
	.rd_buffer_full	(rd_buffer_full	),

	.wr_to_bus	(wr_to_bus	),
	.rd_from_bus	(rd_from_bus	)
);

genvar i;

generate
for(i =0 ; i< NUM_PE;i=i+1) 
begin : PE
	slave_controller #( .PE_ID(i)) slave_inst(
		.clk		(clk		),
		.rstn		(rstn		),

		.dest_addr	(pe_dest_addr[i]),
		.dest_valid	(pe_dest_valid[i]	),
		.dest_data	(pe_dest_data[i]	),

		.src_addr	(pe_src_addr[i]		),
		.src_rq		(src_rq[i]		),
		
		.bus_data	(bus_data		),
		.addr_bus	(addr_bus		),
		.rd_from_bus	(rd_from_bus[i]		),

		.wr_to_bus	(wr_to_bus[i]		),
		
    //////////////////////
		.src_data	(src_data[i]		),
		.src_valid	(src_valid[i]		),
		
		.addr_to_bus	(addr_to_bus[i]		),	
		.valid_to_bus	(valid_to_bus[i]	),
		.wr_fifo_full	(fifo_full[i]		),
		.rd_buffer_full	(rd_buffer_full[i]	)
	);
end
endgenerate

integer seed;

generate
for(i =0 ; i< NUM_PE;i=i+1) begin
    initial begin
        pe_dest_addr[i] = (i+2)%NUM_PE;
        pe_dest_valid[i] = 1'b1;
        pe_dest_data[i] = i;
        seed = 1;
    end
    always @(posedge clk) begin
      pe_dest_addr[i] = $urandom(seed)%NUM_PE;
      pe_dest_valid[i] = ($urandom(seed)%16 == 0 ) ? 1'b1 : 1'b0;
      pe_dest_data[i] = (($urandom(seed)%256)<<8) + i ;
      pe_src_addr[i] = $urandom(seed)%NUM_PE;
      src_rq[i] = ($urandom(seed)%3== 0 ) ? 1'b1 : 1'b0;
    end
end
endgenerate
//always @(posedge clk)
  //seed = seed +1;
endmodule
