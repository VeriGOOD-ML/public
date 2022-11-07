# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 821.0 704.0 891.0 939.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  819.0 702.0 893.0 941.0 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  816.0 699.0 896.0 944.0 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 901.0 704.0 971.0 939.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  899.0 702.0 973.0 941.0 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  896.0 699.0 976.0 944.0 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 981.0 704.0 1051.0 939.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  979.0 702.0 1053.0 941.0 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  976.0 699.0 1056.0 944.0 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 1061.0 704.0 1131.0 939.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1059.0 702.0 1133.0 941.0 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1056.0 699.0 1136.0 944.0 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
