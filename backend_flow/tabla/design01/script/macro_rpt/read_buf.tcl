# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 1797.0 371.5 1867.0 606.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1795.0 369.5 1869.0 608.5 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1792.0 366.5 1872.0 611.5 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 1877.0 371.5 1947.0 606.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1875.0 369.5 1949.0 608.5 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1872.0 366.5 1952.0 611.5 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 1957.0 371.5 2027.0 606.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1955.0 369.5 2029.0 608.5 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1952.0 366.5 2032.0 611.5 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 2037.0 371.5 2107.0 606.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  2035.0 369.5 2109.0 608.5 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2032.0 366.5 2112.0 611.5 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
