# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 2053.0 741.5 2123.0 976.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2051.0 739.5 2125.0 978.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2048.0 736.5 2128.0 981.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 2133.0 741.5 2203.0 976.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2131.0 739.5 2205.0 978.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2128.0 736.5 2208.0 981.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 2213.0 741.5 2283.0 976.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2211.0 739.5 2285.0 978.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2208.0 736.5 2288.0 981.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 2293.0 741.5 2363.0 976.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2291.0 739.5 2365.0 978.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2288.0 736.5 2368.0 981.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
