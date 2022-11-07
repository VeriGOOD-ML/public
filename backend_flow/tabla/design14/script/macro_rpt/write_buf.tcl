# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 2421.0 1095.5 2491.0 1330.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2419.0 1093.5 2493.0 1332.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2416.0 1090.5 2496.0 1335.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 2501.0 1095.5 2571.0 1330.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2499.0 1093.5 2573.0 1332.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2496.0 1090.5 2576.0 1335.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 2581.0 1095.5 2651.0 1330.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2579.0 1093.5 2653.0 1332.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2576.0 1090.5 2656.0 1335.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 2661.0 1095.5 2731.0 1330.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  2659.0 1093.5 2733.0 1332.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  2656.0 1090.5 2736.0 1335.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
