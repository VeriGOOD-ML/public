# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 1413.0 1901.5 1483.0 2136.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1411.0 1899.5 1485.0 2138.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1408.0 1896.5 1488.0 2141.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 1493.0 1901.5 1563.0 2136.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1491.0 1899.5 1565.0 2138.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1488.0 1896.5 1568.0 2141.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 1573.0 1901.5 1643.0 2136.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1571.0 1899.5 1645.0 2138.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1568.0 1896.5 1648.0 2141.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 1653.0 1901.5 1723.0 2136.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1651.0 1899.5 1725.0 2138.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1648.0 1896.5 1728.0 2141.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
