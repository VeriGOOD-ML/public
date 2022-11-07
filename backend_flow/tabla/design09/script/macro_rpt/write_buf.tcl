# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 1241.0 1040.5 1311.0 1275.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1239.0 1038.5 1313.0 1277.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1236.0 1035.5 1316.0 1280.5 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 1321.0 1040.5 1391.0 1275.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1319.0 1038.5 1393.0 1277.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1316.0 1035.5 1396.0 1280.5 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 1401.0 1040.5 1471.0 1275.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1399.0 1038.5 1473.0 1277.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1396.0 1035.5 1476.0 1280.5 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 1481.0 1040.5 1551.0 1275.5 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1479.0 1038.5 1553.0 1277.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1476.0 1035.5 1556.0 1280.5 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
