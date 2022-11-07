# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 1093.0 1084.0 1163.0 1319.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1091.0 1082.0 1165.0 1321.0 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1088.0 1079.0 1168.0 1324.0 -inst u_mem_if/AXI_RD_BUF_3__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 1173.0 1084.0 1243.0 1319.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1171.0 1082.0 1245.0 1321.0 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1168.0 1079.0 1248.0 1324.0 -inst u_mem_if/AXI_RD_BUF_2__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 1253.0 1084.0 1323.0 1319.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1251.0 1082.0 1325.0 1321.0 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1248.0 1079.0 1328.0 1324.0 -inst u_mem_if/AXI_RD_BUF_1__write_buffer/fifo_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 1333.0 1084.0 1403.0 1319.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP].orient R180
createPlaceBlockage -type hard -box  1331.0 1082.0 1405.0 1321.0 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1328.0 1079.0 1408.0 1324.0 -inst u_mem_if/AXI_RD_BUF_0__write_buffer/fifo_buffer/genblk1_fifo_DP 
