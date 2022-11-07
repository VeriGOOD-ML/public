# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 949.0 435.0 1019.0 670.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  947.0 433.0 1021.0 672.0 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  944.0 430.0 1024.0 675.0 -inst u_mem_if/AXI_RD_BUF_3__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 1029.0 435.0 1099.0 670.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1027.0 433.0 1101.0 672.0 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1024.0 430.0 1104.0 675.0 -inst u_mem_if/AXI_RD_BUF_2__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 1109.0 435.0 1179.0 670.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1107.0 433.0 1181.0 672.0 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1104.0 430.0 1184.0 675.0 -inst u_mem_if/AXI_RD_BUF_1__read_buffer/genblk1_fifo_DP 
setObjFPlanBox Instance u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 1189.0 435.0 1259.0 670.0 
dbSet [dbGet top.insts.name -p u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP].orient R0
createPlaceBlockage -type hard -box  1187.0 433.0 1261.0 672.0 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
createPlaceBlockage -type soft -box  1184.0 430.0 1264.0 675.0 -inst u_mem_if/AXI_RD_BUF_0__read_buffer/genblk1_fifo_DP 
