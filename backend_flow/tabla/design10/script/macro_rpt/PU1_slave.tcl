# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 10.0 1077.0 52.0 1214.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  8.0 1075.0 54.0 1216.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  5.0 1072.0 57.0 1219.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 62.0 1135.0 104.0 1214.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  60.0 1133.0 106.0 1216.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  57.0 1130.0 109.0 1219.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 114.0 1135.0 156.0 1214.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  112.0 1133.0 158.0 1216.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  109.0 1130.0 161.0 1219.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
