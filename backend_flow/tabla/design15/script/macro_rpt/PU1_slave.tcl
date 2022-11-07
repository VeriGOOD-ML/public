# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 10.0 820.0 46.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  8.0 818.0 48.0 870.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  5.0 815.0 51.0 873.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 56.0 822.0 79.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  54.0 820.0 81.0 870.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  51.0 817.0 84.0 873.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 89.0 822.0 112.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  87.0 820.0 114.0 870.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  84.0 817.0 117.0 873.0 -inst accelerator_unit/GEN_PU_1__pu_bus_slave_inst/read_from_bus/GEN_FIFO_2__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
