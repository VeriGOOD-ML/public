# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 934.0 820.0 970.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  932.0 818.0 972.0 870.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  929.0 815.0 975.0 873.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 980.0 822.0 1003.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  978.0 820.0 1005.0 870.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  975.0 817.0 1008.0 873.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 1013.0 822.0 1036.0 868.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  1011.0 820.0 1038.0 870.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  1008.0 817.0 1041.0 873.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 