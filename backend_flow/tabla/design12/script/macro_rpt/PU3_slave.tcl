# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 634.0 947.0 676.0 1058.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  632.0 945.0 678.0 1060.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  629.0 942.0 681.0 1063.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 686.0 992.0 728.0 1058.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  684.0 990.0 730.0 1060.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  681.0 987.0 733.0 1063.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 738.0 992.0 780.0 1058.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  736.0 990.0 782.0 1060.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  733.0 987.0 785.0 1063.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
