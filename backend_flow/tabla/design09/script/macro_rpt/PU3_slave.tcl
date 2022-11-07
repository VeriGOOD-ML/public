# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 708.0 918.0 750.0 1000.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  706.0 916.0 752.0 1002.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  703.0 913.0 755.0 1005.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 760.0 947.0 796.0 1000.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  758.0 945.0 798.0 1002.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  755.0 942.0 801.0 1005.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_1__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 806.0 947.0 842.0 1000.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  804.0 945.0 844.0 1002.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  801.0 942.0 847.0 1005.0 -inst accelerator_unit/GEN_PU_3__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
