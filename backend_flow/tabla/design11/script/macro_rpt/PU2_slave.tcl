# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setObjFPlanBox Instance accelerator_unit/GEN_PU_2__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 562.0 694.0 604.0 763.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_2__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP].orient R0
createPlaceBlockage -type hard -box  560.0 692.0 606.0 765.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
createPlaceBlockage -type soft -box  557.0 689.0 609.0 768.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/fifo_wrt_to_bus/genblk1_fifo_bus_DP 
setObjFPlanBox Instance accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 614.0 694.0 650.0 740.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  612.0 692.0 652.0 742.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  609.0 689.0 655.0 745.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_3__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
setObjFPlanBox Instance accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 660.0 694.0 696.0 740.0 
dbSet [dbGet top.insts.name -p accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer].orient R0
createPlaceBlockage -type hard -box  658.0 692.0 698.0 742.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
createPlaceBlockage -type soft -box  655.0 689.0 701.0 745.0 -inst accelerator_unit/GEN_PU_2__pu_bus_slave_inst/read_from_bus/GEN_FIFO_0__genblk1_fifo_wrt_to_bus/genblk1_DP_buffer 
