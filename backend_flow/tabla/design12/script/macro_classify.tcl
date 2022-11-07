# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
proc Write_Macro_File { macro_file macro_count cell_name  instance_name  size_x  size_y } {
    set fp [open $macro_file a+]
    puts $fp "$cell_name,$instance_name,$size_x,$size_y"
    close $fp
    incr macro_count
    
    return $macro_count
}



if {![file exists macro_rpt/]} {
   exec mkdir macro_rpt/
}
set PU0_slave_macro 0
set PU0_PE0_macro  0
set PU0_PE0_slave_macro 0
set PU0_PE1_macro  0
set PU0_PE1_slave_macro 0
set PU0_PE2_macro  0
set PU0_PE2_slave_macro 0
set PU0_PE3_macro  0
set PU0_PE3_slave_macro 0
set PU0_PE4_macro  0
set PU0_PE4_slave_macro 0
set PU0_PE5_macro  0
set PU0_PE5_slave_macro 0
set PU0_PE6_macro  0
set PU0_PE6_slave_macro 0
set PU0_PE7_macro  0
set PU0_PE7_slave_macro 0
set PU1_slave_macro 0
set PU1_PE0_macro  0
set PU1_PE0_slave_macro 0
set PU1_PE1_macro  0
set PU1_PE1_slave_macro 0
set PU1_PE2_macro  0
set PU1_PE2_slave_macro 0
set PU1_PE3_macro  0
set PU1_PE3_slave_macro 0
set PU1_PE4_macro  0
set PU1_PE4_slave_macro 0
set PU1_PE5_macro  0
set PU1_PE5_slave_macro 0
set PU1_PE6_macro  0
set PU1_PE6_slave_macro 0
set PU1_PE7_macro  0
set PU1_PE7_slave_macro 0
set PU2_slave_macro 0
set PU2_PE0_macro  0
set PU2_PE0_slave_macro 0
set PU2_PE1_macro  0
set PU2_PE1_slave_macro 0
set PU2_PE2_macro  0
set PU2_PE2_slave_macro 0
set PU2_PE3_macro  0
set PU2_PE3_slave_macro 0
set PU2_PE4_macro  0
set PU2_PE4_slave_macro 0
set PU2_PE5_macro  0
set PU2_PE5_slave_macro 0
set PU2_PE6_macro  0
set PU2_PE6_slave_macro 0
set PU2_PE7_macro  0
set PU2_PE7_slave_macro 0
set PU3_slave_macro 0
set PU3_PE0_macro  0
set PU3_PE0_slave_macro 0
set PU3_PE1_macro  0
set PU3_PE1_slave_macro 0
set PU3_PE2_macro  0
set PU3_PE2_slave_macro 0
set PU3_PE3_macro  0
set PU3_PE3_slave_macro 0
set PU3_PE4_macro  0
set PU3_PE4_slave_macro 0
set PU3_PE5_macro  0
set PU3_PE5_slave_macro 0
set PU3_PE6_macro  0
set PU3_PE6_slave_macro 0
set PU3_PE7_macro  0
set PU3_PE7_slave_macro 0
set read_macro 0
set write_macro 0
set other_macro 0
set total_macro 0


set fp [ open "macro_rpt/macro_PU0_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE0.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE0_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE1.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE1_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE2.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE2_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE3.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE3_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE4.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE4_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE5.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE5_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE6.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE6_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE7.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU0_PE7_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE0.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE0_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE1.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE1_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE2.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE2_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE3.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE3_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE4.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE4_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE5.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE5_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE6.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE6_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE7.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU1_PE7_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE0.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE0_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE1.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE1_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE2.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE2_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE3.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE3_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE4.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE4_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE5.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE5_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE6.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE6_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE7.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU2_PE7_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE0.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE0_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE1.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE1_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE2.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE2_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE3.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE3_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE4.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE4_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE5.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE5_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE6.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE6_slave.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE7.txt" w]
close $fp

set fp [ open "macro_rpt/macro_PU3_PE7_slave.txt" w]
close $fp

set fp [ open "macro_rpt/read_macro.txt" w]
close $fp
set fp [ open "macro_rpt/write_macro.txt" w]
close $fp
set fp [ open "macro_rpt/other_macro.txt" w]
close $fp


set insts [dbGet top.insts]
foreach inst $insts {
    set inst_base [dbGet $inst.cell.baseClass]
    if {$inst_base == "block"} {
        set cell_name [dbGet $inst.cell.name]
        set instance_name [dbGet $inst.defName]
        set size_x [dbGet $inst.cell.size_x]
        set size_y [dbGet $inst.cell.size_y]
        set other_macro_flag 1
        incr total_macro
        set PU0   [regexp {(.*)GEN_PU_0__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0 ]
        if $PU0  {
            set PU0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_slave.txt"             $PU0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE0   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE0 ]
        if $PU0_PE0  {
            set PU0_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE0.txt"                 $PU0_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE0_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE0_slave ]
        if $PU0_PE0_slave  {
            set PU0_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE0_slave.txt"                 $PU0_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE1   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE1 ]
        if $PU0_PE1  {
            set PU0_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE1.txt"                 $PU0_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE1_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE1_slave ]
        if $PU0_PE1_slave  {
            set PU0_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE1_slave.txt"                 $PU0_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE2   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE2 ]
        if $PU0_PE2  {
            set PU0_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE2.txt"                 $PU0_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE2_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE2_slave ]
        if $PU0_PE2_slave  {
            set PU0_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE2_slave.txt"                 $PU0_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE3   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE3 ]
        if $PU0_PE3  {
            set PU0_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE3.txt"                 $PU0_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE3_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE3_slave ]
        if $PU0_PE3_slave  {
            set PU0_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE3_slave.txt"                 $PU0_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE4   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE4 ]
        if $PU0_PE4  {
            set PU0_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE4.txt"                 $PU0_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE4_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE4_slave ]
        if $PU0_PE4_slave  {
            set PU0_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE4_slave.txt"                 $PU0_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE5   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE5 ]
        if $PU0_PE5  {
            set PU0_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE5.txt"                 $PU0_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE5_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE5_slave ]
        if $PU0_PE5_slave  {
            set PU0_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE5_slave.txt"                 $PU0_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE6   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE6 ]
        if $PU0_PE6  {
            set PU0_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE6.txt"                 $PU0_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE6_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE6_slave ]
        if $PU0_PE6_slave  {
            set PU0_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE6_slave.txt"                 $PU0_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE7   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE7 ]
        if $PU0_PE7  {
            set PU0_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE7.txt"                 $PU0_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU0_PE7_slave   [regexp {(.*)GEN_PU_0__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU0_PE7_slave ]
        if $PU0_PE7_slave  {
            set PU0_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU0_PE7_slave.txt"                 $PU0_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1   [regexp {(.*)GEN_PU_1__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1 ]
        if $PU1  {
            set PU1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_slave.txt"             $PU1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE0   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE0 ]
        if $PU1_PE0  {
            set PU1_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE0.txt"                 $PU1_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE0_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE0_slave ]
        if $PU1_PE0_slave  {
            set PU1_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE0_slave.txt"                 $PU1_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE1   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE1 ]
        if $PU1_PE1  {
            set PU1_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE1.txt"                 $PU1_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE1_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE1_slave ]
        if $PU1_PE1_slave  {
            set PU1_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE1_slave.txt"                 $PU1_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE2   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE2 ]
        if $PU1_PE2  {
            set PU1_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE2.txt"                 $PU1_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE2_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE2_slave ]
        if $PU1_PE2_slave  {
            set PU1_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE2_slave.txt"                 $PU1_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE3   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE3 ]
        if $PU1_PE3  {
            set PU1_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE3.txt"                 $PU1_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE3_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE3_slave ]
        if $PU1_PE3_slave  {
            set PU1_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE3_slave.txt"                 $PU1_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE4   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE4 ]
        if $PU1_PE4  {
            set PU1_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE4.txt"                 $PU1_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE4_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE4_slave ]
        if $PU1_PE4_slave  {
            set PU1_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE4_slave.txt"                 $PU1_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE5   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE5 ]
        if $PU1_PE5  {
            set PU1_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE5.txt"                 $PU1_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE5_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE5_slave ]
        if $PU1_PE5_slave  {
            set PU1_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE5_slave.txt"                 $PU1_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE6   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE6 ]
        if $PU1_PE6  {
            set PU1_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE6.txt"                 $PU1_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE6_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE6_slave ]
        if $PU1_PE6_slave  {
            set PU1_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE6_slave.txt"                 $PU1_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE7   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE7 ]
        if $PU1_PE7  {
            set PU1_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE7.txt"                 $PU1_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU1_PE7_slave   [regexp {(.*)GEN_PU_1__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU1_PE7_slave ]
        if $PU1_PE7_slave  {
            set PU1_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU1_PE7_slave.txt"                 $PU1_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2   [regexp {(.*)GEN_PU_2__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2 ]
        if $PU2  {
            set PU2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_slave.txt"             $PU2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE0   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE0 ]
        if $PU2_PE0  {
            set PU2_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE0.txt"                 $PU2_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE0_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE0_slave ]
        if $PU2_PE0_slave  {
            set PU2_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE0_slave.txt"                 $PU2_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE1   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE1 ]
        if $PU2_PE1  {
            set PU2_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE1.txt"                 $PU2_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE1_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE1_slave ]
        if $PU2_PE1_slave  {
            set PU2_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE1_slave.txt"                 $PU2_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE2   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE2 ]
        if $PU2_PE2  {
            set PU2_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE2.txt"                 $PU2_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE2_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE2_slave ]
        if $PU2_PE2_slave  {
            set PU2_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE2_slave.txt"                 $PU2_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE3   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE3 ]
        if $PU2_PE3  {
            set PU2_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE3.txt"                 $PU2_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE3_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE3_slave ]
        if $PU2_PE3_slave  {
            set PU2_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE3_slave.txt"                 $PU2_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE4   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE4 ]
        if $PU2_PE4  {
            set PU2_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE4.txt"                 $PU2_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE4_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE4_slave ]
        if $PU2_PE4_slave  {
            set PU2_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE4_slave.txt"                 $PU2_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE5   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE5 ]
        if $PU2_PE5  {
            set PU2_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE5.txt"                 $PU2_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE5_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE5_slave ]
        if $PU2_PE5_slave  {
            set PU2_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE5_slave.txt"                 $PU2_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE6   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE6 ]
        if $PU2_PE6  {
            set PU2_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE6.txt"                 $PU2_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE6_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE6_slave ]
        if $PU2_PE6_slave  {
            set PU2_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE6_slave.txt"                 $PU2_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE7   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE7 ]
        if $PU2_PE7  {
            set PU2_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE7.txt"                 $PU2_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU2_PE7_slave   [regexp {(.*)GEN_PU_2__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU2_PE7_slave ]
        if $PU2_PE7_slave  {
            set PU2_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU2_PE7_slave.txt"                 $PU2_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3   [regexp {(.*)GEN_PU_3__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3 ]
        if $PU3  {
            set PU3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_slave.txt"             $PU3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE0   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE0 ]
        if $PU3_PE0  {
            set PU3_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE0.txt"                 $PU3_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE0_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE0_slave ]
        if $PU3_PE0_slave  {
            set PU3_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE0_slave.txt"                 $PU3_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE1   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE1 ]
        if $PU3_PE1  {
            set PU3_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE1.txt"                 $PU3_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE1_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE1_slave ]
        if $PU3_PE1_slave  {
            set PU3_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE1_slave.txt"                 $PU3_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE2   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE2 ]
        if $PU3_PE2  {
            set PU3_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE2.txt"                 $PU3_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE2_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE2_slave ]
        if $PU3_PE2_slave  {
            set PU3_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE2_slave.txt"                 $PU3_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE3   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE3 ]
        if $PU3_PE3  {
            set PU3_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE3.txt"                 $PU3_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE3_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE3_slave ]
        if $PU3_PE3_slave  {
            set PU3_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE3_slave.txt"                 $PU3_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE4   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE4 ]
        if $PU3_PE4  {
            set PU3_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE4.txt"                 $PU3_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE4_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE4_slave ]
        if $PU3_PE4_slave  {
            set PU3_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE4_slave.txt"                 $PU3_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE5   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE5 ]
        if $PU3_PE5  {
            set PU3_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE5.txt"                 $PU3_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE5_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE5_slave ]
        if $PU3_PE5_slave  {
            set PU3_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE5_slave.txt"                 $PU3_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE6   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE6 ]
        if $PU3_PE6  {
            set PU3_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE6.txt"                 $PU3_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE6_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE6_slave ]
        if $PU3_PE6_slave  {
            set PU3_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE6_slave.txt"                 $PU3_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE7   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE7 ]
        if $PU3_PE7  {
            set PU3_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE7.txt"                 $PU3_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU3_PE7_slave   [regexp {(.*)GEN_PU_3__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU3_PE7_slave ]
        if $PU3_PE7_slave  {
            set PU3_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU3_PE7_slave.txt"                 $PU3_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4   [regexp {(.*)GEN_PU_4__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4 ]
        if $PU4  {
            set PU4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_slave.txt"             $PU4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE0   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE0 ]
        if $PU4_PE0  {
            set PU4_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE0.txt"                 $PU4_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE0_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE0_slave ]
        if $PU4_PE0_slave  {
            set PU4_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE0_slave.txt"                 $PU4_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE1   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE1 ]
        if $PU4_PE1  {
            set PU4_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE1.txt"                 $PU4_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE1_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE1_slave ]
        if $PU4_PE1_slave  {
            set PU4_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE1_slave.txt"                 $PU4_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE2   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE2 ]
        if $PU4_PE2  {
            set PU4_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE2.txt"                 $PU4_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE2_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE2_slave ]
        if $PU4_PE2_slave  {
            set PU4_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE2_slave.txt"                 $PU4_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE3   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE3 ]
        if $PU4_PE3  {
            set PU4_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE3.txt"                 $PU4_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE3_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE3_slave ]
        if $PU4_PE3_slave  {
            set PU4_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE3_slave.txt"                 $PU4_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE4   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE4 ]
        if $PU4_PE4  {
            set PU4_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE4.txt"                 $PU4_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE4_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE4_slave ]
        if $PU4_PE4_slave  {
            set PU4_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE4_slave.txt"                 $PU4_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE5   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE5 ]
        if $PU4_PE5  {
            set PU4_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE5.txt"                 $PU4_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE5_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE5_slave ]
        if $PU4_PE5_slave  {
            set PU4_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE5_slave.txt"                 $PU4_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE6   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE6 ]
        if $PU4_PE6  {
            set PU4_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE6.txt"                 $PU4_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE6_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE6_slave ]
        if $PU4_PE6_slave  {
            set PU4_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE6_slave.txt"                 $PU4_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE7   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE7 ]
        if $PU4_PE7  {
            set PU4_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE7.txt"                 $PU4_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU4_PE7_slave   [regexp {(.*)GEN_PU_4__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU4_PE7_slave ]
        if $PU4_PE7_slave  {
            set PU4_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU4_PE7_slave.txt"                 $PU4_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5   [regexp {(.*)GEN_PU_5__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5 ]
        if $PU5  {
            set PU5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_slave.txt"             $PU5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE0   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE0 ]
        if $PU5_PE0  {
            set PU5_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE0.txt"                 $PU5_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE0_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE0_slave ]
        if $PU5_PE0_slave  {
            set PU5_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE0_slave.txt"                 $PU5_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE1   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE1 ]
        if $PU5_PE1  {
            set PU5_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE1.txt"                 $PU5_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE1_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE1_slave ]
        if $PU5_PE1_slave  {
            set PU5_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE1_slave.txt"                 $PU5_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE2   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE2 ]
        if $PU5_PE2  {
            set PU5_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE2.txt"                 $PU5_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE2_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE2_slave ]
        if $PU5_PE2_slave  {
            set PU5_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE2_slave.txt"                 $PU5_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE3   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE3 ]
        if $PU5_PE3  {
            set PU5_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE3.txt"                 $PU5_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE3_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE3_slave ]
        if $PU5_PE3_slave  {
            set PU5_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE3_slave.txt"                 $PU5_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE4   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE4 ]
        if $PU5_PE4  {
            set PU5_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE4.txt"                 $PU5_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE4_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE4_slave ]
        if $PU5_PE4_slave  {
            set PU5_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE4_slave.txt"                 $PU5_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE5   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE5 ]
        if $PU5_PE5  {
            set PU5_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE5.txt"                 $PU5_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE5_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE5_slave ]
        if $PU5_PE5_slave  {
            set PU5_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE5_slave.txt"                 $PU5_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE6   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE6 ]
        if $PU5_PE6  {
            set PU5_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE6.txt"                 $PU5_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE6_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE6_slave ]
        if $PU5_PE6_slave  {
            set PU5_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE6_slave.txt"                 $PU5_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE7   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE7 ]
        if $PU5_PE7  {
            set PU5_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE7.txt"                 $PU5_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU5_PE7_slave   [regexp {(.*)GEN_PU_5__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU5_PE7_slave ]
        if $PU5_PE7_slave  {
            set PU5_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU5_PE7_slave.txt"                 $PU5_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6   [regexp {(.*)GEN_PU_6__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6 ]
        if $PU6  {
            set PU6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_slave.txt"             $PU6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE0   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE0 ]
        if $PU6_PE0  {
            set PU6_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE0.txt"                 $PU6_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE0_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE0_slave ]
        if $PU6_PE0_slave  {
            set PU6_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE0_slave.txt"                 $PU6_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE1   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE1 ]
        if $PU6_PE1  {
            set PU6_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE1.txt"                 $PU6_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE1_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE1_slave ]
        if $PU6_PE1_slave  {
            set PU6_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE1_slave.txt"                 $PU6_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE2   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE2 ]
        if $PU6_PE2  {
            set PU6_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE2.txt"                 $PU6_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE2_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE2_slave ]
        if $PU6_PE2_slave  {
            set PU6_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE2_slave.txt"                 $PU6_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE3   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE3 ]
        if $PU6_PE3  {
            set PU6_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE3.txt"                 $PU6_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE3_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE3_slave ]
        if $PU6_PE3_slave  {
            set PU6_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE3_slave.txt"                 $PU6_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE4   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE4 ]
        if $PU6_PE4  {
            set PU6_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE4.txt"                 $PU6_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE4_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE4_slave ]
        if $PU6_PE4_slave  {
            set PU6_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE4_slave.txt"                 $PU6_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE5   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE5 ]
        if $PU6_PE5  {
            set PU6_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE5.txt"                 $PU6_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE5_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE5_slave ]
        if $PU6_PE5_slave  {
            set PU6_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE5_slave.txt"                 $PU6_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE6   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE6 ]
        if $PU6_PE6  {
            set PU6_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE6.txt"                 $PU6_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE6_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE6_slave ]
        if $PU6_PE6_slave  {
            set PU6_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE6_slave.txt"                 $PU6_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE7   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE7 ]
        if $PU6_PE7  {
            set PU6_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE7.txt"                 $PU6_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU6_PE7_slave   [regexp {(.*)GEN_PU_6__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU6_PE7_slave ]
        if $PU6_PE7_slave  {
            set PU6_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU6_PE7_slave.txt"                 $PU6_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7   [regexp {(.*)GEN_PU_7__pu_bus_slave_inst(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7 ]
        if $PU7  {
            set PU7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_slave.txt"             $PU7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE0   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_0__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE0 ]
        if $PU7_PE0  {
            set PU7_PE0_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE0.txt"                 $PU7_PE0_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE0_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_0__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE0_slave ]
        if $PU7_PE0_slave  {
            set PU7_PE0_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE0_slave.txt"                 $PU7_PE0_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE1   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_1__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE1 ]
        if $PU7_PE1  {
            set PU7_PE1_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE1.txt"                 $PU7_PE1_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE1_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_1__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE1_slave ]
        if $PU7_PE1_slave  {
            set PU7_PE1_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE1_slave.txt"                 $PU7_PE1_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE2   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_2__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE2 ]
        if $PU7_PE2  {
            set PU7_PE2_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE2.txt"                 $PU7_PE2_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE2_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_2__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE2_slave ]
        if $PU7_PE2_slave  {
            set PU7_PE2_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE2_slave.txt"                 $PU7_PE2_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE3   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_3__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE3 ]
        if $PU7_PE3  {
            set PU7_PE3_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE3.txt"                 $PU7_PE3_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE3_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_3__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE3_slave ]
        if $PU7_PE3_slave  {
            set PU7_PE3_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE3_slave.txt"                 $PU7_PE3_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE4   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_4__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE4 ]
        if $PU7_PE4  {
            set PU7_PE4_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE4.txt"                 $PU7_PE4_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE4_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_4__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE4_slave ]
        if $PU7_PE4_slave  {
            set PU7_PE4_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE4_slave.txt"                 $PU7_PE4_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE5   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_5__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE5 ]
        if $PU7_PE5  {
            set PU7_PE5_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE5.txt"                 $PU7_PE5_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE5_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_5__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE5_slave ]
        if $PU7_PE5_slave  {
            set PU7_PE5_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE5_slave.txt"                 $PU7_PE5_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE6   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_6__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE6 ]
        if $PU7_PE6  {
            set PU7_PE6_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE6.txt"                 $PU7_PE6_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE6_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_6__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE6_slave ]
        if $PU7_PE6_slave  {
            set PU7_PE6_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE6_slave.txt"                 $PU7_PE6_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE7   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_7__genblk1_pe_unit(.*)} $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE7 ]
        if $PU7_PE7  {
            set PU7_PE7_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE7.txt"                 $PU7_PE7_macro $cell_name $instance_name $size_x $size_y ]
        }

        set PU7_PE7_slave   [regexp {(.*)GEN_PU_7__pu_unit/GEN_PE_7__genblk1_pe_bus_slave_inst(.*)}                 $instance_name match]
        set other_macro_flag [expr $other_macro_flag - $PU7_PE7_slave ]
        if $PU7_PE7_slave  {
            set PU7_PE7_slave_macro [ Write_Macro_File "macro_rpt/macro_PU7_PE7_slave.txt"                 $PU7_PE7_slave_macro $cell_name $instance_name $size_x $size_y ]
        }

        if $other_macro_flag {
            set read_macro_flag   [regexp {(.*)AXI_RD_BUF(.*)read_buffer(.*)}                 $instance_name match]
            if $read_macro_flag  {
                set read_macro [ Write_Macro_File "macro_rpt/read_macro.txt"                 $read_macro $cell_name $instance_name $size_x $size_y ]
            }

            set write_macro_flag   [regexp {(.*)AXI_RD_BUF(.*)write_buffer(.*)}                 $instance_name match]
            if $write_macro_flag  {
                set write_macro [ Write_Macro_File "macro_rpt/write_macro.txt"                 $write_macro $cell_name $instance_name $size_x $size_y ]
            }

            set other_macro [ Write_Macro_File "macro_rpt/other_macro.txt" $other_macro $cell_name $instance_name $size_x $size_y ]
        }
    }
}


puts "PU0_slave_macro:    $PU0_slave_macro"
puts "PU0_PE0_macro:    $PU0_PE0_macro"
puts "PU0_PE0_slave_macro:    $PU0_PE0_slave_macro"
puts "PU0_PE1_macro:    $PU0_PE1_macro"
puts "PU0_PE1_slave_macro:    $PU0_PE1_slave_macro"
puts "PU0_PE2_macro:    $PU0_PE2_macro"
puts "PU0_PE2_slave_macro:    $PU0_PE2_slave_macro"
puts "PU0_PE3_macro:    $PU0_PE3_macro"
puts "PU0_PE3_slave_macro:    $PU0_PE3_slave_macro"
puts "PU0_PE4_macro:    $PU0_PE4_macro"
puts "PU0_PE4_slave_macro:    $PU0_PE4_slave_macro"
puts "PU0_PE5_macro:    $PU0_PE5_macro"
puts "PU0_PE5_slave_macro:    $PU0_PE5_slave_macro"
puts "PU0_PE6_macro:    $PU0_PE6_macro"
puts "PU0_PE6_slave_macro:    $PU0_PE6_slave_macro"
puts "PU0_PE7_macro:    $PU0_PE7_macro"
puts "PU0_PE7_slave_macro:    $PU0_PE7_slave_macro"
puts "PU1_slave_macro:    $PU1_slave_macro"
puts "PU1_PE0_macro:    $PU1_PE0_macro"
puts "PU1_PE0_slave_macro:    $PU1_PE0_slave_macro"
puts "PU1_PE1_macro:    $PU1_PE1_macro"
puts "PU1_PE1_slave_macro:    $PU1_PE1_slave_macro"
puts "PU1_PE2_macro:    $PU1_PE2_macro"
puts "PU1_PE2_slave_macro:    $PU1_PE2_slave_macro"
puts "PU1_PE3_macro:    $PU1_PE3_macro"
puts "PU1_PE3_slave_macro:    $PU1_PE3_slave_macro"
puts "PU1_PE4_macro:    $PU1_PE4_macro"
puts "PU1_PE4_slave_macro:    $PU1_PE4_slave_macro"
puts "PU1_PE5_macro:    $PU1_PE5_macro"
puts "PU1_PE5_slave_macro:    $PU1_PE5_slave_macro"
puts "PU1_PE6_macro:    $PU1_PE6_macro"
puts "PU1_PE6_slave_macro:    $PU1_PE6_slave_macro"
puts "PU1_PE7_macro:    $PU1_PE7_macro"
puts "PU1_PE7_slave_macro:    $PU1_PE7_slave_macro"
puts "PU2_slave_macro:    $PU2_slave_macro"
puts "PU2_PE0_macro:    $PU2_PE0_macro"
puts "PU2_PE0_slave_macro:    $PU2_PE0_slave_macro"
puts "PU2_PE1_macro:    $PU2_PE1_macro"
puts "PU2_PE1_slave_macro:    $PU2_PE1_slave_macro"
puts "PU2_PE2_macro:    $PU2_PE2_macro"
puts "PU2_PE2_slave_macro:    $PU2_PE2_slave_macro"
puts "PU2_PE3_macro:    $PU2_PE3_macro"
puts "PU2_PE3_slave_macro:    $PU2_PE3_slave_macro"
puts "PU2_PE4_macro:    $PU2_PE4_macro"
puts "PU2_PE4_slave_macro:    $PU2_PE4_slave_macro"
puts "PU2_PE5_macro:    $PU2_PE5_macro"
puts "PU2_PE5_slave_macro:    $PU2_PE5_slave_macro"
puts "PU2_PE6_macro:    $PU2_PE6_macro"
puts "PU2_PE6_slave_macro:    $PU2_PE6_slave_macro"
puts "PU2_PE7_macro:    $PU2_PE7_macro"
puts "PU2_PE7_slave_macro:    $PU2_PE7_slave_macro"
puts "PU3_slave_macro:    $PU3_slave_macro"
puts "PU3_PE0_macro:    $PU3_PE0_macro"
puts "PU3_PE0_slave_macro:    $PU3_PE0_slave_macro"
puts "PU3_PE1_macro:    $PU3_PE1_macro"
puts "PU3_PE1_slave_macro:    $PU3_PE1_slave_macro"
puts "PU3_PE2_macro:    $PU3_PE2_macro"
puts "PU3_PE2_slave_macro:    $PU3_PE2_slave_macro"
puts "PU3_PE3_macro:    $PU3_PE3_macro"
puts "PU3_PE3_slave_macro:    $PU3_PE3_slave_macro"
puts "PU3_PE4_macro:    $PU3_PE4_macro"
puts "PU3_PE4_slave_macro:    $PU3_PE4_slave_macro"
puts "PU3_PE5_macro:    $PU3_PE5_macro"
puts "PU3_PE5_slave_macro:    $PU3_PE5_slave_macro"
puts "PU3_PE6_macro:    $PU3_PE6_macro"
puts "PU3_PE6_slave_macro:    $PU3_PE6_slave_macro"
puts "PU3_PE7_macro:    $PU3_PE7_macro"
puts "PU3_PE7_slave_macro:    $PU3_PE7_slave_macro"
puts "other_macro:   $other_macro"
puts "total_macro:   $total_macro"
puts "read_macro:    $read_macro"
puts "write_macro:   $write_macro"
