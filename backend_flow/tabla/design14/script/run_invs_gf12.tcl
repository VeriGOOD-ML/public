# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
###############################
source design_setting.tcl
source floorplan.tcl

source generate_power_cutrow.tcl
source cut_row.tcl


cutRow -site $site -area 2902.536 130.560 2903.040 218.112
cutRow -site $site -area 2902.536 218.112 2903.040 218.688
cutRow -site $site -area 2902.536 345.984 2903.040 434.112
cutRow -site $site -area 2902.536 434.112 2903.040 434.688
cutRow -site $site -area 2902.536 1478.400 2903.040 1478.976
cutRow -site $site -area 2902.536 1478.976 2903.040 1565.952
cutRow -site $site -area 2902.536 1565.952 2903.040 1566.528
cutRow -site $site -area 2902.536 1693.824 2903.040 1781.376
cutRow -site $site -area 2902.536 1781.376 2903.040 1781.952
cutRow -site $site -area 2258.592 1478.976 2259.096 1501.440
cutRow -site $site -area 2258.592 1263.552 2259.096 1352.256
cutRow -site $site -area 2258.592 1048.128 2259.096 1137.408
cutRow -site $site -area 2258.592 775.680 2259.096 864.960
cutRow -site $site -area 2258.592 560.832 2259.096 649.536
cutRow -site $site -area 2258.592 412.224 2259.096 434.112
cutRow -site $site -area 2902.536 218.688 2903.040 219.264
cutRow -site $site -area 2902.536 1566.528 2903.040 1567.104
cutRow -site $site -area 2902.536 1781.952 2903.040 1782.528

source addEndCap.tcl
source gf12_power_stripes.tcl
source clean_pdn_drc.tcl


verify_drc
fixVia -minStep

verify_drc
fixVia -minStep

verify_drc
fixVia -minStep

verify_drc
fixVia -minStep


saveDesign $encDir/post_floorplan_${design}.enc
saveNetlist $encDir/post_floorplan_${design}.v
defOut -routing  $encDir/post_floorplan_${design}.def


###########################################
setFillerMode -fitGap true
# Specifies the minimum sites gap between instances
setPlaceMode -place_detail_legalization_inst_gap 1
# Enables placer to honor and fix double pattern constaint violations between adjacent cells
setPlaceMode -place_detail_color_aware_legal true
setPlaceMode -place_global_place_io_pins false
place_opt_design -out_dir $rptDir -prefix place
refinePlace

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

group_path -name regs_placement -from [all_registers] -to [all_registers]
group_path -name ingrp_placement -from [all_inputs] -to [all_registers]
group_path -name outgrp_placement -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_placement -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_placement.mtarpt
report_timing -path_group  ingrp_placement -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_placement.mtarpt
report_timing -path_group  outgrp_placement -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_placement.mtarpt


summaryReport -noHtml -outfile ${rptDir}/placement_summary.rpt
report_power -outfile ${rptDir}/placement_power.rpt
report_timing -path_type full_clock > ${rptDir}/placement_timing.rpt
report_power -leakage -outfile ${rptDir}/placement_leakage_power.rpt
timeDesign -preCTS >  ${rptDir}/placement_timeDesign.rpt
write_sdc > ${rptDir}/placement.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/placement_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp


defOut -routing ${encDir}/placement_${design}.def
saveNetlist ${encDir}/placement_${design}.v
saveDesign ${encDir}/placement_${design}.enc


##############################################
setOptMode -unfixClkInstForOpt false
create_ccopt_clock_tree_spec -file $design.ccopt
ccopt_design

# Use actual clock network
set_interactive_constraint_modes [all_constraint_modes -active]
set_propagated_clock [all_clocks]
set_clock_propagation propagated

# Post-CTS timing optimization
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
setOptMode -usefulSkew true
optDesign -postCTS -hold


# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

group_path -name regs_cts -from [all_registers] -to [all_registers]
group_path -name ingrp_cts -from [all_inputs] -to [all_registers]
group_path -name outgrp_cts -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_cts -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_cts.mtarpt
report_timing -path_group  ingrp_cts -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_cts.mtarpt
report_timing -path_group  outgrp_cts -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_cts.mtarpt



summaryReport -noHtml -outfile ${rptDir}/cts_summary.rpt
report_power -outfile ${rptDir}/cts_power.rpt
report_timing -path_type full_clock > ${rptDir}/cts_timing.rpt
report_power -leakage -outfile ${rptDir}/cts_leakage_power.rpt
timeDesign -postCTS >  ${rptDir}/cts_timeDesign.rpt
write_sdc > ${rptDir}/cts.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/cts_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp


defOut -routing ${encDir}/cts_${design}.def
saveNetlist ${encDir}/cts_${design}.v
saveDesign ${encDir}/cts_${design}.enc



##################################################################
setNanoRouteMode -routeTopRoutingLayer 13
setNanoRouteMode -routeBottomRoutingLayer 2
setNanoRouteMode -routeWithSiDriven true
setNanoRouteMode -routeWithTimingDriven true
setNanoRouteMode -routeExpUseAutoVia true

## Fix antenna violations
setNanoRouteMode -routeInsertAntennaDiode true
setNanoRouteMode -drouteFixAntenna true

## Recommended by lib owners
# Prevent router modifying M1 pins shapes
setNanoRouteMode -routeWithViaInPin "1:1"
setNanoRouteMode -routeWithViaOnlyForStandardCellPin "1:1"

## minimizes via count during the route
setNanoRouteMode -routeConcurrentMinimizeViaCountEffort high


## allows route of tie off nets to internal cell pin shapes rather than routing to special net structure.
setNanoRouteMode -routeAllowPowerGroundPin true

## weight multi cut use high and spend more time optimizing dcut use.
setNanoRouteMode -drouteUseMultiCutViaEffort high

## limit VIAs to ongrid only for VIA1 (S1)
setNanoRouteMode -drouteOnGridOnly "via 1:1"
setNanoRouteMode -drouteAutoStop false

#SM suggestion for solving long extraction runtime during GR
setNanoRouteMode -grouteExpWithTimingDriven false


routeDesign

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

group_path -name regs_route -from [all_registers] -to [all_registers]
group_path -name ingrp_route -from [all_inputs] -to [all_registers]
group_path -name outgrp_route -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_route.mtarpt
report_timing -path_group  ingrp_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_route.mtarpt
report_timing -path_group  outgrp_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_route.mtarpt



summaryReport -noHtml -outfile ${rptDir}/route_summary.rpt
report_power -outfile ${rptDir}/route_power.rpt
report_timing  -path_type full_clock > ${rptDir}/route_timing.rpt
report_power -leakage -outfile ${rptDir}/route_leakage_power.rpt
timeDesign -postRoute >  ${rptDir}/route_timeDesign.rpt
write_sdc > ${rptDir}/route.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/route_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp



defOut -routing ${encDir}/route_${design}.def
saveNetlist ${encDir}/route_${design}.v
saveDesign ${encDir}/route_${design}.enc


###########################################################################
# fix drc
verify_drc -limit 100000000 -report verify_drc.rpt
editDeleteViolations
routeDesign

verifyProcessAntenna -report antenna.rpt -error 100000000 -maxFloatingAreaDiffNet -pgnet
verify_drc -limit 100000000 -report verify_drc.rpt

ecoRoute -fix_drc

fixVia -minStep
verify_drc -limit 100000000 -report verify_drc.rpt

fixVia -minStep
verify_drc -limit 100000000 -report verify_drc.rpt

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

group_path -name regs_fix_drc_route -from [all_registers] -to [all_registers]
group_path -name ingrp_fix_drc_route -from [all_inputs] -to [all_registers]
group_path -name outgrp_fix_drc_route -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_fix_drc_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_fix_drc_route.mtarpt
report_timing -path_group  ingrp_fix_drc_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_fix_drc_route.mtarpt
report_timing -path_group  outgrp_fix_drc_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_fix_drc_route.mtarpt




summaryReport -noHtml -outfile ${rptDir}/fix_drc_route_summary.rpt
report_power -outfile ${rptDir}/fix_drc_route_power.rpt
report_timing -path_type full_clock > ${rptDir}/fix_drc_route_timing.rpt
report_power -leakage -outfile ${rptDir}/fix_drc_route_leakage_power.rpt
timeDesign -postRoute >  ${rptDir}/fix_drc_route_timeDesign.rpt
write_sdc > ${rptDir}/fix_drc_route.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/fix_drc_route_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp



defOut -routing ${encDir}/fix_drc_route_${design}.def
saveNetlist ${encDir}/fix_drc_route_${design}.v
saveDesign ${encDir}/fix_drc_route_${design}.enc


############################################################################
## post route optimization
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both
setOptMode -powerEffort high -leakageToDynamicRatio 0.5


# design report
group_path -name regs_before_opt_route -from [all_registers] -to [all_registers]
group_path -name ingrp_before_opt_route -from [all_inputs] -to [all_registers]
group_path -name outgrp_before_opt_route -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_before_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_before_opt_route.mtarpt
report_timing -path_group  ingrp_before_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_before_opt_route.mtarpt
report_timing -path_group  outgrp_before_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_before_opt_route.mtarpt


summaryReport -noHtml -outfile ${rptDir}/before_opt_route_summary.rpt
report_power -outfile ${rptDir}/before_opt_route_power.rpt
report_timing -path_type full_clock > ${rptDir}/before_opt_route_timing.rpt
report_power -leakage -outfile ${rptDir}/before_opt_route_leakage_power.rpt
timeDesign -postRoute >  ${rptDir}/before_opt_route_timeDesign.rpt
write_sdc > ${rptDir}/before_opt_route.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/before_opt_route_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp


defOut -routing ${encDir}/before_opt_route_${design}.def
saveNetlist ${encDir}/before_opt_route_${design}.v
saveDesign ${encDir}/before_opt_route_${design}.enc




optDesign -postRoute -hold -setup




# design report
group_path -name regs_post_opt_route -from [all_registers] -to [all_registers]
group_path -name ingrp_post_opt_route -from [all_inputs] -to [all_registers]
group_path -name outgrp_post_opt_route -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_post_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_post_opt_route.mtarpt
report_timing -path_group  ingrp_post_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_post_opt_route.mtarpt
report_timing -path_group  outgrp_post_opt_route -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_post_opt_route.mtarpt



summaryReport -noHtml -outfile ${rptDir}/post_opt_route_summary.rpt
report_power -outfile ${rptDir}/post_opt_route_power.rpt
report_timing -path_type full_clock > ${rptDir}/post_opt_route_timing.rpt
report_power -leakage -outfile ${rptDir}/post_opt_route_leakage_power.rpt
timeDesign -postRoute >  ${rptDir}/post_opt_route_timeDesign.rpt
write_sdc > ${rptDir}/post_opt_route.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/post_opt_route_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp


defOut -routing ${encDir}/post_opt_route_${design}.def
saveNetlist ${encDir}/post_opt_route_${design}.v
saveDesign ${encDir}/post_opt_route_${design}.enc



# leakage recovery
setOptMode -leakageToDynamicRatio 0.5
optPower -postRoute  -effortLevel high


##################################################
# Report Design
# SPEF generation
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both


extractRC
 
 
rcOut -rc_corner Cmax -spef ${encDir}/invs_Cmax_$design\.spef
rcOut -rc_corner Cmin -spef ${encDir}/invs_Cmin_$design\.spef



# DEF generation


# design report
group_path -name regs_invs -from [all_registers] -to [all_registers]
group_path -name ingrp_invs -from [all_inputs] -to [all_registers]
group_path -name outgrp_invs -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_invs -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_invs.mtarpt
report_timing -path_group  ingrp_invs -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_invs.mtarpt
report_timing -path_group  outgrp_invs -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_invs.mtarpt



summaryReport -noHtml -outfile ${rptDir}/invs_summary.rpt
report_power -outfile ${rptDir}/invs_power.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing.rpt
report_power -leakage -outfile ${rptDir}/invs_leakage_power.rpt
report_power -view POWER_VIEW -outfile ${rptDir}/invs_ff_power.rpt
report_power -view WC_VIEW -outfile ${rptDir}/invs_ss_power.rpt
timeDesign -postRoute >  ${rptDir}/invs_timeDesign.rpt
write_sdc > ${rptDir}/invs_route.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/invs_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp



defOut -routing ${encDir}/invs_$design\.def 
saveDesign ${encDir}/${design}.enc
saveNetlist ${encDir}/invs_$design\.v

###############################################################
#### Write Updated Report
set path_collection [report_timing -collection]
 
set WNS 0
 
foreach_in_collection path $path_collection {
    set WNS [get_property $path slack]
}
 
set clock_period 0
 
set clock_periods [get_property [get_clocks] period]
 
set clock_period [lindex $clock_periods 0]
 
set effective_clock_period [expr $clock_period - $WNS]
 
write_sdc > ${rptDir}/${design}_updated.sdc
 
exec sed -i "s/period/period@/g" ${rptDir}/${design}_updated.sdc
exec sed -i "s/@.*//g" ${rptDir}/${design}_updated.sdc
exec sed -i "s/period/period  ${effective_clock_period}/g" ${rptDir}/${design}_updated.sdc
 
update_constraint_mode -ilm_sdc_files $sdc -name CON -sdc_files ${rptDir}/${design}_updated.sdc
set_propagated_clock [all_clocks]
set_clock_propagation propagated 
 
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both

# design report
group_path -name regs_invs_updated -from [all_registers] -to [all_registers]
group_path -name ingrp_invs_updated -from [all_inputs] -to [all_registers]
group_path -name outgrp_invs_updated -from [all_registers] -to [all_outputs]

report_timing -path_group  regs_invs_updated -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_invs_updated.mtarpt
report_timing -path_group  ingrp_invs_updated -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_invs_updated.mtarpt
report_timing -path_group  outgrp_invs_updated -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_invs_updated.mtarpt



summaryReport -noHtml -outfile ${rptDir}/invs_summary_updated.rpt
report_power -outfile ${rptDir}/invs_power_updated.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing_updated.rpt
report_power -leakage -outfile ${rptDir}/invs_leakage_power_updated.rpt
timeDesign -postRoute >  ${rptDir}/invs_timeDesign_updated.rpt
report_power -view POWER_VIEW -outfile ${rptDir}/invs_ff_power_updated.rpt
report_power -view WC_VIEW -outfile ${rptDir}/invs_ss_power_updated.rpt
write_sdc > ${rptDir}/invs_route_updated.sdc

set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/invs_cell_distribution_updated.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp

defOut -routing ${encDir}/invs_updated_$design\.def 
saveDesign ${encDir}/${design}_updated.enc
saveNetlist ${encDir}/invs_updated_$design\.v
saveNetlist -flat ${encDir}/invs_updated_${design}_flat.v

exit


