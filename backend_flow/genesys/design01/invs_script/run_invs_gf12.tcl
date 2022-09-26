# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
###############################
source design_setting.tcl
source floorplan.tcl

# power routing
source generate_power_cutrow.tcl
source cut_row.tcl
source addEndCap.tcl
source gf12_power_stripes.tcl
source clean_pdn_drc.tcl


###########################################
setFillerMode -fitGap true
# Specifies the minimum sites gap between instances
setPlaceMode -place_detail_legalization_inst_gap 1
# Enables placer to honor and fix double pattern constaint violations between adjacent cells
setPlaceMode -place_detail_color_aware_legal true
setPlaceMode -place_global_place_io_pins false
place_opt_design -out_dir $rptDir -prefix place
refinePlace

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

############################################################################
## post route optimization
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
optDesign -postRoute -hold -setup

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

# Report Statistics at Target Clock Period
# design report
summaryReport -noHtml -outfile ${rptDir}/invs_summary_target.rpt
report_power  -outfile ${rptDir}/invs_power_target.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing_target.rpt
report_power  -leakage -outfile ${rptDir}/invs_leakage_power_target.rpt
report_power  -report_prefix systolic_array_inst_target  \
              -hierarchical_instances systolic_array_inst \
              -cell_type {all}
report_power  -report_prefix simd_array_target \
              -hierarchical_instances simd_array \
              -cell_type {all} 
timeDesign -postRoute >  ${rptDir}/invs_timeDesign_target.rpt
write_sdc > ${rptDir}/invs_route_target.sdc

defOut -routing ${encDir}/invs_$design\.def 
saveDesign ${encDir}/${design}.enc
saveNetlist ${encDir}/invs_$design\.v

###############################################################
#### Write Updated Report
# update sdc to effective clock period
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

# Report Statistics at Effective Clock Period
# design report
summaryReport -noHtml -outfile ${rptDir}/invs_summary_eff.rpt
report_power  -outfile ${rptDir}/invs_power_eff.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing_eff.rpt
report_power  -leakage -outfile ${rptDir}/invs_leakage_power_eff.rpt
report_power  -report_prefix systolic_array_inst_eff  \
              -hierarchical_instances systolic_array_inst \
              -cell_type {all}
report_power  -report_prefix simd_array_eff \
              -hierarchical_instances simd_array \
              -cell_type {all} 
timeDesign -postRoute >  ${rptDir}/invs_timeDesign_eff.rpt
write_sdc > ${rptDir}/invs_route_eff.sdc


defOut -routing ${encDir}/invs_updated_$design\.def 
saveDesign ${encDir}/${design}_updated.enc
saveNetlist ${encDir}/invs_updated_$design\.v
saveNetlist -flat ${encDir}/invs_updated_${design}_flat.v

exit
