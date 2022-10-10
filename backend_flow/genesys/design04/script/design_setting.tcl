setMultiCpuUsage -localCpu 16 
set design genesys_top_module
set libdir "/oasis/tscc/scratch/zhw033/RTML/gf12lp"
set sramdir "/oasis/tscc/scratch/zhw033/RTML/RF_GF12_Macros"

set lef "
    ${libdir}/lef/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_84cpp_tech.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_lvt_c14.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_rvt_c14.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_slvt_c14.lef \
    ${sramdir}/lef/DPRFW256B8.lef \
    ${sramdir}/lef/DPRFW4096B8.lef \
    ${sramdir}/lef/DPRFW32B32.lef \
    ${sramdir}/lef/DPRFW64B32.lef \
    ${sramdir}/lef/DPRFW256B32.lef \
    ${sramdir}/lef/DPRFW2048B32.lef \
    "

set libworst "
    ${libdir}/lib/sc9mcpp84_12lp_base_lvt_c14_tt_nominal_max_0p80v_25c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_rvt_c14_tt_nominal_max_0p80v_25c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_slvt_c14_tt_nominal_max_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW256B8_tt_nominal_0p80v_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW4096B8_tt_nominal_0p80v_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW32B32_tt_nominal_0p80v_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW64B32_tt_nominal_0p80v_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW256B32_tt_nominal_0p80v_0p80v_25c.lib \
    ${sramdir}/lib/DPRFW2048B32_tt_nominal_0p80v_0p80v_25c.lib \
    "

set libbest $libworst
#set qrc_max "/home/zf4_techdata/libraries/GF_12nm/pdk/12LP/V1.0_2.1/PEX/QRC/tcad/13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB/SigCmax/qrcTechFile"
set qrc_max "${libdir}/SigCmax/qrcTechFile"
set qrc_min $qrc_max 

#set layer_map "/home/zf4_techdata/libraries/GF_12nm/pdk/12LP/V1.0_2.1/PlaceRoute/Innovus/Techfiles/13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_TQRC.map"
set site "sc9mcpp84_12lp" 
 
set rptDir rpt
set encDir enc
 
if {![file exists $rptDir]} {
    exec mkdir $rptDir
}

if {![file exists $encDir]} {
    exec mkdir $encDir
}

# Since the inconsistency of time units of standard cells and macros, we need to set up the timing unit
setLibraryUnit -time 1ps  -cap 0.001pf
set_default_switching_activity -input_activity 0.1 -seq_activity 0.1

# default settings
set init_pwr_net { VDD }
set init_gnd_net { VSS }
set init_verilog "$netlist"
set init_design_netlisttype "Verilog"
set init_design_settop 1
set init_top_cell "$design"
set init_lef_file "$lef"

# MCMM setup
create_library_set -name WC_LIB -timing $libworst
create_library_set -name BC_LIB -timing $libbest
create_rc_corner -name Cmax -qx_tech_file $qrc_max -T 25
create_rc_corner -name Cmin -qx_tech_file $qrc_min -T 25
create_delay_corner -name WC -library_set WC_LIB -rc_corner Cmax
create_delay_corner -name BC -library_set BC_LIB -rc_corner Cmin
create_constraint_mode -name CON -sdc_file $sdc
create_analysis_view -name WC_VIEW -delay_corner WC -constraint_mode CON
create_analysis_view -name BC_VIEW -delay_corner BC -constraint_mode CON
init_design -setup {WC_VIEW} -hold {BC_VIEW}
set_analysis_view -leakage WC_VIEW -dynamic WC_VIEW -setup WC_VIEW -hold BC_VIEW
set_interactive_constraint_modes {CON}

# set process node
setDesignMode -process 12 -powerEffort high
