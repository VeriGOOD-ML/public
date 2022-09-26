clearGlobalNets
globalNetConnect VDD -type pgpin -pin VDD -inst *   -override
globalNetConnect VSS -type pgpin -pin VSS -inst *   -override
globalNetConnect VDD -type pgpin -pin VDDCE -inst *  -override
globalNetConnect VDD -type pgpin -pin VDDPE -inst *  -override
globalNetConnect VSS -type pgpin -pin VSSE -inst *   -override

globalNetConnect VDD -type pgpin -pin VNW -inst *   -override
globalNetConnect VSS -type pgpin -pin VPW -inst *   -override

globalNetConnect VDD -type tiehi -all  -override
globalNetConnect VSS -type tielo -all  -override
 
setGenerateViaMode -auto true
generateVias

createBasicPathGroups -expanded

#createPGPin VDD
#createPGPin VSS

editDelete -type Special -net { VDD VSS }

#source macro_power_routing.tcl
#source power_routing.tcl

setAddStripeMode -orthogonal_only false -ignore_DRC false
setViaGenMode -ignore_DRC false
setViaGenMode -optimize_cross_via true
setViaGenMode -allow_wire_shape_change false
setViaGenMode -extend_out_wire_end false
setViaGenMode -viarule_preference generated

sroute


source macro_power_routing.tcl
source power_routing.tcl


# Romove possible drcs
verify_drc
fixVia -minStep

verify_drc
fixVia -minStep

