#########################
# floorplan
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
setGenerateViaMode -auto true
generateVias
createBasicPathGroups -expanded

floorPlan -site $site -s  2390.40 6850.288 5.0 5.0 5.0 5.0
checkPlace

source place_macro.tcl
source place_pin.tcl
checkPlace

