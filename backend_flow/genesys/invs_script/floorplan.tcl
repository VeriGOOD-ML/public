#########################
# floorplan
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
setGenerateViaMode -auto true
generateVias
createBasicPathGroups -expanded

floorPlan -site $site -s 1298.768 1567.232 5.0 5.0 5.0 5.0
source place_macro.tcl
source place_pin.tcl
