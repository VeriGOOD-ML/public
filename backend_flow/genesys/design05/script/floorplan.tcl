#########################
# floorplan
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
setGenerateViaMode -auto true
generateVias
createBasicPathGroups -expanded

floorPlan -site $site -s 2964.624 3134.464 5.0 5.0 5.0 5.0
source place_macro.tcl
source place_pin.tcl

