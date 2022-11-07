# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
#########################
# floorplan
clearGlobalNets
globalNetConnect VDD -type pgpin -pin VDD -all  -override
globalNetConnect VSS -type pgpin -pin VSS -all  -override
globalNetConnect VDD -type tiehi -all  -override
globalNetConnect VSS -type tielo -all  -override


setOptMode -powerEffort high -leakageToDynamicRatio 0.5

setGenerateViaMode -auto true
generateVias

createBasicPathGroups -expanded

floorPlan -site $site -s 1560.0 2134.0  5.0 5.0 5.0 5.0

source place_macro.tcl
source place_pin.tcl

defOut -routing ${encDir}/${design}_floorplan_auto.def
saveNetlist ${encDir}/${design}_floorplan_auto.v
saveDesign ${encDir}/${design}_floorplan_auto.enc
