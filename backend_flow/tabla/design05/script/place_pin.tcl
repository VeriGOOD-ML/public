# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
set terms [dbGet top.terms.name]
setPinAssignMode -pinEditInBatch true
editPin -pin $terms -spreadType RANGE -layer K2  -start {1769.0 2441.0}  -end {1769.0  773.0}
setPinAssignMode -pinEditInBatch false
