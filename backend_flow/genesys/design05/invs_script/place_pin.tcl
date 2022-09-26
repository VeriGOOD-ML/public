# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {1569.624 0.0}  -end {530.472 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {530.472 3139.464}  -end {1569.624 3139.464}
setPinAssignMode -pinEditInBatch false

