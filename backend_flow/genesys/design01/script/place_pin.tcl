# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {1303.77 0.0}  -end {267.74 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {267.74 1572.23}  -end {1303.77 1572.23}
setPinAssignMode -pinEditInBatch false
