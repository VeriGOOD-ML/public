setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {710.744 0.0}  -end {5.0 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {5.0 2564.232}  -end {710.744 2564.232}
setPinAssignMode -pinEditInBatch false
