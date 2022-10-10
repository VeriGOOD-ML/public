setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {1346.0 0.0}  -end {310.0 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {310.0 1572.0}  -end {1346.0 1572.0}
setPinAssignMode -pinEditInBatch false
