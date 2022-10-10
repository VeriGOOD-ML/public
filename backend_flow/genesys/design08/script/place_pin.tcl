setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {1255.944 0.0}  -end {5.0 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {5.0 6855.288}  -end {1255.944 6855.288}
setPinAssignMode -pinEditInBatch false
