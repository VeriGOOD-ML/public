setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {2255.944 0.0}  -end {616.488 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {616.488 3139.464}  -end {2255.944 3139.464}
setPinAssignMode -pinEditInBatch false
