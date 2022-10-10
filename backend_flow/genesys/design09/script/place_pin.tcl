setPinAssignMode -pinEditInBatch true
set input_terms [dbGet [dbGet -p1 top.terms.direction input].name]
set output_terms [dbGet [dbGet -p1 top.terms.direction output].name]
editPin -pin $input_terms -spreadType RANGE -layer K2  -start {2867.432 0.0}  -end {1227.976 0.0}
editPin -pin $output_terms -spreadType RANGE -layer K2  -start {1227.976 6273.928}  -end {2867.432 6273.928}
setPinAssignMode -pinEditInBatch false
