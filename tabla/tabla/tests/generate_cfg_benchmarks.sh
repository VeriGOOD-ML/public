LOC=30
FEAT=20
PU=1
PE=128
python3 test_benchmarks.py -b svm_wifi -fs $LOC $FEAT -cfg $PU $PE
mv ../compilation_output/svm_wifi_${LOC}_${FEAT} ../compilation_output/svm_wifi_${LOC}_${FEAT}_${PU}PU_${PE}PE
##
#PU=2
#PE=64
#python3 test_benchmarks.py -b svm_wifi -fs $LOC $FEAT -cfg $PU $PE
#mv ../compilation_output/svm_wifi_${LOC}_${FEAT} ../compilation_output/svm_wifi_${LOC}_${FEAT}_${PU}PU_${PE}PE
##
#PU=4
#PE=64
#python3 test_benchmarks.py -b svm_wifi -fs $LOC $FEAT -cfg $PU $PE
#mv ../compilation_output/svm_wifi_${LOC}_${FEAT} ../compilation_output/svm_wifi_${LOC}_${FEAT}_${PU}PU_${PE}PE
##
#PU=4
#PE=128
#python3 test_benchmarks.py -b svm_wifi -fs $LOC $FEAT -cfg $PU $PE
#mv ../compilation_output/svm_wifi_${LOC}_${FEAT} ../compilation_output/svm_wifi_${LOC}_${FEAT}_${PU}PU_${PE}PE
