`define CONFIGURATION_SELECT(select,index) (\
(select == 0 && index == 0) ? 8 : /*NUM_PU*/\
(select == 0 && index == 1) ? 8 : /*NUM_PE*/\
(select == 0 && index == 2) ? 16 : /*INPUT_BITWIDTH*/\
(select == 0 && index == 3) ? 16 : /*INTERNAL_BITWIDTH*/\
(select == 0 && index == 4) ? 0 : /*BENCHMARK*/\
(select == 88 && index == 0) ? 8 : /*NUM_PU*/\
(select == 88 && index == 1) ? 8 : /*NUM_PE*/\
(select == 88 && index == 2) ? 16 : /*INPUT_BITWIDTH*/\
(select == 88 && index == 3) ? 16 : /*INTERNAL_BITWIDTH*/\
(select == 88 && index == 4) ? 88 : /*BENCHMARK*/\
(index == 0) ? \
( (select-1)/8 == 0 ? 8 : 4) : \
(index == 1) ? \
( ((select-1) %8)/4 == 0 ? 8 : 16) : \
(index == 2) ? \
( ((select-1) %4)/2 == 0 ? 16 : 8) : \
(index == 3) ? \
( ((select-1) %4)/2 == 0 ? 32 : 24) : \
(index == 4) ? \
( ((select-1) %2) == 0 ? 2 : 3) : \
0)

`define MEMORY_SIZE(benchmark,index) (\
(benchmark == 0 && index == 0) ? 256 :  	/*data ----------------		*/\
(benchmark == 0 && index == 1) ? 256 :  	/*weight                    */\
(benchmark == 0 && index == 2) ? 256 :  	/*interim                   */\
(benchmark == 0 && index == 3) ? 8   :  	/*index in instruction      */\
(benchmark == 0 && index == 4) ? 14  :  	/*instruction address length*/\
(benchmark == 0 && index == 5) ? 512 :  	/* bus read depth           */\
(benchmark == 0 && index == 6) ? 512 :  	/* bus write depth          */\
(benchmark == 0 && index == 7) ? 256 :  	/* neigh fifo depth         */\
(benchmark == 2 && index == 0) ? 64 :  	/*data ----------------     */\
(benchmark == 2 && index == 1) ? 160 :  	/*weight                    */\
(benchmark == 2 && index == 2) ? 96:  	/*interim                   */\
(benchmark == 2 && index == 3) ? 8  :  	/*index in instruction      */\
(benchmark == 2 && index == 4) ? 12  :  	/*instruction address length*/\
(benchmark == 2 && index == 5) ? 64 :  	/* bus read depth           */\
(benchmark == 2 && index == 6) ? 128 :  	/* bus write depth          */\
(benchmark == 2 && index == 7) ? 256 :  	/* neigh fifo depth         */\
(benchmark == 3 && index == 0) ? 16 :  	/*data ----------------     */\
(benchmark == 3 && index == 1) ? 352 :  	/*weight                    */\
(benchmark == 3 && index == 2) ? 96:  	/*interim                   */\
(benchmark == 3 && index == 3) ? 9  :  	/*index in instruction      */\
(benchmark == 3 && index == 4) ? 13  :  	/*instruction address length*/\
(benchmark == 3 && index == 5) ? 128 :  	/* bus read depth           */\
(benchmark == 3 && index == 6) ? 256 :  	/* bus write depth          */\
(benchmark == 3 && index == 7) ? 256 :  	/* neigh fifo depth         */\
(benchmark == 88 && index == 0) ? 256 :  	/*data ----------------     */\
(benchmark == 88 && index == 1) ? 256 :  	/*weight                    */\
(benchmark == 88 && index == 2) ? 128:  	/*interim                   */\
(benchmark == 88 && index == 3) ? 8  :  	/*index in instruction      */\
(benchmark == 88 && index == 4) ? 14  :  	/*instruction address length*/\
(benchmark == 88 && index == 5) ? 256 :  	/* bus read depth           */\
(benchmark == 88 && index == 6) ? 256 :  	/* bus write depth          */\
(benchmark == 88 && index == 7) ? 1024 :  	/* neigh fifo depth         */\
0)