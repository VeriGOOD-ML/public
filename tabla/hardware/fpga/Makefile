FILELIST := file.list
TESTLIST := tb.list
SRC := $(shell cat $(FILELIST) | egrep -v '\#')

BINDIR := bin
WAVEFORM_DIR := $(BINDIR)/waveform

COMPILER := iverilog
COMPILER_FLAGS := -Wall
COMPILER_OUTPUT := $(BINDIR)/compiler.out

SIMULATOR := vvp
SIMULATOR_FLAGS := -vcd
WAVEFORM := $(WAVEFORM_DIR)/TB.vcd

INCLUDE_DIR := include

WAVEFORM_VIEWER := gtkwave

TEST_CASES = $(shell cat $(TESTLIST) | egrep -v '\#')

TB_COMPILE := $(TEST_CASES:%=$(BINDIR)/%.out)

###############################################################################
all: $(COMPILER_OUTPUT)

compile: $(SRC) $(FILELIST)
	@echo "Compiling.."
	@mkdir -p $(BINDIR)
	@$(COMPILER) $(COMPILER_FLAGS) -c $(FILELIST) -o $(COMPILER_OUTPUT) -I $(INCLUDE_DIR)

simulate: $(COMPILER_OUTPUT) $(SRC) $(FILELIST)
	@echo "Simulating.."
	@mkdir -p $(WAVEFORM_DIR)
	@mkdir -p $(WAVEFORM_DIR)/top
	@$(SIMULATOR) $(COMPILER_OUTPUT) $(SIMULATOR_FLAGS)

display: $(WAVEFORM)
	@$(WAVEFORM_VIEWER) $(WAVEFORM) &

$(WAVEFORM): $(COMPILER_OUTPUT) $(SRC) $(FILELIST)
	@mkdir -p $(WAVEFORM_DIR)
	@mkdir -p $(WAVEFORM_DIR)/top
	@$(SIMULATOR) $(COMPILER_OUTPUT) $(SIMULATOR_FLAGS)

$(COMPILER_OUTPUT): $(SRC) $(FILELIST)
	@echo "Compiling.."
	@mkdir -p $(BINDIR)
	@$(COMPILER) $(COMPILER_FLAGS) -c $(FILELIST) -o $(COMPILER_OUTPUT) -I $(INCLUDE_DIR)
	@echo "Generated $@"

test: $(TEST_CASES)

$(TEST_CASES:%=$(BINDIR)/%.out): $(SRC) $(FILELIST)
	@echo "Compiling Testbench - $@"
	@mkdir -p $(BINDIR)
	@$(COMPILER) $(COMPILER_FLAGS) -c $(FILELIST) -o $@ -I $(INCLUDE_DIR) -s $(@:$(BINDIR)/%.out=%)_tb

$(TEST_CASES): %:$(BINDIR)/%.out
	@echo "Simulating Testbench $<"
	@mkdir -p $(WAVEFORM_DIR)
	@$(SIMULATOR) $< $(SIMULATOR_FLAGS)

clean: 
	@\rm -rf $(WAVEFORM) $(COMPILER_OUTPUT) $(TEST_CASES:%=$(BINDIR)/%.out)
