# Host build and tests of the on-chip front end.
#   make            libitofe objects + test programs
#   make test       diff the C front end against the Python reference over the corpus (needs the reference venv, see FRONTEND.md)
CC ?= cc
ES = espeak
ESRC = $(wildcard $(ES)/src/*.c)
USRC = $(wildcard $(ES)/ucd/*.c)
CFLAGS ?= -O2
INC = -I. -I$(ES)/src -I$(ES)/include -I$(ES)/include/compat -I$(ES)/ucd/include
BUILD = build
OBJS = $(patsubst %.c,$(BUILD)/%.o,$(ESRC) $(USRC) ito_frontend.c)

all: $(BUILD)/fe_cli

$(BUILD)/%.o: %.c ito_frontend.h ito_symbols.inc
	@mkdir -p $(dir $@)
	$(CC) -std=gnu99 $(CFLAGS) -Wno-everything $(INC) -c $< -o $@

$(BUILD)/fe_cli: test/fe_cli.c $(OBJS)
	$(CC) -std=gnu99 $(CFLAGS) $(INC) -o $@ $^ -lm
$(BUILD)/fe_bench: test/fe_bench.c $(OBJS)
	$(CC) -std=gnu99 $(CFLAGS) $(INC) -o $@ $^ -lm

$(BUILD)/espeak_data.bin: tools/pack_data.py
	python3 tools/pack_data.py $(ESPEAK_DATA) $@

clean:
	rm -rf $(BUILD)
.PHONY: all clean
