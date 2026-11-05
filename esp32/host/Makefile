# Host build of the on-chip engine (the same C sources as the firmware; -ffp-contract=off keeps it bit-exact with it).
#   make            ito_cli, host_test_v3, gen_selftest, opcount_v3
#   make test       host test of the chip weights against golden/ (8- and 16-bit activations, ASan/UBSan); the weights
#                   are fetched into models/ first (../tools/fetch_weights.sh: gated Hugging Face repo, or $ITO_WEIGHTS_DIR)
#   make test VOICE=male   the same for the male voice (VOICE=g also works): ito_male_esp32s3.bin against golden/g/
#   make test_split        the verdict state machine of the runtime split choice (engine/itofs_split.c) with simulated resets and test results
#   make test_frontend   the on-chip text front end (esp32/frontend: espeak-ng 1.52.0 en-us port + phonemizer / nltk logic) against golden ids made by the Python front end; needs no weights
#   make test_sched        the boot calibration / start-delay policy (engine/itofs_sched.c) against a model of the speaker, with simulated timings (no weights needed)
CC ?= cc
CFLAGS = -std=c99 -D_DEFAULT_SOURCE -O2 -ffp-contract=off -Wall -Wno-unused-function -I../engine
ENGINE = ../engine/itofs.c ../engine/itofs.h
VOICE ?= female
ifeq ($(filter g G m M male,$(VOICE)),)
BLOB = ../../models/ito_female_esp32s3.bin
GOLDEN = golden
else
BLOB = ../../models/ito_male_esp32s3.bin
GOLDEN = golden/g
endif

all: ito_cli ito_batch host_test_v3 gen_selftest opcount_v3 w4_dump host_test_sched host_test_split host_test_frontend

ito_cli: ito_cli.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ ito_cli.c ../engine/itofs.c -lm
ito_batch: ito_batch.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ ito_batch.c ../engine/itofs.c -lm
host_test_v3: host_test_v3.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ host_test_v3.c ../engine/itofs.c -lm
host_test_v3_asan: host_test_v3.c $(ENGINE)
	$(CC) -std=c99 -D_DEFAULT_SOURCE -O1 -g -fsanitize=address,undefined -ffp-contract=off -I../engine -o $@ host_test_v3.c ../engine/itofs.c -lm
gen_selftest: gen_selftest.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ gen_selftest.c ../engine/itofs.c -lm
opcount_v3: opcount_v3.c $(ENGINE)
	$(CC) $(CFLAGS) -DITOFS_OPCOUNT -o $@ opcount_v3.c ../engine/itofs.c -lm

w4_dump: w4_dump.c $(ENGINE)
	$(CC) $(CFLAGS) -D_ISOC11_SOURCE -o $@ w4_dump.c ../engine/itofs.c -lm

host_test_sched: host_test_sched.c ../engine/itofs_sched.c ../engine/itofs_sched.h ../engine/itofs_equiv.c ../engine/itofs_equiv.h
	$(CC) $(CFLAGS) -o $@ host_test_sched.c ../engine/itofs_sched.c ../engine/itofs_equiv.c -lm
test_sched: host_test_sched
	./host_test_sched

host_test_split: host_test_split.c ../engine/itofs_split.c ../engine/itofs_split.h
	$(CC) $(CFLAGS) -o $@ host_test_split.c ../engine/itofs_split.c
test_split: host_test_split
	./host_test_split


# ---- on-chip text front end (esp32/frontend) ----
FE = ../frontend
FE_SRC = $(FE)/ito_frontend.c $(wildcard $(FE)/espeak/src/*.c) $(wildcard $(FE)/espeak/ucd/*.c)
FE_INC = -I$(FE) -I$(FE)/espeak/src -I$(FE)/espeak/include -I$(FE)/espeak/include/compat -I$(FE)/espeak/ucd/include
host_test_frontend: $(FE)/test/host_test_frontend.c $(FE_SRC) $(FE)/ito_frontend.h $(FE)/ito_symbols.inc
	$(CC) -std=gnu99 -O2 -w $(FE_INC) -o $@ $(FE)/test/host_test_frontend.c $(FE_SRC) -lm
test_frontend: host_test_frontend
	./host_test_frontend $(FE)/data/espeak_en_us.bin $(FE)/test/golden_single.tsv $(FE)/test/golden_stream.tsv

$(BLOB):
	VOICE=$(VOICE) ../tools/fetch_weights.sh

test: test_frontend host_test_v3 host_test_v3_asan $(BLOB)
	./host_test_v3 $(BLOB) $(GOLDEN) --act 8 | tail -2
	./host_test_v3 $(BLOB) $(GOLDEN) --act 16 | tail -2
	ASAN_OPTIONS=detect_leaks=0 ./host_test_v3_asan $(BLOB) $(GOLDEN) --act 8 > asan.log 2>&1; \
	  if grep -qE "runtime error|AddressSanitizer" asan.log; then echo "ASan/UBSan: FINDINGS (asan.log)"; exit 1; \
	  else tail -1 asan.log; echo "ASan/UBSan: clean"; fi

clean:
	rm -f ito_batch host_test_frontend host_test_split host_test_sched w4_dump ito_cli host_test_v3 host_test_v3_asan gen_selftest opcount_v3 asan.log golden/*.wav

.PHONY: all test test_sched test_split test_frontend clean
