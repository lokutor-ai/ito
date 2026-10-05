# Host build of the on-chip engine (the same C sources as the firmware; -ffp-contract=off keeps it bit-exact with it).
#   make            ito_cli, host_test_v3, gen_selftest, opcount_v3
#   make test       host test of the chip weights against golden/ (8- and 16-bit activations, ASan/UBSan); the weights
#                   are fetched into models/ first (../tools/fetch_weights.sh: gated Hugging Face repo, or $ITO_WEIGHTS_DIR)
#   make test VOICE=male   the same for the male voice (VOICE=g also works): ito_male_esp32s3.bin against golden/g/
CC ?= cc
CFLAGS = -std=c99 -O2 -ffp-contract=off -Wall -Wno-unused-function -I../engine
ENGINE = ../engine/itofs.c ../engine/itofs.h
VOICE ?= female
ifeq ($(filter g G m M male,$(VOICE)),)
BLOB = ../../models/ito_female_esp32s3.bin
GOLDEN = golden
else
BLOB = ../../models/ito_male_esp32s3.bin
GOLDEN = golden/g
endif

all: ito_cli host_test_v3 gen_selftest opcount_v3 w4_dump

ito_cli: ito_cli.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ ito_cli.c ../engine/itofs.c -lm
host_test_v3: host_test_v3.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ host_test_v3.c ../engine/itofs.c -lm
host_test_v3_asan: host_test_v3.c $(ENGINE)
	$(CC) -std=c99 -O1 -g -fsanitize=address,undefined -ffp-contract=off -I../engine -o $@ host_test_v3.c ../engine/itofs.c -lm
gen_selftest: gen_selftest.c $(ENGINE)
	$(CC) $(CFLAGS) -o $@ gen_selftest.c ../engine/itofs.c -lm
opcount_v3: opcount_v3.c $(ENGINE)
	$(CC) $(CFLAGS) -DITOFS_OPCOUNT -o $@ opcount_v3.c ../engine/itofs.c -lm

w4_dump: w4_dump.c $(ENGINE)
	$(CC) $(CFLAGS) -D_ISOC11_SOURCE -o $@ w4_dump.c ../engine/itofs.c -lm

$(BLOB):
	VOICE=$(VOICE) ../tools/fetch_weights.sh

test: host_test_v3 host_test_v3_asan $(BLOB)
	./host_test_v3 $(BLOB) $(GOLDEN) --act 8 | tail -2
	./host_test_v3 $(BLOB) $(GOLDEN) --act 16 | tail -2
	ASAN_OPTIONS=detect_leaks=0 ./host_test_v3_asan $(BLOB) $(GOLDEN) --act 8 > asan.log 2>&1; \
	  if grep -qE "runtime error|AddressSanitizer" asan.log; then echo "ASan/UBSan: FINDINGS (asan.log)"; exit 1; \
	  else tail -1 asan.log; echo "ASan/UBSan: clean"; fi

clean:
	rm -f w4_dump ito_cli host_test_v3 host_test_v3_asan gen_selftest opcount_v3 asan.log golden/*.wav

.PHONY: all test clean
