#!/usr/bin/env python3
"""Multi-sentence test texts: random runs of corpus sentences (1..9), plus run-ons that exceed 400 tokens.  gen_multi.py corpus.tsv N seed"""
import random, sys
lines = [l.rstrip("\n").split("\t", 1)[1] for l in open(sys.argv[1], encoding="utf-8") if "\t" in l]
random.seed(int(sys.argv[3]))
for _ in range(int(sys.argv[2])):
    k = random.choice([1, 2, 2, 3, 3, 4, 5, 6, 9])
    t = " ".join(random.choice(lines) for _ in range(k))
    print("multi\t" + t)
longs = [l for l in lines if len(l) > 200]
for _ in range(12):
    print("runon\t" + ", and ".join(l.rstrip(".!?;:") for l in random.sample(longs, 5)) + ".")
