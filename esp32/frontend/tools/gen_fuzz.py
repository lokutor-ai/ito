#!/usr/bin/env python3
"""Deterministic stress corpus: random sequences of words, letters, abbreviations, numbers and punctuation/odd whitespace.
   gen_fuzz.py N seed > fuzz.tsv"""
import random, sys
n = int(sys.argv[1]); random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
W = "the a an of in on at to is was be have she he they we you it good morning coffee ready meeting weather quickly bright river mountain seven twelve hundred thought through enough though".split()
L = list("abcdefghijklmnopqrstuvwxyzAIXYZ")
AB = ["Mr.", "Dr.", "St.", "p.m.", "a.m.", "U.S.", "e.g.", "i.e.", "etc.", "Inc.", "vs.", "No.", "Jan.", "Ph.D.", "J.", "T.S."]
NUM = ["3", "42", "3.5", "0.75", "1,250", "3,000,000", "$5", "$5.99", "$1,250.75", "12:30", "2024", "1st", "100%", "5km", "10kg", "B2B", "3D"]
PUN = [",", ".", "!", "?", ";", ":", "...", "..", "?!", "!!", "(", ")", "[", "]", "{", "}", "“", "”", "«", "»", "—", "…", "-", "--", "'", "‘", "’", '"', "&", "/", "*", "#", "@", "%", "+", "=", "<", ">"]
SP = [" ", " ", " ", " ", "  ", "\t", "\n", "  "]
def tok():
    r = random.random()
    if r < .55: return random.choice(W)
    if r < .65: return random.choice(L)
    if r < .75: return random.choice(AB)
    if r < .85: return random.choice(NUM)
    return random.choice(PUN)
for k in range(n):
    parts = []
    for _ in range(random.randint(1, 14)):
        t = tok()
        if random.random() < .3: t = t + random.choice(PUN)
        if random.random() < .12: t = random.choice(PUN) + t
        parts.append(t); parts.append(random.choice(SP) if random.random() < .9 else "")
    s = "".join(parts).strip("\n\t")
    if s.strip(): print("fuzz\t" + s.replace("\n", "   ") if False else "fuzz\t" + s.replace("\t", " ").replace("\n", " "))
