#!/usr/bin/env python3
"""Hostile-input corpus: ordinary words mixed with accented Latin, Greek, Cyrillic, CJK, Arabic/Devanagari digits, emoji, combining marks,
zero-width and no-break spaces, control characters, repeated punctuation.   gen_unicode_fuzz.py N seed > out.tsv"""
import random, sys
n = int(sys.argv[1]); random.seed(int(sys.argv[2]))
W = "the a of in on at to is was good morning coffee ready meeting weather quickly bright river seven hundred thought".split()
U = ["café", "naïve", "München", "São", "über", "Δ", "π", "Ω", "привет", "да", "日本", "東京", "你好",
     "مرحبا", "१२३", "١٢٣", "\U0001F642", "\U0001F44B", "é", "ä", "​", " ", " ", "﻿", "­", "–", "‘", "’", "‚", "„", "‹", "›",
     "°", "§", "¶", "€", "£", "¥", "©", "®", "™", "½", "¾", "²", "→", "×", "÷", "∞", "ﬁ", "ß", "æ", "œ", "ı", "ł", "ẞ"]
P = [",", ".", "!", "?", ";", ":", "...", "(", ")", "-", "—", "…", "“", "”", "«", "»"]
for _ in range(n):
    parts = []
    for _ in range(random.randint(2, 12)):
        r = random.random()
        parts.append(random.choice(W) if r < .45 else random.choice(U) if r < .85 else random.choice(P))
        if random.random() < .15: parts.append(random.choice(P))
    s = random.choice([" ", " ", " ", "  ", " "]).join(parts)
    s = s.replace("\x00", "")
    if s.strip(): print("unicode\t" + s.replace("\t", " ").replace("\n", " "))
