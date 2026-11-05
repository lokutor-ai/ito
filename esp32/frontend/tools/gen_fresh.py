#!/usr/bin/env python3
"""Fresh test texts for the on-chip front end, invented for the on-silicon run (not part of corpus/corpus.tsv, which the port was tuned on).
Seeded template + word-pool generator; every output line is  category<TAB>text.
    gen_fresh.py fresh N SEED     sentences: names, numbers, dates, times, money, URLs, e-mail, typos, questions, mixed punctuation, long and short
    gen_fresh.py bulk N SEED      the standalone-run set: 1..150 words each (long ones are runs of sentences)
    gen_fresh.py para N SEED      multi-sentence paragraphs (3..9 sentences, <= 150 words)"""
import random, sys

NAMES = "Maria Gonzalez|Tobias Wren|Priya Raman|Kenji Watanabe|Olumide Adeyemi|Sofia Lindqvist|Dr. Amelia Hartley|Mr. Peterson|Mrs. Okafor|Professor Ivanov|Captain Reyes|Lena Kowalski|Ahmed Al-Farsi|Chloe Dubois|Jamal Whitfield|Ingrid Solberg|Rosa Delgado|Nikolai Petrov|Hannah O'Connor|Wei Zhang|Ms. Abernathy|Giuseppe Moretti|Fatima Zahra|Liam Murphy".split("|")
PLACES = "Lisbon|Nairobi|Reykjavik|Osaka|Buenos Aires|Montreal|the Atacama Desert|Mount Kilimanjaro|Lake Geneva|Cape Town|St. Louis|the Mekong Delta|Edinburgh|Marrakech|Anchorage|Vancouver|Santiago|Kyoto|Tbilisi|Zanzibar|Rotterdam|the Outer Hebrides".split("|")
NOUNS = "lighthouse|spreadsheet|orchestra|thermostat|bicycle|telescope|invoice|harvest|committee|algorithm|kitchen|passport|glacier|violin|battery|library|bakery|schedule|engine|garden|firmware|manuscript|ferry|laboratory|fountain|prototype|neighborhood|keyboard|airport|cathedral|satellite|workshop|receipt|compass|umbrella|vineyard|tournament|invention|wardrobe|volcano".split("|")
ADJ = "quiet|enormous|fragile|ancient|brilliant|unexpected|crowded|bitter|gentle|stubborn|temporary|remarkable|rusty|cheerful|reluctant|sudden|peculiar|ordinary|vivid|exhausted|precise|careless|generous|hollow".split("|")
VERBS = "repaired|measured|announced|forgot|discovered|painted|postponed|translated|borrowed|inspected|celebrated|rewrote|abandoned|delivered|photographed|questioned|cancelled|assembled|recorded|noticed".split("|")
ADV = "carefully|quickly|reluctantly|twice|yesterday|without warning|before sunrise|at the last minute|on purpose|every morning|against all advice|for the third time".split("|")
MONTHS = "January February March April May June July August September October November December".split()
ACR = "NASA|USB|HTTP|GPS|PDF|CEO|FAQ|DNA|HVAC|NATO|SQL|IKEA|LED|WiFi|UNESCO|ETA|RSVP|ASAP|PhD|API".split("|")
UNITS = ["km", "kg", "mph", "GB", "MHz", "ft", "lbs", "mL", "cm", "%", "°C", "°F"]
DOMAINS = "example.com|openai-clone.org|news.bbc.co.uk|github.com/foo/bar|my-site.io/about|www.wikipedia.org|docs.python.org/3/library|shop.store.net/cart?id=42".split("|")
TYPOS = "teh recieve definately seperate occured wich untill adress becuase tommorow wierd thier beleive goverment accomodate neccessary embarass".split()
rnd = random.Random()

def pick(l): return rnd.choice(l)
def ordinal(n): return "%d%s" % (n, "th" if 10 < n < 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
def tm(): return "%d:%02d %s" % (rnd.randrange(1, 13), pick([0, 5, 15, 30, 45, 59]), pick(["AM", "PM", "a.m.", "p.m."]))
def money(): return "$%d.%02d" % (rnd.randrange(1, 900), rnd.randrange(0, 100)) if rnd.random() < .6 else "$%s" % "{:,}".format(rnd.randrange(1000, 9999999))
def pct(): return "%d%%" % rnd.randrange(1, 100) if rnd.random() < .7 else "%.1f%%" % (rnd.random() * 100)
def date():
    f = rnd.randrange(4)
    if f == 0: return "%s %d, %d" % (pick(MONTHS), rnd.randrange(1, 29), rnd.randrange(1990, 2031))
    if f == 1: return "%d/%d/%d" % (rnd.randrange(1, 13), rnd.randrange(1, 29), rnd.randrange(1990, 2031))
    if f == 2: return "the %s of %s" % (ordinal(rnd.randrange(4, 30)), pick(MONTHS))
    return "%s %d" % (pick(MONTHS), rnd.randrange(1, 29))
def an(w): return ("an " if w[0] in "aeiou" else "a ") + w
def name(): return pick(NAMES)

def simple():
    t = rnd.randrange(30)
    A, B = pick(ADJ), pick(NOUNS)
    if t == 0: return "%s %s the %s %s %s in %s." % (name(), pick(VERBS), A, B, pick(ADV), pick(PLACES))
    if t == 1: return "The %s %s was %s on %s." % (A, B, pick(VERBS), date())
    if t == 2: return "Please call %s at %d-%d-%04d before %s." % (name(), rnd.randrange(200, 999), rnd.randrange(200, 999), rnd.randrange(0, 9999), tm())
    if t == 3: return "It costs %s, which is %s more than last %s." % (money(), pct(), pick(["year", "month", "week"]))
    if t == 4: return "Visit %s for details, or write to %s@%s." % (pick(DOMAINS), pick(["info", "support", "maria.g", "hello", "j_smith"]), pick(["example.com", "mail.org", "lokutor.com", "uni.edu"]))
    if t == 5: return "Did %s really visit the %s %s?" % (name(), B, pick(["on Tuesday", "last night", "in " + pick(PLACES), "without telling anyone"]))
    if t == 6: return "Why is the %s %s so %s?" % (A, B, pick(ADJ))
    if t == 7: return "%s! That %s %s is %s." % (pick(["Wow", "Stop", "Careful", "Listen", "Hurry", "Look out"]), A, B, pick(ADJ))
    if t == 8: return "The %s of %s was %s, %s, and %s." % (B, pick(PLACES), A, pick(ADJ), pick(ADJ))
    if t == 9: return "We met %s in %s in %d, and she still owes me %s." % (name(), pick(PLACES), rnd.randrange(1985, 2025), money())
    if t == 10: return "The %s %s the %s, but %s %s %s." % (A, B, pick(["ignored", "followed", "surprised", "outran"]), name(), pick(VERBS), pick(ADV))
    if t == 11: return "%s %s %s the %s." % (pick(["Mr.", "Mrs.", "Dr.", "Prof."]), pick(["Smith", "Nakamura", "Oyelaran", "Brandt", "Lopez"]), pick(VERBS), B)
    if t == 12: return "The meeting is on %s at %s in room %d%s." % (date(), tm(), rnd.randrange(1, 40), pick(["A", "B", "C", ""]))
    if t == 13: return "The %s weighs %d %s and measures %d x %d cm." % (B, rnd.randrange(2, 900), pick(["kg", "lbs", "grams"]), rnd.randrange(2, 90), rnd.randrange(2, 90))
    if t == 14: return "%s said, \"The %s %s is not what it seems.\"" % (name(), A, B)
    if t == 15: return "I can't believe they didn't %s the %s; it's %s, isn't it?" % (pick(["fix", "ship", "paint", "check", "open"]), B, A)
    if t == 16: return "Our %s (%s) will arrive between %s and %s." % (B, pick(ACR), date(), date())
    if t == 17: return "The %s reported that %s of %s %s were %s." % (pick(ACR), pct(), "{:,}".format(rnd.randrange(20, 90000)), pick(["people", "users", "voters", "students", "households"]), pick(["satisfied", "undecided", "unreachable", "late"]))
    if t == 18: return "Teh %s was %s, wich is why we %s untill tommorow." % (B, pick(TYPOS), pick(["waited", "stayed", "argued"]))
    if t == 19: return "We definately recieve %s %s %s every %s." % (pick(["a", "the", "some"]), A, B, pick(["week", "morning", "quarter"]))
    if t == 20: return "Is it %s or %s? I need to know by %s." % (A, pick(ADJ), date())
    if t == 21: return "In %d, %s moved from %s to %s." % (rnd.randrange(1900, 2024), name(), pick(PLACES), pick(PLACES))
    if t == 22: return "Chapter %d: The %s %s" % (rnd.randrange(1, 40), A.capitalize(), B.capitalize())
    if t == 23: return "Version %d.%d.%d fixes %d bugs in the %s module." % (rnd.randrange(0, 12), rnd.randrange(0, 30), rnd.randrange(0, 99), rnd.randrange(2, 400), B)
    if t == 24: return "She paid %s for %s %s %s, plus %s tax." % (money(), pick(["two", "three", "a dozen", "forty", "1,200"]), A, B + "s", pct())
    if t == 25: return "%s, and I mean this, is the most %s %s in %s." % (name(), A, B, pick(PLACES))
    if t == 26: return "Hello, %s. Your order #%d has shipped via %s." % (name().split()[0], rnd.randrange(1000, 999999), pick(["UPS", "FedEx", "DHL", "USPS"]))
    if t == 27: return "Wait... what? You %s the whole %s?!" % (pick(["sold", "lost", "painted", "ate"]), B)
    if t == 28: return "The temperature fell to minus %d °C, so the %s %s was closed until %s." % (rnd.randrange(2, 40), A, B, pick(MONTHS))
    return "%s %s %s the %s %s." % (name(), pick(ADV), pick(VERBS), A, B)

def short():
    return pick(["Yes.", "No way.", "Okay, fine.", "Thank you!", "Good morning.", "Hello there.", "Really?", "Not today.", "See you tomorrow.", "What time is it?", "Stop!", "Maybe later.",
                 "Welcome back.", "It works.", "Call me.", "Dinner is ready.", "Pardon me?", "One moment, please.", "Happy birthday!", "%s." % pick(NAMES), "%s?" % pick(PLACES),
                 "Number %d." % rnd.randrange(1, 100), "%s %d." % (pick(MONTHS), rnd.randrange(1, 29)), "Eight o'clock.", "A, B, C.", "%s." % pick(ACR)])

def clause():
    A, B = pick(ADJ), pick(NOUNS)
    return pick([
        "the %s %s %s %s" % (A, B, pick(VERBS), pick(ADV)),
        "%s %s the %s" % (name(), pick(VERBS), B),
        "although the %s was %s" % (B, A),
        "because %s had already %s it" % (name().split()[0], pick(VERBS)),
        "while %s waited in %s" % (name().split()[0], pick(PLACES)),
        "which is why the %s %s was never %s" % (A, B, pick(VERBS)),
        "and then %s asked whether the %s could be %s by %s" % (name().split()[0], B, pick(VERBS), date()),
        "so we left %s with %d %ss" % (pick(PLACES), rnd.randrange(2, 90), pick(NOUNS)),
    ])

def long_sentence():
    parts = [clause() for _ in range(rnd.randrange(3, 7))]
    s = pick([", ", "; ", ", but ", ", and "]).join(parts)
    return s[0].upper() + s[1:] + pick([".", ".", "!", "."])

def sentence(kind=None):
    k = kind or rnd.choices(["simple", "short", "long"], [6, 2, 2])[0]
    return {"simple": simple, "short": short, "long": long_sentence}[k]()

def words(s): return len(s.split())

def main():
    mode, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rnd.seed(seed)
    out = []
    if mode == "fresh":
        for i in range(n):
            k = rnd.choices(["simple", "short", "long"], [6, 2, 2])[0]
            out.append((k, sentence(k)))
    elif mode == "e2e":                 # n texts per length in 5..200 words, sentences of normal length (for the end-to-end timing)
        for tgt in [5, 6, 8, 10, 12, 15, 18, 22, 28, 35, 45, 60, 80, 100, 125, 150, 175, 200]:
            for _ in range(n):
                s = ""
                while words(s) < tgt:
                    s = (s + " " + sentence("short" if tgt < 7 else None)).strip()
                    if tgt < 30 and words(s) >= tgt: break
                out.append(("w%d" % tgt, s))
    elif mode == "bulk":
        targets = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 28, 35, 45, 60, 80, 100, 125, 150]
        for i in range(n):
            tgt = targets[i % len(targets)]
            s = ""
            while words(s) < tgt:
                s = (s + " " + sentence("short" if tgt < 4 else None)).strip()
                if tgt < 30 and words(s) >= tgt: break
            ws = s.split()
            if len(ws) > 150: s = " ".join(ws[:150]).rstrip(",;") + "."
            out.append(("w%d" % tgt, s))
    else:
        for i in range(n):
            k = rnd.randrange(3, 10)
            p = " ".join(sentence() for _ in range(k))
            while words(p) > 150 and ". " in p: p = p.rsplit(". ", 1)[0] + "."
            out.append(("para%d" % k, p))
    for c, t in out: print(c + "\t" + t)
if __name__ == '__main__':
    main()
