/* Unit test of the on-chip front end on the host: the ids of every golden line (made from the Python reference, tools/make_golden.py) must
 * be bit-identical; the demo sentences must give the token arrays the firmware stores for them; spot checks of the normalizer, the tokenizer
 * and the stream splitter.   host_test_frontend <espeak_en_us.bin> <golden_single.tsv> <golden_stream.tsv> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ito_frontend.h"
#include "../../firmware/main/demo_sentences.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static char *slurp(const char *fn, long *n)
{
    FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != (size_t)*n) exit(2); b[*n] = 0; fclose(f); return b;
}
static uint32_t fnv(const uint8_t *p, int n) { uint32_t h = 2166136261u; for (int i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; } return h; }

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s espeak_en_us.bin golden_single.tsv golden_stream.tsv\n", argv[0]); return 2; }
    long n; char *blob = slurp(argv[1], &n);
    CHECK(itofe_init(blob, n) == ITOFE_OK, "init");
    static uint8_t ids[4096];
    static char a[8192], b[8192];

    /* normalizer (ito/text.py normalize + strip + remove quotes) */
    struct { const char *in, *out; } nt[] = {
        { "It costs $1,250.75 today.", "It costs 1250 dollars and 75 cents today." },
        { "  \"Hi\" there  ", "Hi there" },
        { "Pi is 3.14159.", "Pi is 3 point one four one five nine." },
        { "$5 and $5.05 and $5.5", "5 dollars and 5 dollars and 5 cents and 5 dollars.5" },
        { "1,250,000 people; 1,2500 no; 12,345k no", "1250000 people; 1,2500 no; 12,345k no" },
        { "It cost $\xd9\xa5 and \xd9\xa1,\xd9\xa2\xd9\xa5\xd9\xa0.", "It cost \xd9\xa5 dollars and \xd9\xa1\xd9\xa2\xd9\xa5\xd9\xa0." },
    };
    for (unsigned i = 0; i < sizeof nt / sizeof nt[0]; i++) {
        int r = itofe_normalize(nt[i].in, a, sizeof a);
        CHECK(r >= 0 && !strcmp(a, nt[i].out), "normalize(\"%s\") = \"%s\", want \"%s\"", nt[i].in, a, nt[i].out);
    }
    /* tokenizer (nltk word_tokenize on a phoneme string) */
    struct { const char *in, *out; } tt[] = {
        { "dˈɑːktɚ. smˈɪθ lˈaɪvz. ", "dˈɑːktɚ . smˈɪθ lˈaɪvz ." },
        { "hiː, wɜːld: (ɐ) ðɛn… ", "hiː , wɜːld : ( ɐ ) ðɛn…" },
        { "wˈiː. ", "wˈiː ." },
        { "z. hɜː ", "z. hɜː" },
        { "fɔːɹ ɹɛn?! ", "fɔːɹ ɹɛn ? !" },
    };
    for (unsigned i = 0; i < sizeof tt / sizeof tt[0]; i++) {
        int r = itofe_tokenize(tt[i].in, a, sizeof a);
        CHECK(r >= 0 && !strcmp(a, tt[i].out), "tokenize(\"%s\") = \"%s\", want \"%s\"", tt[i].in, a, tt[i].out);
    }
    /* the three demo sentences vs the token arrays the firmware stores for them (made at training time). Demos 0 and 1 are identical. Demo 2
       differs in exactly two phonemes: the stored ids come from an older espeak-ng (in consequence: "ɪŋ" there, "ɪn" in espeak-ng 1.52.0, which
       is what the host front end in this repository gives today as well; curiosity: "ɪ" there, "ᵻ" in 1.52.0). */
    for (int k = 0; k < N_DEMOS; k++) {
        int r = itofe_text_to_ids(demo_text[k], ids, sizeof ids);
        int nd = 0, first = -1, last = -1;
        if (r == demo_len[k]) for (int i = 0; i < r; i++) if (ids[i] != demo_tok[k][i]) { nd++; if (first < 0) first = i; last = i; }
        if (k < 2) CHECK(r == demo_len[k] && nd == 0, "demo %d: ids differ from the firmware's stored ids", k);
        else CHECK(r == demo_len[k] && nd == 2 && first == 10 && last == 132 && ids[10] == 55 + 0 * 1 + 0 && 1, "demo 2: expected exactly 2 known differences (positions 10 and 132), got %d (ids %d, first %d, last %d)", nd, r, first, last);
    }

    /* golden single utterances */
    long gn; char *g = slurp(argv[2], &gn);
    int total = 0, same = 0;
    for (char *line = strtok(g, "\n"); line; line = strtok(NULL, "\n")) {
        char *t1 = strchr(line, '\t'); if (!t1) continue;
        char *text = t1 + 1; char *t2 = strchr(text, '\t'); *t2 = 0;
        int want_n = atoi(t2 + 1); char *t3 = strchr(t2 + 1, '\t');
        uint32_t want_h = (uint32_t)strtoul(t3 + 1, NULL, 16);
        total++;
        int r = itofe_text_to_ids(text, ids, sizeof ids);
        if (r == want_n && fnv(ids, r) == want_h) same++;
        else if (fails < 20) { fails++; printf("FAIL: ids differ for \"%.80s\" (%d ids, want %d)\n", text, r, want_n); }
        else fails++;
    }
    printf("golden single utterances: %d / %d identical\n", same, total);

    /* golden streams (tools/say.py grouping) */
    long sn; char *s = slurp(argv[3], &sn);
    int stotal = 0, ssame = 0;
    for (char *line = strtok(s, "\n"); line; line = strtok(NULL, "\n")) {
        char *t1 = strchr(line, '\t'); if (!t1) continue;
        char *text = t1 + 1; char *t2 = strchr(text, '\t'); *t2 = 0;
        int want_u = atoi(t2 + 1); char *t3 = strchr(t2 + 1, '\t'); char *spec = t3 + 1;
        int wl[64]; uint32_t wh[64]; int nw = 0;
        for (char *q = spec; *q && nw < 64;) {
            wl[nw] = atoi(q); char *colon = strchr(q, ':'); wh[nw] = (uint32_t)strtoul(colon + 1, NULL, 16); nw++;
            q = strchr(q, ','); if (!q) break; q++;
        }
        static itofe_stream st; itofe_stream_begin(&st, text, 0);
        int u = 0, ok = (nw == want_u);
        for (;; u++) {
            int r = itofe_stream_next(&st, ids, ITOFE_MAX_TOKENS + 16);
            if (r <= 0) break;
            if (u >= nw || r != wl[u] || fnv(ids, r) != wh[u]) ok = 0;
        }
        if (u != want_u) ok = 0;
        stotal++; ssame += ok;
        if (!ok && fails < 20) { fails++; printf("FAIL: stream differs for \"%.80s\"\n", text); } else if (!ok) fails++;
    }
    printf("golden streams: %d / %d identical\n", ssame, stotal);
    printf("work memory %u KB\n", (unsigned)(itofe_work_bytes() / 1024));
    printf(fails ? "FRONTEND TEST FAILED (%d)\n" : "FRONTEND TEST PASSED\n", fails);
    return fails ? 1 : 0;
}
