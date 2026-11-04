/* Ito on-chip front end, see ito_frontend.h. GPLv3 or later.
 *
 * The code follows, step by step, what the host does (ito/text.py -> phonemizer 3.x EspeakBackend -> nltk word_tokenize); the
 * comments say which Python rule each block reproduces. The nltk part is a hand-written subset of NLTKWordTokenizer and Punkt
 * that covers the characters a phoneme string can contain; tests/diff_corpus.py compares it with nltk on every corpus line. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ito_frontend.h"
#include "ito_espeak.h"
#include "ucd/ucd.h"
#include "ito_symbols.inc"

#define MAXCP 8192              /* code points in the working arrays (the tokenizer pads with spaces) */
#define MAXPARTS 200
#define POOL_BYTES 32768
#define PHBUF 8192              /* one phonemized chunk */
#define TEXTB (ITOFE_MAX_TEXT + 16)
#define FULLN (ITOFE_MAX_TEXT * 2)

typedef struct {
    uint32_t *cp, *cp2;         /* [MAXCP] */
    int *boff;                  /* [MAXCP+1] */
    int *mt;                    /* [MAXCP] period-context matches */
    uint8_t *det;               /* [MAXCP] */
    char *pool;                 /* [POOL_BYTES] phonemized chunk strings */
    char *ph;                   /* [PHBUF] espeak output of one chunk */
    char *line;                 /* [TEXTB*2] group text / postprocess scratch */
    char *norm;                 /* [TEXTB] */
    char *raw;                  /* [TEXTB*4] phonemizer output */
    char *tok;                  /* [TEXTB*4] tokenizer output */
    char *tmp;                  /* [TEXTB*4] */
    uint8_t *full;              /* [FULLN] ids of a whole sentence before the 400 limit is applied */
} ctx_t;
static ctx_t C;
static int inited;

/* ------------------------------------------------------------------ UTF-8 and character classes */
static int u8dec(const unsigned char *s, uint32_t *cp)
{
    if (s[0] < 0x80) { *cp = s[0]; return s[0] ? 1 : 0; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) { *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) { *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4; }
    *cp = 0xFFFD; return 1;
}
static int u8enc(uint32_t c, char *o)
{
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3F)); o[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3F)); o[2] = (char)(0x80 | ((c >> 6) & 0x3F)); o[3] = (char)(0x80 | (c & 0x3F)); return 4;
}
/* Python str.isspace() == regex \s for str patterns */
static int pyspace(uint32_t c)
{
    if (c < 0x80) return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20);
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static int isdig(uint32_t c) { return c >= '0' && c <= '9'; }   /* [0-9] (phonemizer's decimal-separator rule) */
static int isword(uint32_t c)
{
    if (c < 0x80) return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    return ucd_isalnum(c);
}
static int in_set(uint32_t c, const uint32_t *set) { for (; *set; set++) if (*set == c) return 1; return 0; }

static int decode(const char *s, uint32_t *cp, int *boff, int cap)
{
    int n = 0;
    const unsigned char *p = (const unsigned char *)s;
    for (;;) {
        uint32_t c;
        int l = u8dec(p, &c);
        if (l == 0) break;
        if (n >= cap) return -1;
        if (boff) boff[n] = (int)((const char *)p - s);
        cp[n++] = c;
        p += l;
    }
    if (boff) boff[n] = (int)((const char *)p - s);
    return n;
}
static int encode(const uint32_t *cp, int n, char *out, size_t cap)
{
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        if (o + 5 > cap) return -1;
        o += u8enc(cp[i], out + o);
    }
    out[o] = 0;
    return (int)o;
}

/* growing string into a fixed buffer */
typedef struct { char *p; size_t n, cap; int err; } sb_t;
static void sb_put(sb_t *b, const char *s, size_t l) { if (b->n + l + 1 > b->cap) { b->err = 1; return; } memcpy(b->p + b->n, s, l); b->n += l; b->p[b->n] = 0; }
static void sb_c(sb_t *b, char c) { sb_put(b, &c, 1); }
static void sb_s(sb_t *b, const char *s) { sb_put(b, s, strlen(s)); }
static void sb_init(sb_t *b, char *p, size_t cap) { b->p = p; b->n = 0; b->cap = cap; b->err = 0; if (cap) p[0] = 0; }

/* ------------------------------------------------------------------ normalize: ito/text.py normalize(), then .strip().replace('"', "") */
/* Python's \d is any Unicode decimal digit (category Nd) and int() reads them: zero code points of the Nd blocks of the BMP */
static const uint16_t ND_ZERO[] = { 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0DE6, 0x0E50,
    0x0ED0, 0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900,
    0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0 };
static int dval(uint32_t c)         /* decimal digit value or -1 */
{
    if (c < 0x80) return (c >= '0' && c <= '9') ? (int)(c - '0') : -1;
    if (c < 0x660 || c > 0xFF19) return -1;
    for (int i = 0; ND_ZERO[i]; i++) if (c >= ND_ZERO[i] && c < (uint32_t)ND_ZERO[i] + 10) return (int)(c - ND_ZERO[i]);
    return -1;
}
static const char *const DIGW[10] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };

static int boundary_at(const uint32_t *a, int n, int k)   /* \b at k where a[k-1] is a digit (a word character) */
{
    return k >= n || !isword(a[k]);
}
typedef struct { uint32_t *p; int n, cap, err; } cb_t;
static void cb_c(cb_t *b, uint32_t c) { if (b->n >= b->cap) { b->err = 1; return; } b->p[b->n++] = c; }
static void cb_s(cb_t *b, const char *s) { while (*s) cb_c(b, (unsigned char)*s++); }
static void cb_cp(cb_t *b, const uint32_t *a, int from, int to) { for (int i = from; i < to; i++) cb_c(b, a[i]); }

/* re.sub(r"(?<=\d),(?=\d{3}\b)", "") */
static void n_commas(const uint32_t *a, int n, cb_t *o)
{
    for (int i = 0; i < n; i++) {
        if (a[i] == ',' && i > 0 && dval(a[i - 1]) >= 0 && i + 3 < n && dval(a[i + 1]) >= 0 && dval(a[i + 2]) >= 0 && dval(a[i + 3]) >= 0 && boundary_at(a, n, i + 4))
            continue;
        cb_c(o, a[i]);
    }
}
/* re.sub(r"\$(\d+)\.(\d{2})\b", lambda m: f"{m.group(1)} dollars and {int(m.group(2))} cents") */
static void n_dollar_cents(const uint32_t *a, int n, cb_t *o)
{
    for (int i = 0; i < n; i++) {
        if (a[i] == '$') {
            int j = i + 1;
            while (j < n && dval(a[j]) >= 0) j++;
            if (j > i + 1 && j + 2 < n && a[j] == '.' && dval(a[j + 1]) >= 0 && dval(a[j + 2]) >= 0 && boundary_at(a, n, j + 3)) {
                cb_cp(o, a, i + 1, j);
                cb_s(o, " dollars and ");
                char num[8]; snprintf(num, sizeof num, "%d", dval(a[j + 1]) * 10 + dval(a[j + 2]));
                cb_s(o, num);
                cb_s(o, " cents");
                i = j + 2;
                continue;
            }
        }
        cb_c(o, a[i]);
    }
}
/* re.sub(r"\$(\d+)\b", r"\1 dollars") */
static void n_dollars(const uint32_t *a, int n, cb_t *o)
{
    for (int i = 0; i < n; i++) {
        if (a[i] == '$') {
            int j = i + 1;
            while (j < n && dval(a[j]) >= 0) j++;
            if (j > i + 1 && boundary_at(a, n, j)) {
                cb_cp(o, a, i + 1, j);
                cb_s(o, " dollars");
                i = j - 1;
                continue;
            }
        }
        cb_c(o, a[i]);
    }
}
/* re.sub(r"(\d+)\.(\d+)", lambda m: m1 + " point " + " ".join(digit words of m2)) */
static void n_decimals(const uint32_t *a, int n, cb_t *o)
{
    int i = 0;
    while (i < n) {
        if (dval(a[i]) >= 0) {
            int b = i;
            while (b < n && dval(a[b]) >= 0) b++;
            if (b + 1 < n && a[b] == '.' && dval(a[b + 1]) >= 0) {
                int e = b + 1;
                while (e < n && dval(a[e]) >= 0) e++;
                cb_cp(o, a, i, b);
                cb_s(o, " point ");
                for (int k = b + 1; k < e; k++) {
                    if (k > b + 1) cb_c(o, ' ');
                    cb_s(o, DIGW[dval(a[k])]);
                }
                i = e;
            } else {
                cb_cp(o, a, i, b);
                i = b;
            }
        } else {
            cb_c(o, a[i++]);
        }
    }
}

int itofe_normalize(const char *text, char *out, size_t cap)
{
    if (!inited) return ITOFE_E_BLOB;
    if (strlen(text) > ITOFE_MAX_TEXT) return ITOFE_E_TOO_LONG;
    int n = decode(text, C.cp, NULL, MAXCP / 2);
    if (n < 0) return ITOFE_E_TOO_LONG;
    uint32_t *a = C.cp, *b = C.cp2;
    cb_t o;
    o = (cb_t){ b, 0, MAXCP, 0 }; n_commas(a, n, &o);       if (o.err) return ITOFE_E_TOO_LONG;
    { uint32_t *t = a; a = b; b = t; n = o.n; }
    o = (cb_t){ b, 0, MAXCP, 0 }; n_dollar_cents(a, n, &o); if (o.err) return ITOFE_E_TOO_LONG;
    { uint32_t *t = a; a = b; b = t; n = o.n; }
    o = (cb_t){ b, 0, MAXCP, 0 }; n_dollars(a, n, &o);      if (o.err) return ITOFE_E_TOO_LONG;
    { uint32_t *t = a; a = b; b = t; n = o.n; }
    o = (cb_t){ b, 0, MAXCP, 0 }; n_decimals(a, n, &o);     if (o.err) return ITOFE_E_TOO_LONG;
    { uint32_t *t = a; a = b; b = t; n = o.n; }
    /* .strip() and .replace('"', "") */
    int s0 = 0, e0 = n;
    while (s0 < e0 && pyspace(a[s0])) s0++;
    while (e0 > s0 && pyspace(a[e0 - 1])) e0--;
    size_t k = 0;
    for (int i = s0; i < e0; i++) {
        if (a[i] == '"') continue;
        if (k + 5 > cap) return ITOFE_E_TOO_LONG;
        k += u8enc(a[i], out + k);
    }
    out[k] = 0;
    return (int)k;
}

/* ------------------------------------------------------------------ phonemizer (3.x) EspeakBackend, preserve_punctuation=True, with_stress=True */
static const uint32_t MARKS_OTHER[] = { ';', ':', '!', '?', 0xA1, 0xBF, 0x2014, 0x2026, '"', 0xAB, 0xBB, 0x201C, 0x201D, '(', ')', '{', '}', '[', ']', 0 };
static const uint32_t INLINE_MARKS[] = { '"', 0xAB, 0xBB, 0x201C, 0x201D, 0x2018, 0x2019, 0x201E, 0x2039, 0x203A, '(', ')', '[', ']', '{', '}', 0 };

typedef struct { int is_mark; int s, e; int chunk; } part_t;   /* [s,e) bytes of the normalized text */

/* _postprocess_line() of the espeak backend (separator.phone = '', separator.word = ' ', strip=False, with_stress=True). Appends to pool. */
static int words_count(const char *s)
{
    int n = 0, in = 0;
    for (; *s; s++) { if (*s == ' ') in = 0; else if (!in) { in = 1; n++; } }
    return n;
}
static char *pool_put_pp(size_t *poolpos, const char *line)
{
    char *dst = C.pool + *poolpos;
    size_t cap = POOL_BYTES - *poolpos;
    size_t n = 0;
    /* line.strip() */
    const char *b = line, *e = line + strlen(line);
    while (b < e && (*b == ' ' || (*b >= 9 && *b <= 13))) b++;
    while (e > b && (e[-1] == ' ' || (e[-1] >= 9 && e[-1] <= 13))) e--;
    /* .replace('\n', ' ').replace('  ', ' ') */
    char *t = C.line;
    size_t tn = 0;
    for (const char *p = b; p < e; p++) t[tn++] = (*p == '\n') ? ' ' : *p;
    size_t un = 0;
    char *u = C.line + TEXTB;
    for (size_t i = 0; i < tn;) {
        if (t[i] == ' ' && i + 1 < tn && t[i + 1] == ' ') { u[un++] = ' '; i += 2; }
        else u[un++] = t[i++];
    }
    /* '__' -> '_' ; '_ ' -> ' ' ; remove the remaining '_' (separator.phone is empty) */
    if (un == 0) { if (cap) dst[0] = 0; return NULL; }
    /* (the three steps compose into: drop every '_'; the only visible effect of '_ ' -> ' ' is the same) */
    for (size_t i = 0; i < un; i++) {
        if (u[i] == '_') continue;
        if (n + 3 >= cap) return NULL;
        dst[n++] = u[i];
    }
    /* trailing word separator */
    if (n + 2 > cap) return NULL;
    dst[n++] = ' ';
    dst[n] = 0;
    *poolpos += n + 1;
    return dst;
}

static int is_inline_mark(const char *s, int len)
{
    int any = 0;
    int i = 0;
    while (i < len) {
        uint32_t c; int l = u8dec((const unsigned char *)s + i, &c);
        if (l == 0) break;
        i += l;
        if (pyspace(c)) continue;
        if (!in_set(c, INLINE_MARKS)) return 0;
        any = 1;
    }
    return any;
}

typedef struct { const char *p; int n; } slice_t;

int itofe_phonemizer(const char *norm, char *out, size_t cap)
{
    if (!inited) return ITOFE_E_BLOB;
    int n = decode(norm, C.cp, C.boff, MAXCP);
    if (n < 0) return ITOFE_E_TOO_LONG;
    size_t blen = strlen(norm);

    /* Punctuation.preserve(): split at (\s*(?:[others]|(?<![0-9])[,.]|[,.](?![0-9]))+\s*)+ */
    static part_t parts[MAXPARTS];
    int np = 0, nchunk = 0;
    int cursor = 0;
    int i = 0;
    #define ISMARK(k) (in_set(C.cp[k], MARKS_OTHER) || ((C.cp[k] == ',' || C.cp[k] == '.') && !((k) > 0 && isdig(C.cp[(k) - 1]) && (k) + 1 < n && isdig(C.cp[(k) + 1]))))
    while (i < n) {
        if (pyspace(C.cp[i]) || ISMARK(i)) {
            int j = i, has = 0;
            while (j < n && (pyspace(C.cp[j]) || ISMARK(j))) { if (!pyspace(C.cp[j])) has = 1; j++; }
            if (has) {
                if (C.boff[i] > cursor) {
                    if (np + 2 > MAXPARTS) return ITOFE_E_TOO_LONG;
                    parts[np].is_mark = 0; parts[np].s = cursor; parts[np].e = C.boff[i]; parts[np].chunk = nchunk++; np++;
                }
                if (np + 2 > MAXPARTS) return ITOFE_E_TOO_LONG;
                parts[np].is_mark = 1; parts[np].s = C.boff[i]; parts[np].e = C.boff[j]; parts[np].chunk = -1; np++;
                cursor = C.boff[j];
            }
            i = j;
        } else i++;
    }
    #undef ISMARK
    if ((int)blen > cursor) {
        if (np + 2 > MAXPARTS) return ITOFE_E_TOO_LONG;
        parts[np].is_mark = 0; parts[np].s = cursor; parts[np].e = (int)blen; parts[np].chunk = nchunk++; np++;
    }

    /* phonemize every chunk (espeak, then _postprocess_line) */
    static char *cph[MAXPARTS];            /* postprocessed phonemes per chunk, in the pool */
    size_t pp = 0;
    static int chunk_part[MAXPARTS];
    for (int k = 0; k < np; k++) if (!parts[k].is_mark) chunk_part[parts[k].chunk] = k;
    for (int c = 0; c < nchunk; c++) {
        part_t *pt = &parts[chunk_part[c]];
        int l = pt->e - pt->s;
        if (l >= (int)(TEXTB * 2)) return ITOFE_E_TOO_LONG;
        memcpy(C.line + TEXTB * 0, norm + pt->s, l);                     /* C.line is also the pp scratch: use C.raw for the text */
        memcpy(C.raw, norm + pt->s, l); C.raw[l] = 0;
        if (itoes_text_to_phonemes(C.raw, C.ph, PHBUF) < 0) return ITOFE_E_ESPEAK;
        cph[c] = pool_put_pp(&pp, C.ph);
        if (!cph[c]) { cph[c] = C.pool + pp; if (pp + 1 > POOL_BYTES) return ITOFE_E_TOO_LONG; C.pool[pp++] = 0; cph[c] = C.pool + pp - 1; }
    }

    /* inline groups: chunks separated by exactly one inline mark (quotes, brackets) are phonemized as a whole as well,
       and the words are cut back into the chunks when the word counts agree (Punctuation._context) */
    static const char *cctx[MAXPARTS];     /* in-context replacement per chunk, or NULL */
    static int cctx_n[MAXPARTS];
    for (int c = 0; c < nchunk; c++) cctx[c] = NULL;
    {
        int group[MAXPARTS], ng = 0;
        for (int k = 0; k <= np; k++) {
            int newgroup_needed = 0, add = 0;
            if (k < np && !parts[k].is_mark) {
                if (ng > 0 && k >= 2 && !parts[k - 2].is_mark && parts[k - 2].chunk == group[ng - 1] &&
                    is_inline_mark(norm + parts[k - 1].s, parts[k - 1].e - parts[k - 1].s)) add = 1;
                else newgroup_needed = 1;
            } else if (k == np) newgroup_needed = 1;   /* end of the line */
            if (add) { group[ng++] = parts[k].chunk; continue; }
            if (newgroup_needed) {
                if (ng > 1) {
                    /* phonemize ' '.join(raw chunks) */
                    size_t tl = 0;
                    for (int g = 0; g < ng; g++) {
                        part_t *pt = &parts[chunk_part[group[g]]];
                        int l = pt->e - pt->s;
                        if (tl + l + 2 > TEXTB * 2) return ITOFE_E_TOO_LONG;
                        if (g) C.raw[tl++] = ' ';
                        memcpy(C.raw + tl, norm + pt->s, l); tl += l;
                    }
                    C.raw[tl] = 0;
                    if (itoes_text_to_phonemes(C.raw, C.ph, PHBUF) < 0) return ITOFE_E_ESPEAK;
                    char *gp = pool_put_pp(&pp, C.ph);
                    if (gp) {
                        int counts[MAXPARTS], total = 0, ok = 1;
                        for (int g = 0; g < ng; g++) { counts[g] = words_count(cph[group[g]]); if (!counts[g]) ok = 0; total += counts[g]; }
                        if (ok && words_count(gp) == total) {
                            /* cut before the first word of each chunk but the first */
                            int starts[MAXPARTS], acc = 0;
                            for (int g = 0; g + 1 < ng; g++) { acc += counts[g]; starts[g] = acc; }
                            int g = 0, nwords = 0;
                            const char *seg = gp;
                            const char *q = gp;
                            for (;;) {
                                /* q at the start of a piece (between spaces) */
                                const char *ep = strchr(q, ' ');
                                int plen = ep ? (int)(ep - q) : (int)strlen(q);
                                if (plen > 0) {
                                    if (g + 1 < ng && nwords == starts[g]) {
                                        /* close the current chunk's text at q (pieces joined with ' ', the separating space dropped) */
                                        int sl = (int)(q - seg);
                                        if (sl > 0 && seg[sl - 1] == ' ') sl--;
                                        char *d = C.pool + pp; if (pp + sl + 1 > POOL_BYTES) return ITOFE_E_TOO_LONG;
                                        memcpy(d, seg, sl); d[sl] = 0; pp += sl + 1;
                                        cctx[group[g]] = d; cctx_n[group[g]] = sl;
                                        g++; seg = q;
                                    }
                                    nwords++;
                                }
                                if (!ep) break;
                                q = ep + 1;
                            }
                            {   /* last chunk: the rest */
                                int sl = (int)strlen(seg);
                                char *d = C.pool + pp; if (pp + sl + 1 > POOL_BYTES) return ITOFE_E_TOO_LONG;
                                memcpy(d, seg, sl); d[sl] = 0; pp += sl + 1;
                                cctx[group[g]] = d; cctx_n[group[g]] = sl;
                            }
                        }
                    }
                }
                ng = 0;
                if (k < np && !parts[k].is_mark) group[ng++] = parts[k].chunk;
            }
        }
    }

    /* Punctuation.restore() */
    sb_t o; sb_init(&o, out, cap);
    int have_mark = 0, mark_space = 0;
    for (int k = 0; k < np; k++) {
        if (parts[k].is_mark) {
            if (o.n > 0 && o.p[o.n - 1] == ' ') { o.n--; o.p[o.n] = 0; }
            sb_put(&o, norm + parts[k].s, parts[k].e - parts[k].s);
            have_mark = 1; mark_space = norm[parts[k].e - 1] == ' ';
        } else {
            int c = parts[k].chunk;
            const char *ch = cctx[c] ? cctx[c] : cph[c];
            if (*ch) { sb_s(&o, ch); have_mark = 0; }
        }
    }
    if (have_mark && !mark_space) sb_c(&o, ' ');
    return o.err ? ITOFE_E_TOO_LONG : (int)o.n;
}

/* ------------------------------------------------------------------ nltk word_tokenize on the phoneme string (see header comment) */
static const char *const ABBREV[] = {   /* nltk punkt_tab english abbrev_types */
    "a.a", "a.c", "a.d", "a.g", "a.h", "a.m", "a.m.e", "a.s", "a.t", "adm", "ala", "ariz", "aug", "ave", "b.f", "b.v", "bros", "c",
    "c.i.t", "c.o.m.b", "c.v", "calif", "chg", "cie", "co", "col", "colo", "conn", "corp", "cos", "ct", "d", "d.c", "d.h", "d.w", "dec",
    "dr", "e", "e.f", "e.h", "e.l", "e.m", "f", "f.g", "f.j", "feb", "fla", "fri", "ft", "g", "g.d", "g.f", "g.k", "ga", "gen", "h",
    "h.c", "h.f", "h.m", "i.m.s", "ill", "inc", "j.b", "j.c", "j.j", "j.k", "j.p", "j.r", "jan", "jr", "k", "kan", "ky", "l", "l.a",
    "l.f", "l.p", "lt", "ltd", "m", "m.b.a", "m.d.c", "m.j", "maj", "messrs", "mg", "mich", "minn", "mr", "mrs", "ms", "n", "n.c",
    "n.d", "n.h", "n.j", "n.m", "n.v", "n.y", "nev", "nov", "oct", "ok", "okla", "ore", "p", "p.a.m", "p.m", "pa", "ph.d", "prof",
    "r", "r.a", "r.h", "r.i", "r.j", "r.k", "r.t", "rep", "reps", "s", "s.a", "s.a.y", "s.c", "s.g", "s.p.a", "s.s", "sen", "sep",
    "sept", "sr", "st", "sw", "t", "t.j", "tenn", "tues", "u.k", "u.n", "u.s", "u.s.a", "u.s.s.r", "v", "va", "vs", "vt", "w", "w.c",
    "w.r", "w.va", "w.w", "wash", "wed", "wis", "yr", NULL };
static int is_abbrev(const char *t)
{
    for (int i = 0; ABBREV[i]; i++) if (strcmp(ABBREV[i], t) == 0) return 1;
    return 0;
}
/* Punkt NonWord: [)";}\]*:@'({[‘’“”«»?!] */
static const uint32_t NONWORD[] = { ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '[', 0x2018, 0x2019, 0x201C, 0x201D, 0xAB, 0xBB, '?', '!', 0 };
/* characters of the Treebank final-period class ([\]\)}>"'»”’ ]) and of Punkt's boundary realignment (["')\]}‘’“”«»]) */
static const uint32_t CLOSERS[] = { ']', ')', '}', '>', '"', '\'', 0xBB, 0x201D, 0x2019, ' ', 0 };
static const uint32_t REALIGN[] = { '"', '\'', ')', ']', '}', 0x2018, 0x2019, 0x201C, 0x201D, 0xAB, 0xBB, 0 };

static int tail_is_closers(const uint32_t *a, int j, int e)     /* a[j..e) is [class]*+ then whitespace */
{
    while (j < e && in_set(a[j], CLOSERS)) j++;
    while (j < e && pyspace(a[j])) j++;
    return j == e;
}
static int ascii_ws(uint32_t c) { return c == ' ' || (c >= 9 && c <= 13); }    /* string.whitespace */

/* Punkt first/second pass for the period at a[i] (a match followed by a non-word or by whitespace + token): is it a sentence break?
   Only the abbreviation list and the "initial" rule can say no; the model has no sentence-starter / collocation statistics for IPA words. */
static int period_breaks(const uint32_t *a, int n, int i)
{
    int s = i;
    while (s > 0 && !pyspace(a[s - 1]) && !in_set(a[s - 1], NONWORD)) s--;
    int len = i - s;
    if (len <= 0 || len >= 24) return 1;
    char t[28]; int ascii = 1;
    for (int k = 0; k < len; k++) { uint32_t c = a[s + k]; if (c >= 0x80) { ascii = 0; break; } t[k] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c); }
    if (ascii) {
        t[len] = 0;
        if (is_abbrev(t)) return 0;
        char *dash = strrchr(t, '-');
        if (dash && is_abbrev(dash + 1)) return 0;
    }
    if (len == 1 && (ucd_isalpha(a[s]) || a[s] == '_')) {                               /* an initial: "X." */
        int j = i + 1;
        if (j < n && pyspace(a[j])) while (j < n && pyspace(a[j])) j++;
        uint32_t nc = j < n ? a[j] : 0;
        if (nc == ';' || nc == ':' || nc == ',' || nc == '?' || nc == '!') return 0;
        if (nc && ucd_islower(nc)) return 0;
    }
    return 1;
}

/* Which periods does word_tokenize() separate from their word? nltk first cuts the text into sentences with Punkt, then the Treebank
   rule only separates the period that ends a sentence (followed by closing quotes/brackets). Marks det[i]=1 for those periods. */
static void find_detached(const uint32_t *a, int n, uint8_t *det, int *mt)
{
    memset(det, 0, n);
    int nm = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] != '.' && a[i] != '?' && a[i] != '!') continue;
        int ok = 0;
        if (i + 1 < n) {
            if (in_set(a[i + 1], NONWORD)) ok = 1;
            else if (pyspace(a[i + 1])) { int j = i + 1; while (j < n && pyspace(a[j])) j++; ok = j < n; }
        }
        if (ok) mt[nm++] = i;            /* period_context_re matches */
    }
    int start = 0;
    for (int k = 0; k < nm; k++) {
        int m = mt[k];
        if (k + 1 < nm) {                /* overlapping contexts: only the last match of a whitespace-free run is kept */
            int ws = 0;
            for (int x = m; x < mt[k + 1]; x++) if (ascii_ws(a[x])) { ws = 1; break; }
            if (!ws) continue;
        }
        if (a[m] == '.' && !period_breaks(a, n, m)) continue;
        /* sentence = [start, m+1) plus the closing punctuation Punkt moves back from the next sentence */
        int s2 = m + 1;
        while (s2 < n && pyspace(a[s2])) s2++;
        int e = m + 1;
        int r = s2;
        while (r < n && in_set(a[r], REALIGN)) r++;
        if (r > s2 && (r >= n || pyspace(a[r]) || (a[r] == '-' && r + 1 < n && a[r + 1] == '-'))) { e = r; s2 = r; while (s2 < n && pyspace(a[s2])) s2++; }
        /* Treebank: ([^\.])(\.)([\]\)}>"'»”’ ]*+)\s*$ on [start, e) */
        for (int q = e - 1; q > start; q--) {
            if (a[q] == '.') { if (a[q - 1] != '.' && tail_is_closers(a, q + 1, e)) det[q] = 1; break; }
        }
        start = s2;
    }
    /* last sentence: [start, len(text.rstrip())) */
    int e = n;
    while (e > start && pyspace(a[e - 1])) e--;
    for (int q = e - 1; q > start; q--) {
        if (a[q] == '.') { if (a[q - 1] != '.' && tail_is_closers(a, q + 1, e)) det[q] = 1; break; }
    }
}

int itofe_tokenize(const char *raw, char *out, size_t cap)
{
    if (!inited) return ITOFE_E_BLOB;
    int n = decode(raw, C.cp, NULL, MAXCP / 2);
    if (n < 0) return ITOFE_E_TOO_LONG;
    uint32_t *a = C.cp, *b = C.cp2;
    int m = 0;
    /* sentence-final periods are separated (Punkt + Treebank) */
    find_detached(a, n, C.det, C.mt);
    for (int i = 0; i < n; i++) {
        if (C.det[i]) b[m++] = ' ';
        b[m++] = a[i];
    }
    { uint32_t *t = a; a = b; b = t; n = m; m = 0; }
    /* PUNCTUATION: ([:,])([^\d]) -> " \1 \2" (the second character is consumed) */
    for (int i = 0; i < n;) {
        if ((a[i] == ':' || a[i] == ',') && i + 1 < n && dval(a[i + 1]) < 0) { b[m++] = ' '; b[m++] = a[i]; b[m++] = ' '; b[m++] = a[i + 1]; i += 2; }
        else b[m++] = a[i++];
    }
    { uint32_t *t = a; a = b; b = t; n = m; m = 0; }
    /* ([:,])$ -> " \1 " ($ also matches before a final newline) */
    {
        int e = n;
        if (e > 0 && a[e - 1] == '\n') e--;
        for (int i = 0; i < n; i++) {
            if (i == e - 1 && (a[i] == ':' || a[i] == ',')) { b[m++] = ' '; b[m++] = a[i]; b[m++] = ' '; }
            else b[m++] = a[i];
        }
        uint32_t *t = a; a = b; b = t; n = m; m = 0;
    }
    /* padding rules: «“‘„` (starting quotes), runs of 2+ periods, ; @ # $ % & , U+2012..2015, ? !, ] [ ( ) { } < >, --, » ” ’ */
    for (int i = 0; i < n;) {
        uint32_t c = a[i];
        if (c == '.' && i + 1 < n && a[i + 1] == '.') {
            int j = i; while (j < n && a[j] == '.') j++;
            b[m++] = ' '; for (int k = i; k < j; k++) b[m++] = '.'; b[m++] = ' '; i = j; continue;
        }
        if (c == '-' && i + 1 < n && a[i + 1] == '-') { b[m++] = ' '; b[m++] = '-'; b[m++] = '-'; b[m++] = ' '; i += 2; continue; }
        int pad = c == 0xAB || c == 0x201C || c == 0x2018 || c == 0x201E || c == '`' ||
                  c == ';' || c == '@' || c == '#' || c == '$' || c == '%' || c == '&' ||
                  (c >= 0x2012 && c <= 0x2015) || c == '?' || c == '!' ||
                  c == ']' || c == '[' || c == '(' || c == ')' || c == '{' || c == '}' || c == '<' || c == '>' ||
                  c == 0xBB || c == 0x201D || c == 0x2019;
        if (pad) { b[m++] = ' '; b[m++] = c; b[m++] = ' '; } else b[m++] = c;
        i++;
        if (m + 8 > MAXCP) return ITOFE_E_TOO_LONG;
    }
    { uint32_t *t = a; a = b; b = t; n = m; m = 0; }
    /* text.split() and " ".join() */
    sb_t o; sb_init(&o, out, cap);
    int first = 1;
    for (int i = 0; i < n;) {
        if (pyspace(a[i])) { i++; continue; }
        int j = i; while (j < n && !pyspace(a[j])) j++;
        if (!first) sb_c(&o, ' ');
        first = 0;
        for (int k = i; k < j; k++) { char tmp[5]; int l = u8enc(a[k], tmp); sb_put(&o, tmp, l); }
        i = j;
    }
    return o.err ? ITOFE_E_TOO_LONG : (int)o.n;
}

/* ------------------------------------------------------------------ symbols */
static int sym_id(uint32_t c)
{
    int lo = 0, hi = ITOFE_N_SYMBOLS - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (itofe_symtab[mid].cp == c) return itofe_symtab[mid].id;
        if (itofe_symtab[mid].cp < c) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
int itofe_symbols(const char *tok, uint8_t *ids, int cap)
{
    int n = 0;
    if (cap < 1) return ITOFE_E_TOO_LONG;
    ids[n++] = 0;
    const unsigned char *p = (const unsigned char *)tok;
    for (;;) {
        uint32_t c; int l = u8dec(p, &c);
        if (l == 0) break;
        p += l;
        int id = c < 0x10000 ? sym_id(c) : -1;
        if (id < 0) continue;             /* "Unknown symbols are skipped" (ito/text.py phonemes_to_ids) */
        if (n >= cap) return ITOFE_E_TOO_LONG;
        ids[n++] = (uint8_t)id;
    }
    return n;
}

/* ------------------------------------------------------------------ whole pipeline */
int itofe_init(const void *blob, size_t len)
{
    if (inited) return ITOFE_OK;
    size_t sz_cp = MAXCP * sizeof(uint32_t);
    C.cp = ITOFE_ALLOC(sz_cp); C.cp2 = ITOFE_ALLOC(sz_cp);
    C.boff = ITOFE_ALLOC((MAXCP + 1) * sizeof(int)); C.mt = ITOFE_ALLOC(MAXCP * sizeof(int)); C.det = ITOFE_ALLOC(MAXCP);
    C.pool = ITOFE_ALLOC(POOL_BYTES); C.ph = ITOFE_ALLOC(PHBUF);
    C.line = ITOFE_ALLOC(TEXTB * 2 + 64); C.norm = ITOFE_ALLOC(TEXTB);
    C.raw = ITOFE_ALLOC(TEXTB * 4); C.tok = ITOFE_ALLOC(TEXTB * 4); C.tmp = ITOFE_ALLOC(TEXTB * 4); C.full = ITOFE_ALLOC(FULLN);
    if (!C.cp || !C.cp2 || !C.boff || !C.mt || !C.det || !C.pool || !C.ph || !C.line || !C.norm || !C.raw || !C.tok || !C.tmp || !C.full) return ITOFE_E_NOMEM;
    if (itoes_init(blob, len) != 0) return ITOFE_E_BLOB;
    inited = 1;
    return ITOFE_OK;
}
size_t itofe_work_bytes(void)
{
    return MAXCP * 4 * 2 + (MAXCP + 1) * 4 + MAXCP * 5 + POOL_BYTES + PHBUF + TEXTB * 2 + 64 + TEXTB + TEXTB * 4 * 3 + FULLN;
}

int itofe_text_to_ids(const char *text, uint8_t *ids, int cap)
{
    int r = itofe_normalize(text, C.norm, TEXTB);
    if (r < 0) return r;
    r = itofe_phonemizer(C.norm, C.raw, TEXTB * 4);
    if (r < 0) return r;
    r = itofe_tokenize(C.raw, C.tok, TEXTB * 4);
    if (r < 0) return r;
    return itofe_symbols(C.tok, ids, cap);
}

/* ------------------------------------------------------------------ stream: tools/say.py sentences() */
void itofe_stream_begin(itofe_stream *s, const char *text, int fast_first)
{
    memset(s, 0, sizeof(*s));
    s->text = text; s->fast_first = fast_first;
    /* text.strip() */
    const unsigned char *p = (const unsigned char *)text;
    for (;;) { uint32_t c; int l = u8dec(p, &c); if (!l || !pyspace(c)) break; p += l; }
    s->pos = (size_t)((const char *)p - text);
}
/* next sentence of re.split(r'(?<=[.!?;])\s+', text.strip()), stripped; copies it to out; returns its length or 0 at the end */
static int next_sentence(itofe_stream *s, char *out, int cap)
{
    const char *t = s->text;
    size_t i = s->pos, L = strlen(t);
    while (i < L) {                                /* skip whitespace (also the separator after the previous sentence) */
        uint32_t c; int l = u8dec((const unsigned char *)t + i, &c);
        if (!pyspace(c)) break;
        i += l;
    }
    if (i >= L) { s->pos = L; return 0; }
    size_t start = i, end = i;
    uint32_t prev = 0;
    while (i < L) {
        uint32_t c; int l = u8dec((const unsigned char *)t + i, &c);
        if (pyspace(c) && (prev == '.' || prev == '!' || prev == '?' || prev == ';')) break;   /* split here */
        prev = c; i += l;
        if (!pyspace(c)) end = i;
    }
    /* the sentence is t[start, i) stripped; if the text ends (no split), strip trailing whitespace */
    int n = (int)(end - start);
    if (n >= cap) return ITOFE_E_TOO_LONG;
    memcpy(out, t + start, n); out[n] = 0;
    s->pos = i;
    return n > 0 ? n : -1000;
}
/* one "piece": a sentence, or the first half of one that has more than 400 tokens (say.py splits at the middle word).
   Writes the piece text to text_out (ITOFE_MAX_TEXT + 8 bytes) and its ids to ids. */
static int next_piece(itofe_stream *s, char *text_out, uint8_t *ids, int cap)
{
    if (!s->cur[0]) {
        int n = next_sentence(s, s->cur, sizeof s->cur);
        if (n == 0) return 0;
        if (n < 0 && n != -1000) return n;
    }
    uint8_t *full = C.full;
    int k = itofe_text_to_ids(s->cur, full, FULLN);
    if (k < 0) return k;
    if (k <= cap) memcpy(ids, full, k);
    if (k > ITOFE_MAX_TOKENS) {
        /* words = p.split(); cut = max(1, len(words)//2); piece = " ".join(words[:cut]); rest = " ".join(words[cut:]) */
        const char *p = s->cur;
        int nw = 0;
        { int in = 0; for (const unsigned char *q = (const unsigned char *)p; *q;) { uint32_t c; int l = u8dec(q, &c); if (pyspace(c)) in = 0; else if (!in) { in = 1; nw++; } q += l; } }
        int cut = nw / 2; if (cut < 1) cut = 1;
        char *piece = text_out; char rest[ITOFE_MAX_TEXT + 8];
        size_t pn = 0, rn = 0; int w = 0, in = 0;
        for (const unsigned char *q = (const unsigned char *)p; *q;) {
            uint32_t c; int l = u8dec(q, &c);
            if (pyspace(c)) in = 0;
            else {
                if (!in) { in = 1; w++; if (w <= cut ? pn : rn) { if (w <= cut) piece[pn++] = ' '; else rest[rn++] = ' '; } }
                if (w <= cut) { memcpy(piece + pn, q, l); pn += l; } else { memcpy(rest + rn, q, l); rn += l; }
            }
            q += l;
        }
        piece[pn] = 0; rest[rn] = 0;
        strcpy(s->cur, rest);
        k = itofe_text_to_ids(piece, full, FULLN);      /* the first half is not checked again (say.py), it may still exceed 400: truncated to cap */
        if (k < 0) return k;
        if (k > cap) k = cap;
        memcpy(ids, full, k);
        return k;
    }
    strcpy(text_out, s->cur);
    s->cur[0] = 0;
    if (k > cap) k = cap;
    return k;
}

int itofe_stream_next(itofe_stream *s, uint8_t *ids, int cap)
{
    int n;

    if (s->have_next) { memcpy(ids, s->next_ids, s->next_n); n = s->next_n; strcpy(s->gtext, s->next_text); s->have_next = 0; }
    else { n = next_piece(s, s->gtext, ids, cap); if (n <= 0) return n; }
    int pieces = 1;
    if (!(s->fast_first && s->n_out == 0)) {
        static char ptext[ITOFE_MAX_TEXT + 8], mtext[2 * ITOFE_MAX_TEXT + 16];
        uint8_t tmp[ITOFE_MAX_TOKENS + 16];
        for (;;) {                                  /* merge following pieces while the merged utterance has <= 200 tokens (say.py) */
            int k = next_piece(s, ptext, tmp, sizeof tmp);
            if (k < 0) return k;
            if (k == 0) break;
            int accept = 0;
            int sum = n + k;                         /* the merged text has at most this many ids (a sentence-final period of the first */
            if (sum <= ITOFE_GROUP_TOKENS) accept = 1;   /* text can only lose its separating space); exact check in the borderline */
            else if (sum <= ITOFE_GROUP_TOKENS + 8) {
                if (strlen(s->gtext) + strlen(ptext) + 2 < sizeof mtext) {
                    snprintf(mtext, sizeof mtext, "%s %s", s->gtext, ptext);
                    uint8_t *chk = (uint8_t *)C.tmp + 0;   /* scratch is reused by the pipeline: use a local */
                    uint8_t *ids2 = C.full;
                    (void)chk;
                    int m = itofe_text_to_ids(mtext, ids2, FULLN);
                    accept = m >= 0 && m <= ITOFE_GROUP_TOKENS;
                }
            }
            if (accept) {
                size_t gl = strlen(s->gtext);
                if (gl + strlen(ptext) + 2 >= sizeof s->gtext) accept = 0;
                else { s->gtext[gl] = ' '; strcpy(s->gtext + gl + 1, ptext); n = sum; pieces++; }
            }
            if (!accept) {
                memcpy(s->next_ids, tmp, k); s->next_n = k; strcpy(s->next_text, ptext); s->have_next = 1;
                break;
            }
        }
        if (pieces > 1) {                            /* the exact ids of the merged text (differ from the concatenation only where a final period loses its space) */
            int m = itofe_text_to_ids(s->gtext, ids, cap);
            if (m < 0) return m;
            n = m;
        }
    }
    s->n_out++;
    return n;
}

/* all stages at once, for the tests */
int itofe_stages(const char *text, char *norm, char *raw, char *tok, uint8_t *ids, int cap)
{
    norm[0] = raw[0] = tok[0] = 0;
    int r = itofe_normalize(text, C.norm, TEXTB); if (r < 0) return r;
    strcpy(norm, C.norm);
    r = itofe_phonemizer(C.norm, C.raw, TEXTB * 4); if (r < 0) return r;
    strcpy(raw, C.raw);
    r = itofe_tokenize(C.raw, C.tok, TEXTB * 4); if (r < 0) return r;
    strcpy(tok, C.tok);
    return itofe_symbols(C.tok, ids, cap);
}
