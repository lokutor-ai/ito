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

#define MAXCP 4096              /* code points in the working arrays (the tokenizer pads with spaces) */
#define MAXPARTS 200
#define POOL_BYTES 24576
#define PHBUF 8192              /* one phonemized chunk */
#define TEXTB (ITOFE_MAX_TEXT + 16)

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
static int isdig(uint32_t c) { return c >= '0' && c <= '9'; }   /* ASCII digits (Unicode digits are not normalised: documented limitation) */
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
static const char *const DIGW[10] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };

static int boundary_after(const char *s, size_t L, size_t k)   /* \b at k where s[k-1] is a digit (word char) */
{
    if (k >= L) return 1;
    uint32_t c; u8dec((const unsigned char *)s + k, &c);
    return !isword(c);
}
/* re.sub(r"(?<=\d),(?=\d{3}\b)", "") */
static void n_commas(const char *s, sb_t *o)
{
    size_t L = strlen(s);
    for (size_t i = 0; i < L; i++) {
        if (s[i] == ',' && i > 0 && isdig((unsigned char)s[i - 1]) && i + 3 < L &&
            isdig((unsigned char)s[i + 1]) && isdig((unsigned char)s[i + 2]) && isdig((unsigned char)s[i + 3]) &&
            boundary_after(s, L, i + 4))
            continue;
        sb_c(o, s[i]);
    }
}
/* re.sub(r"\$(\d+)\.(\d{2})\b", "N dollars and M cents") */
static void n_dollar_cents(const char *s, sb_t *o)
{
    size_t L = strlen(s);
    for (size_t i = 0; i < L; i++) {
        if (s[i] == '$') {
            size_t j = i + 1;
            while (j < L && isdig((unsigned char)s[j])) j++;
            if (j > i + 1 && j + 2 < L && s[j] == '.' && isdig((unsigned char)s[j + 1]) && isdig((unsigned char)s[j + 2]) && boundary_after(s, L, j + 3)) {
                sb_put(o, s + i + 1, j - i - 1);
                sb_s(o, " dollars and ");
                char num[8]; snprintf(num, sizeof num, "%d", (s[j + 1] - '0') * 10 + (s[j + 2] - '0'));
                sb_s(o, num);
                sb_s(o, " cents");
                i = j + 2;
                continue;
            }
        }
        sb_c(o, s[i]);
    }
}
/* re.sub(r"\$(\d+)\b", r"\1 dollars") */
static void n_dollars(const char *s, sb_t *o)
{
    size_t L = strlen(s);
    for (size_t i = 0; i < L; i++) {
        if (s[i] == '$') {
            size_t j = i + 1;
            while (j < L && isdig((unsigned char)s[j])) j++;
            if (j > i + 1 && boundary_after(s, L, j)) {
                sb_put(o, s + i + 1, j - i - 1);
                sb_s(o, " dollars");
                i = j - 1;
                continue;
            }
        }
        sb_c(o, s[i]);
    }
}
/* re.sub(r"(\d+)\.(\d+)", lambda m: m1 + " point " + " ".join(digit words of m2)) */
static void n_decimals(const char *s, sb_t *o)
{
    size_t L = strlen(s);
    size_t i = 0;
    while (i < L) {
        if (isdig((unsigned char)s[i])) {
            size_t b = i;
            while (b < L && isdig((unsigned char)s[b])) b++;
            if (b + 1 < L && s[b] == '.' && isdig((unsigned char)s[b + 1])) {
                size_t e = b + 1;
                while (e < L && isdig((unsigned char)s[e])) e++;
                sb_put(o, s + i, b - i);
                sb_s(o, " point ");
                for (size_t k = b + 1; k < e; k++) {
                    if (k > b + 1) sb_c(o, ' ');
                    sb_s(o, DIGW[s[k] - '0']);
                }
                i = e;
            } else {
                sb_put(o, s + i, b - i);
                i = b;
            }
        } else {
            sb_c(o, s[i++]);
        }
    }
}

int itofe_normalize(const char *text, char *out, size_t cap)
{
    if (!inited) return ITOFE_E_BLOB;
    size_t L = strlen(text);
    if (L > ITOFE_MAX_TEXT) return ITOFE_E_TOO_LONG;
    char *b1 = C.tmp, *b2 = C.tmp + TEXTB * 2;
    sb_t o;
    sb_init(&o, b1, TEXTB * 2); n_commas(text, &o);
    if (o.err) return ITOFE_E_TOO_LONG;
    sb_init(&o, b2, TEXTB * 2); n_dollar_cents(b1, &o);
    if (o.err) return ITOFE_E_TOO_LONG;
    sb_init(&o, b1, TEXTB * 2); n_dollars(b2, &o);
    if (o.err) return ITOFE_E_TOO_LONG;
    sb_init(&o, b2, TEXTB * 2); n_decimals(b1, &o);
    if (o.err) return ITOFE_E_TOO_LONG;
    /* .strip() */
    const unsigned char *s = (const unsigned char *)b2;
    const unsigned char *e = s + o.n;
    for (;;) { uint32_t c; int l = u8dec(s, &c); if (s >= e || !pyspace(c)) break; s += l; }
    while (e > s) {
        const unsigned char *q = e - 1;
        while (q > s && (*q & 0xC0) == 0x80) q--;
        uint32_t c; u8dec(q, &c);
        if (!pyspace(c)) break;
        e = q;
    }
    /* .replace('"', "") */
    size_t n = 0;
    for (; s < e; s++) {
        if (*s == '"') continue;
        if (n + 1 >= cap) return ITOFE_E_TOO_LONG;
        out[n++] = (char)*s;
    }
    out[n] = 0;
    return (int)n;
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
        if ((a[i] == ':' || a[i] == ',') && i + 1 < n && !isdig(a[i + 1])) { b[m++] = ' '; b[m++] = a[i]; b[m++] = ' '; b[m++] = a[i + 1]; i += 2; }
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
    C.raw = ITOFE_ALLOC(TEXTB * 4); C.tok = ITOFE_ALLOC(TEXTB * 4); C.tmp = ITOFE_ALLOC(TEXTB * 4);
    if (!C.cp || !C.cp2 || !C.boff || !C.mt || !C.det || !C.pool || !C.ph || !C.line || !C.norm || !C.raw || !C.tok || !C.tmp) return ITOFE_E_NOMEM;
    if (itoes_init(blob, len) != 0) return ITOFE_E_BLOB;
    inited = 1;
    return ITOFE_OK;
}
size_t itofe_work_bytes(void)
{
    return MAXCP * 4 * 2 + (MAXCP + 1) * 4 + MAXCP * 5 + POOL_BYTES + PHBUF + TEXTB * 2 + 64 + TEXTB + TEXTB * 4 * 3;
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
