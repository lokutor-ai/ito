/* fe_cli <blob> <corpus.tsv|-> : for each corpus line prints a JSON object {"norm","raw","tok","ids"} (same fields as tools/make_reference.py).
 * fe_cli <blob> --text "..." : prints the ids of the text. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ito_frontend.h"

static char *slurp(const char *fn, long *n)
{
    FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != (size_t)*n) exit(2); b[*n] = 0; fclose(f); return b;
}
static void jstr(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { putchar('\\'); putchar(*s); }
        else if ((unsigned char)*s < 0x20) printf("\\u%04x", (unsigned char)*s);
        else putchar(*s);
    }
    putchar('"');
}
extern int itofe_stages(const char *text, char *norm, char *raw, char *tok, uint8_t *ids, int cap);
int main(int argc, char **argv)
{
    long n; char *blob = slurp(argv[1], &n);
    int r = itofe_init(blob, n);
    if (r) { fprintf(stderr, "init %d\n", r); return 1; }
    static char norm[8192], raw[8192], tok[8192]; static uint8_t ids[1024];
    if (argc > 3 && !strcmp(argv[2], "--text")) {
        int k = itofe_stages(argv[3], norm, raw, tok, ids, 1024);
        printf("%s\n%s\n", raw, tok);
        for (int i = 0; i < k; i++) printf("%d%c", ids[i], i + 1 < k ? ',' : '\n');
        return 0;
    }
    long cn; char *corpus = slurp(argv[2], &cn);
    for (char *line = strtok(corpus, "\n"); line; line = strtok(NULL, "\n")) {
        char *tab = strchr(line, '\t'); char *text = tab ? tab + 1 : line;
        int k = itofe_stages(text, norm, raw, tok, ids, 1024);
        printf("{\"rc\":%d,\"text\":", k); jstr(text);
        printf(",\"norm\":"); jstr(norm); printf(",\"raw\":"); jstr(raw); printf(",\"tok\":"); jstr(tok); printf(",\"ids\":[");
        for (int i = 0; i < k; i++) printf("%d%s", ids[i], i + 1 < k ? "," : "");
        printf("]}\n");
    }
    return 0;
}
