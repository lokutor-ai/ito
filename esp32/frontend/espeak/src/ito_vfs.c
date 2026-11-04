/* ITO: see ito_vfs.h. Part of the Ito on-chip front end; GPLv3 (see ../COPYING). */
#include <stdint.h>
#include <string.h>
#include "ito_vfs.h"

typedef struct { char name[28]; uint32_t off, len; } vfs_ent;
static const uint8_t *blob_p;
static uint32_t blob_n;
static const vfs_ent *ents;

int ito_vfs_set_blob(const void *blob, size_t len)
{
	const uint8_t *b = (const uint8_t *)blob;
	if (len < 12 || memcmp(b, "ITFE", 4) != 0) return -1;
	uint32_t ver, n;
	memcpy(&ver, b + 4, 4); memcpy(&n, b + 8, 4);
	if (ver != 1 || 12 + (size_t)n * sizeof(vfs_ent) > len) return -2;
	blob_p = b; blob_n = n; ents = (const vfs_ent *)(b + 12);
	return 0;
}

const void *ito_vfs_find(const char *path, size_t *len)
{
	if (!blob_p || !path) return NULL;
	while (*path == '/') path++;
	/* "lang/gmw/en-US" and "voices/..." keep their directories; "phontab" etc. have none. Names are stored exactly so. */
	for (uint32_t i = 0; i < blob_n; i++) {
		if (strncmp(ents[i].name, path, sizeof(ents[i].name)) == 0 && strlen(path) < sizeof(ents[i].name)) {
			if (len) *len = ents[i].len;
			return blob_p + ents[i].off;
		}
	}
	return NULL;
}
