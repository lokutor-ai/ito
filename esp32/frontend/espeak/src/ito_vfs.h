/* ITO: read-only virtual file system over one memory blob (replaces the espeak-ng data directory).
 * Blob layout (little endian): "ITFE" u32 version u32 n_entries, then n_entries x { char name[28]; u32 offset; u32 length },
 * then the file contents (each 4-byte aligned, offsets from the start of the blob). */
#ifndef ITO_VFS_H
#define ITO_VFS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int ito_vfs_set_blob(const void *blob, size_t len);                 /* 0 = ok */
const void *ito_vfs_find(const char *path, size_t *len);            /* leading '/' and directory prefixes up to the data dir are ignored */
#ifdef __cplusplus
}
#endif
#endif
