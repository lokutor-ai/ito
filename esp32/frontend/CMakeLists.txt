# ESP-IDF component: the on-chip text front end (see FRONTEND.md). The espeak-ng data (273 KB) is embedded in the app image (flash, read through the cache).
file(GLOB ES_SRCS espeak/src/*.c espeak/ucd/*.c)
idf_component_register(SRCS ito_frontend.c ito_fe_idf.c ${ES_SRCS}
                       INCLUDE_DIRS . espeak/src
                       PRIV_INCLUDE_DIRS espeak/include espeak/include/compat-idf espeak/ucd/include
                       PRIV_REQUIRES esp_timer heap
                       EMBED_FILES data/espeak_en_us.bin
                       LDFRAGMENTS linker.lf)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -w)
# espeak-ng: every malloc/calloc/realloc/free/strdup goes to PSRAM
foreach(f ${ES_SRCS})
  set_source_files_properties(${f} PROPERTIES COMPILE_OPTIONS "-include;${CMAKE_CURRENT_LIST_DIR}/ito_mem.h")
endforeach()
