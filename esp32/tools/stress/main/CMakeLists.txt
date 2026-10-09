idf_component_register(SRCS "stress.c" "dot_rows_s3.S" INCLUDE_DIRS "." PRIV_REQUIRES esp_timer esp_psram heap esp_system esp_hw_support)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function)
