idf_component_register(SRCS "main.c" "kernels_s3.c" "dot_rows_s3.S" "../../engine/itofs.c"
                       INCLUDE_DIRS "." "../../engine"
                       PRIV_REQUIRES esp_timer esp_driver_i2s esp_driver_uart esp_driver_gpio esp_partition
                                     esp_psram heap esp_system spi_flash esp_mm esp_hw_support)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function -Wno-format-truncation)
# the engine's element-wise loops are bound by address arithmetic (GCC's Xtensa output has no auto-increment addressing): unrolling
# them cuts instructions without touching the order of any floating-point operation
target_compile_options(${COMPONENT_LIB} PRIVATE -funroll-loops)
if(ITOFS_QEMU)
  target_compile_definitions(${COMPONENT_LIB} PRIVATE ITOFS_QEMU=1)
endif()
if(ITOFS_EXTRA_DEFS)      # e.g. -D ITOFS_EXTRA_DEFS="STEADY_FRAMES=16;WIDE_ROWS=12": experiments with the chunk size / wide-layer batch
  target_compile_definitions(${COMPONENT_LIB} PRIVATE ${ITOFS_EXTRA_DEFS})
endif()
if(ITOFS_ICPROF)
  target_compile_definitions(${COMPONENT_LIB} PRIVATE ITOFS_ICPROF=1 ITOFS_PROF=1)
endif()
