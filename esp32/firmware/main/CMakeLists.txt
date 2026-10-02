idf_component_register(SRCS "main.c" "kernels_s3.c" "../../engine/itofs.c"
                       INCLUDE_DIRS "." "../../engine"
                       PRIV_REQUIRES esp_timer esp_driver_i2s esp_driver_uart esp_driver_gpio esp_partition
                                     esp_psram heap esp_system spi_flash esp_mm esp_hw_support)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function -Wno-format-truncation)
if(ITOFS_QEMU)
  target_compile_definitions(${COMPONENT_LIB} PRIVATE ITOFS_QEMU=1)
endif()
