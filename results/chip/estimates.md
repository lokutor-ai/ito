**SUPERSEDED (2 October 2026).** This table is the first, op-count-only estimate: it counted int8 MACs at an assumed GOPS and ignored PSRAM traffic and non-MAC instructions, so it was too optimistic (it is for the first release engine, 100 ms first chunk). The current estimates, from exact QEMU instruction counts plus a PSRAM-traffic model, are in esp32/README.md section 4: time to first audio 124-127 / 176-179 / 266-269 ms and real-time factor 0.72-0.78 / 1.15-1.23 / 1.9-2.0 (optimistic / central / pessimistic). Not measured on silicon; real time is not established.

forward-GRU front, fixed style (incremental text side), fitted on 3 golden sentences:
  int8 MACs before the first chunk = 83.6 M + -0.000 M/token; non-GEMM ~10.7 M + -0.000 M cycles/token; weights read 10.2-10.2 MB
  per second of audio: 346.2 M int8 MACs, 1.16 M f32 MACs, ~55.6 M non-GEMM cycles (0.23 core), 39.3 MB weights

TTFA (first 100 ms chunk ready), ms = int8 / G + non-GEMM / 240 MHz:
| tokens | int8 ops | non-GEMM | G = 0.3 GOPS | G = 0.5 GOPS | G = 1.0 GOPS | G = 2.0 GOPS | G needed for < 200 ms |
|---|---|---|---|---|---|---|---|
| 25 (short) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
| 87 (val p25) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
| 110 (val median) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
| 132 (val p75) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
| 165 (val p95) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
| 182 (val max, long) | 84 M | 45 ms | 323 | 212 | 128 | 86 | 0.54 GOPS |
  the work before the first chunk is the same for any sentence length

RTF = int8 per second / G + non-GEMM per second / 240 MHz:
| | G = 0.3 | G = 0.5 | G = 1.0 | G = 2.0 | G needed for RTF < 1 |
|---|---|---|---|---|---|
| RTF | 1.39 | 0.92 | 0.58 | 0.40 | 0.45 GOPS |

PSRAM: the GEMMs read 39.3 MB of weights per second of audio, i.e. 8.8 MACs per weight byte; G is capped at 8.8 x (effective PSRAM MB/s) / 1000 GOPS (40 MB/s -> 0.35, 80 MB/s -> 0.70) unless the copy overlaps compute (wmode gdma).
