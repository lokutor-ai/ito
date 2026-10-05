// itofs_s3_dot_rows: one int8 weight row against several int8 activation rows on the ESP32-S3 PIE, exact int32.
//   acc[r * stride] = sum_{i < 16 * len16} x[r * ldx + i] * w[i]      for r = 0 .. rows-1
// Both x (every row) and w must be 16-byte aligned; 16 MACs per fused multiply-accumulate + load, 2 instructions per
// 16 MACs in the inner loop (esp-nn's unaligned dot needs 4 plus a call per row). The sum is exact (ACCX is 40 bits and
// |sum| < 2^31 for the engine's widths), so the result is bit-identical to any other exact int8 GEMM.
//   a2 x, a3 ldx (bytes), a4 w, a5 (rows << 16) | len16 (len16 >= 1), a6 acc, a7 stride (bytes)
    .text
    .align  4
    .global itofs_s3_dot_rows
    .type   itofs_s3_dot_rows, @function
itofs_s3_dot_rows:
    entry   a1, 32
    extui   a8, a5, 0, 16           // len16
    srli    a9, a5, 16              // rows
    beqz    a9, .Ldr_done
    addi    a10, a8, -1             // fused iterations
.Ldr_row:
    mov     a11, a2
    mov     a12, a4
    ee.zero.accx
    ee.vld.128.ip   q0, a11, 16
    ee.vld.128.ip   q1, a12, 16
    loopgtz a10, .Ldr_inner_end
    ee.vmulas.s8.accx.ld.ip  q0, a11, 16, q0, q1
    ee.vld.128.ip   q1, a12, 16
.Ldr_inner_end:
    ee.vmulas.s8.accx  q0, q1
    add     a2, a2, a3              // (also the 2-instruction gap before reading ACCX)
    addi    a9, a9, -1
    rur.accx_0  a13
    s32i    a13, a6, 0
    add     a6, a6, a7
    bnez    a9, .Ldr_row
.Ldr_done:
    retw.n
    .size   itofs_s3_dot_rows, . - itofs_s3_dot_rows

// itofs_s3_pack_row: copy nv16 * 16 bytes starting at ANY address into a 16-byte aligned buffer (PIE funnel shift: 3 instructions per 16
// bytes). Reads up to 16 bytes past src + nv16 * 16 - 1 (one extra aligned block when src is misaligned).
//   a2 dst (16-byte aligned), a3 src, a4 nv16 (>= 1)
    .align  4
    .global itofs_s3_pack_row
    .type   itofs_s3_pack_row, @function
itofs_s3_pack_row:
    entry   a1, 32
    ee.ld.128.usar.ip   q0, a3, 16      // prime: aligned block holding src[0]; sets SAR_BYTE = src & 15
    loopgtz a4, .Lpr_end
    ee.ld.128.usar.ip   q2, a3, 16
    ee.src.q.qup        q4, q0, q2      // q4 = bytes [src&15, src&15 + 16) of the block pair; q0 <- q2
    ee.vst.128.ip       q4, a2, 16
.Lpr_end:
    retw.n
    .size   itofs_s3_pack_row, . - itofs_s3_pack_row

// itofs_s3_w4_unpack: one int4 weight row -> int8 (see engine/itofs.h for the format). 18 instructions per group of 32 weights.
//   a2 dst (16-byte aligned, ng * 32 bytes written), a3 row (16-byte aligned: ng * 16 bytes of nibbles, then ng parameter bytes (m << 4) | zp), a4 ng (>= 1)
// Per group (all lanes are plain byte lanes):  lo = v & 0x0F,  hi16 = v & 0xF0,  P = broadcast of the parameter byte,
//   m = ((P & 0xF0) * 1) >> 4,  zp = P & 0x0F,   hi4*m = (hi16 * m) >> 4,   lo4*m = lo * m,   zp*m = zp * m   (unsigned 8-bit multiplies; the SAR shift is applied to the
//   16-bit product, every result is <= 105 so nothing is truncated),   w8 = q*m - zp*m  (signed saturating subtract; |w8| <= 105, nothing saturates).
    .align  4
    .global itofs_s3_w4_unpack
    .type   itofs_s3_w4_unpack, @function
itofs_s3_w4_unpack:
    entry   a1, 32
    movi    a6, itofs_w4_consts
    ee.vld.128.ip   q6, a6, 16          // 0x0F
    ee.vld.128.ip   q7, a6, 16          // 0xF0
    ee.vld.128.ip   q5, a6, 0           // 0x01
    slli    a7, a4, 4
    add     a5, a3, a7                  // parameter bytes
    loopgtz a4, .Lw4_end
    ee.vld.128.ip   q0, a3, 16
    ee.vldbc.8      q3, a5
    addi    a5, a5, 1
    ee.andq         q1, q0, q6
    ee.andq         q2, q0, q7
    ee.andq         q4, q3, q7
    ee.andq         q3, q3, q6
    ssai            4
    ee.vmul.u8      q4, q4, q5
    ee.vmul.u8      q2, q2, q4
    ssai            0
    ee.vmul.u8      q1, q1, q4
    ee.vmul.u8      q3, q3, q4
    ee.vsubs.s8     q1, q1, q3
    ee.vsubs.s8     q2, q2, q3
    ee.vst.128.ip   q1, a2, 16
    ee.vst.128.ip   q2, a2, 16
.Lw4_end:
    retw.n
    .size   itofs_s3_w4_unpack, . - itofs_s3_w4_unpack

    .data
    .align  16
    .global itofs_w4_consts
itofs_w4_consts:
    .fill   16, 1, 0x0F
    .fill   16, 1, 0xF0
    .fill   16, 1, 0x01
