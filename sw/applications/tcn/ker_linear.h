#pragma once
#ifndef KER_LINEAR_H_
#define KER_LINEAR_H_

#include <stdint.h>
#include <stddef.h>

#include "ker_xpulp.h"

#include "build_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration

#ifndef FC_OUT_TILE
#define FC_OUT_TILE 64  // tile size over output channels
#endif

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif


#ifndef XHEEP_FLASH_READ_FN_TYPEDEF
#define XHEEP_FLASH_READ_FN_TYPEDEF
typedef int (*flash_read_fn)(uint32_t offset, void *dst, size_t nbytes);
#endif

// Weights layout: W:[Cout, Cin] row-major by output channel
typedef struct {
  uint32_t off_w;  // int8  weights, size = Cout * Cin
  uint32_t off_b;  // int32 bias   , size = Cout
  uint32_t off_m;  // int32 mul    , size = Cout
  uint32_t off_r;  // int32 rshift , size = Cout (can be >=0 or <0)
  int Cin;
  int Cout;
} fc_flash_desc_t;


// Core dot for one row X[n,:] against a tile of rows of W.
// x: [Cin], w_tile: [oN, Cin] (row-major by oc), acc: [oN]
static inline void fc_core_outtile_one_row(
    const int8_t *__restrict x, int Cin,
    const int8_t *__restrict w_tile, int oN,
    const int32_t *__restrict b_tile,   // [oN]
    int32_t *__restrict acc)            // [oN]
{
  // Initialize with bias (or zero)
  if (b_tile) {
    for (int o = 0; o < oN; ++o) {
      acc[o] = b_tile[o];
    }
  } else {
    for (int o = 0; o < oN; ++o) {
      acc[o] = 0;
    }
  }

#if XHEEP_USE_XPULP_DOT4
  // Packed dot4 path (CORE-V XCVSIMD)
  int i = 0;
  for (; i + 4 <= Cin; i += 4) {
    const uint32_t xp = xheep_load_u32_unaligned((const void *)(x + i));
    for (int o = 0; o < oN; ++o) {
      const int8_t *wrow = w_tile + (size_t)o * (size_t)Cin + (size_t)i;
      const uint32_t wp = xheep_load_u32_unaligned((const void *)wrow);
      acc[o] += dot4_i8_xpulp_packed_l(xp, wp);
    }
  }
  // Remainder
  for (; i < Cin; ++i) {
    const int32_t xv = (int32_t)x[i];
    for (int o = 0; o < oN; ++o) {
      acc[o] += xv * (int32_t)w_tile[(size_t)o * (size_t)Cin + (size_t)i];
    }
  }
  return;
#endif

  // Main accumulation: Cin outer, oN inner
  for (int i = 0; i < Cin; ++i) {
    const int32_t xv = (int32_t)x[i];
    const int8_t *__restrict wptr = w_tile + i;  // start at weight[0, i]

    int o = 0;

    // Unroll by 4 over output channels when possible
    for (; o + 3 < oN; o += 4) {
      // wptr points at [o, i], stride Cin between outputs
      const int32_t w0 = (int32_t)wptr[0 * Cin];
      const int32_t w1 = (int32_t)wptr[1 * Cin];
      const int32_t w2 = (int32_t)wptr[2 * Cin];
      const int32_t w3 = (int32_t)wptr[3 * Cin];

      acc[o + 0] += xv * w0;
      acc[o + 1] += xv * w1;
      acc[o + 2] += xv * w2;
      acc[o + 3] += xv * w3;

      wptr += 4 * Cin;
    }

    // Remainder (if oN is not multiple of 4)
    for (; o < oN; ++o) {
      const int32_t wv = (int32_t)*wptr;
      acc[o] += xv * wv;
      wptr += Cin;
    }
  }
}

// Compute pre-shift R1 for 32-bit requant from Cin.
#define REQUANT32_M_SHIFT 15

static inline int fc_requant_R1(int Cin)
{
  // |acc| <= Cin * 127 * 127 = Cin * 16129
  unsigned v = (unsigned)Cin * 16129u;
  int bl = 0;
  while (v) { v >>= 1; bl++; }
  int acc_bits = bl + 1; // +1 for sign
  int r1 = acc_bits + 16 - 31;
  return r1 > 0 ? r1 : 0;
}

// Per-channel requant
static inline void fc_requant_outtile(
    const int32_t *__restrict acc,       // [oN]
    const int32_t *__restrict m_tile,    // [oN] M_small (16-bit range)
    const int32_t *__restrict r_tile,    // [oN] R2 (post-shift, always > 0)
    int oN,
    int Cin,
    int8_t *__restrict y_row_seg)        // [oN]
{
  const int R1 = fc_requant_R1(Cin);
  const int32_t rnd1 = R1 > 0 ? ((int32_t)1 << (R1 - 1)) : 0;

  for (int o = 0; o < oN; ++o) {
    // Step 1: pre-shift accumulator to reduce bit-width
    const int32_t a = (acc[o] + rnd1) >> R1;

    // Step 2: multiply
    const int32_t prod = a * m_tile[o];

    // Step 3: post-shift with rounding
    const int32_t r2 = r_tile[o];
    const int32_t rnd2 = (int32_t)1 << (r2 - 1);
    const int32_t rq = (prod + rnd2) >> r2;

    y_row_seg[o] = SATURATE_INT8(rq);
  }
}

// ----------- Batched FC: multiple tokens, same weight tile -----------
// Reuses each weight load across ntok tokens, reducing memory traffic.

#ifndef FC_TOKEN_BATCH
#define FC_TOKEN_BATCH 4
#endif

// x_batch:   first token pointer; successive tokens at stride x_stride bytes
// acc_flat:  [ntok * FC_OUT_TILE] int32, row stride = FC_OUT_TILE
static inline void fc_core_outtile_batch(
    const int8_t *__restrict x_batch, int Cin, int x_stride, int ntok,
    const int8_t *__restrict w_tile, int oN,
    const int32_t *__restrict b_tile,
    int32_t *__restrict acc_flat)
{
  for (int n = 0; n < ntok; ++n) {
    int32_t *a = acc_flat + n * FC_OUT_TILE;
    if (b_tile)
      for (int o = 0; o < oN; ++o) a[o] = b_tile[o];
    else
      for (int o = 0; o < oN; ++o) a[o] = 0;
  }

#if XHEEP_USE_XPULP_DOT4
  {
    int i = 0;
    for (; i + 4 <= Cin; i += 4) {
      uint32_t xp[FC_TOKEN_BATCH];
      for (int n = 0; n < ntok; ++n)
        xp[n] = xheep_load_u32_unaligned(x_batch + (size_t)n * (size_t)x_stride + i);

      for (int o = 0; o < oN; ++o) {
        const uint32_t wp = xheep_load_u32_unaligned(w_tile + (size_t)o * (size_t)Cin + i);
        for (int n = 0; n < ntok; ++n)
          acc_flat[n * FC_OUT_TILE + o] += dot4_i8_xpulp_packed_l(xp[n], wp);
      }
    }
    // Remainder (Cin not multiple of 4)
    for (; i < Cin; ++i) {
      int32_t xv[FC_TOKEN_BATCH];
      for (int n = 0; n < ntok; ++n)
        xv[n] = (int32_t)x_batch[(size_t)n * (size_t)x_stride + i];
      for (int o = 0; o < oN; ++o) {
        const int32_t wv = (int32_t)w_tile[(size_t)o * (size_t)Cin + i];
        for (int n = 0; n < ntok; ++n)
          acc_flat[n * FC_OUT_TILE + o] += xv[n] * wv;
      }
    }
    return;
  }
#endif

  // Scalar: Cin outer, oN middle (4-way unrolled), ntok inner — shares weight across tokens
  for (int i = 0; i < Cin; ++i) {
    int32_t xv[FC_TOKEN_BATCH];
    for (int n = 0; n < ntok; ++n)
      xv[n] = (int32_t)x_batch[(size_t)n * (size_t)x_stride + i];

    const int8_t *__restrict wptr = w_tile + i;
    int o = 0;
    for (; o + 3 < oN; o += 4) {
      const int32_t w0 = (int32_t)wptr[0];
      const int32_t w1 = (int32_t)wptr[(size_t)Cin];
      const int32_t w2 = (int32_t)wptr[(size_t)2 * (size_t)Cin];
      const int32_t w3 = (int32_t)wptr[(size_t)3 * (size_t)Cin];
      for (int n = 0; n < ntok; ++n) {
        acc_flat[n * FC_OUT_TILE + o + 0] += xv[n] * w0;
        acc_flat[n * FC_OUT_TILE + o + 1] += xv[n] * w1;
        acc_flat[n * FC_OUT_TILE + o + 2] += xv[n] * w2;
        acc_flat[n * FC_OUT_TILE + o + 3] += xv[n] * w3;
      }
      wptr += (size_t)4 * (size_t)Cin;
    }
    for (; o < oN; ++o) {
      const int32_t wv = (int32_t)*wptr;
      for (int n = 0; n < ntok; ++n) acc_flat[n * FC_OUT_TILE + o] += xv[n] * wv;
      wptr += Cin;
    }
  }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KER_LINEAR_H_