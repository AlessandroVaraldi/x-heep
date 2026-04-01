#pragma once
#ifndef KER_XPULP_SIMD_H_
#define KER_XPULP_SIMD_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef XHEEP_HAS_XCVSIMD
#if defined(__riscv) && defined(__riscv_xcvsimd)
#define XHEEP_HAS_XCVSIMD 1
#else
#define XHEEP_HAS_XCVSIMD 0
#endif
#endif

#ifndef XHEEP_USE_XPULP_DOT4
#define XHEEP_USE_XPULP_DOT4 XHEEP_HAS_XCVSIMD
#endif

static inline uint32_t xheep_load_u32_unaligned(const void *p)
{
    const uintptr_t addr = (uintptr_t)p;
    if ((addr & 3u) == 0u) {
        return *(const uint32_t *)p;
    }
    uint32_t v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

#if XHEEP_USE_XPULP_DOT4
static inline int32_t dot4_i8_xpulp_packed_l(uint32_t a_packed, uint32_t b_packed)
{
    int32_t res;
    __asm__ volatile(
        "cv.dotsp.b %0, %1, %2\n"
        : "=r"(res)
        : "r"(a_packed), "r"(b_packed));
    return res;
}
#endif

static inline int32_t xheep_dot_i8_i8_any(const int8_t *__restrict a, const int8_t *__restrict b, int n)
{
    int32_t s = 0;
    int i = 0;

#if XHEEP_USE_XPULP_DOT4
    for (; i + 4 <= n; i += 4) {
        const uint32_t ap = xheep_load_u32_unaligned((const void *)(a + i));
        const uint32_t bp = xheep_load_u32_unaligned((const void *)(b + i));
        s += dot4_i8_xpulp_packed_l(ap, bp);
    }
#endif

    for (; i < n; ++i)  s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KER_XPULP_SIMD_H_
