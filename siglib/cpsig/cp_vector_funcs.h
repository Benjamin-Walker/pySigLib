/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include "cppch.h"
#include "macros.h"

#ifdef VEC
#ifndef __APPLE__
FORCE_INLINE void vec_mult_add(double* out, const double* other, double scalar, uint64_t size)
{
	const uint64_t N = size / 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256d a, b;
	const __m256d scalar_256 = _mm256_set1_pd(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_pd(other);
		a = _mm256_mul_pd(a, scalar_256);
		b = _mm256_loadu_pd(out);
		b = _mm256_add_pd(a, b);
		_mm256_storeu_pd(out, b);
		other += 4;
		out += 4;
	}
	if (tail2) {
		__m128d c, d;
		__m128d scalar_128 = _mm_set1_pd(scalar);

		c = _mm_loadu_pd(other);
		c = _mm_mul_pd(c, scalar_128);
		d = _mm_loadu_pd(out);
		d = _mm_add_pd(c, d);
		_mm_storeu_pd(out, d);
		other += 2;
		out += 2;
	}
	if (tail1) {
		*out += *other * scalar;
	}
}

FORCE_INLINE void vec_mult_assign(double* out, const double* other, double scalar, uint64_t size) 
{
	const uint64_t N = size / 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256d a;
	const __m256d scalar_ = _mm256_set1_pd(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_pd(other);
		a = _mm256_mul_pd(a, scalar_);
		_mm256_storeu_pd(out, a);
		other += 4;
		out += 4;
	}
	if (tail2) {
		__m128d c;
		__m128d scalar_128 = _mm_set1_pd(scalar);

		c = _mm_loadu_pd(other);
		c = _mm_mul_pd(c, scalar_128);
		_mm_storeu_pd(out, c);
		other += 2;
		out += 2;
	}
	if (tail1) {
		*out = *other * scalar;
	}
}

FORCE_INLINE double dot_product(const double* a, const double* b, size_t N) {
	__m256d sum = _mm256_setzero_pd();

	size_t k = 0;
	size_t limit = N & ~3UL;
	for (; k < limit; k += 4) {
		__m256d va = _mm256_loadu_pd(&a[k]);
		__m256d vb = _mm256_loadu_pd(&b[k]);
		sum = _mm256_fmadd_pd(va, vb, sum);
	}

	double tmp[4];
	_mm256_storeu_pd(tmp, sum);
	double out = tmp[0] + tmp[1] + tmp[2] + tmp[3];

	for (; k < N; ++k) {
		out += a[k] * b[k];
	}

	return out;
}

#else
FORCE_INLINE void vec_mult_add(double* out, const double* other, double scalar, uint64_t size) {
    const uint64_t N = size / 2;
    const uint64_t tail = size & 1;

    float64x2_t scalar_v = vdupq_n_f64(scalar);

    for (uint64_t i = 0; i < N; ++i) {
        float64x2_t a = vld1q_f64(other);
        float64x2_t b = vld1q_f64(out);

        a = vmulq_f64(a, scalar_v);
        b = vaddq_f64(b, a);

        vst1q_f64(out, b);

        other += 2;
        out += 2;
    }
    if (tail) {
        *out += (*other) * scalar;
    }
}

FORCE_INLINE void vec_mult_assign(double* out, const double* other, double scalar, uint64_t size) {
    const uint64_t N = size / 2;
    const uint64_t tail = size & 1;

    float64x2_t scalar_v = vdupq_n_f64(scalar);

    for (uint64_t i = 0; i < N; ++i) {
        float64x2_t a = vld1q_f64(other);
        a = vmulq_f64(a, scalar_v);
        vst1q_f64(out, a);

        other += 2;
        out += 2;
    }
    if (tail) {
        *out = (*other) * scalar;
    }
}


#endif

#endif
