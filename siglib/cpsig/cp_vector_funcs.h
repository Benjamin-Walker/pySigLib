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

#ifdef AVX
FORCE_INLINE void vec_mult_add(double* out, double* other, double scalar, uint64_t size)
{
	uint64_t first_loop_remainder = size % 4UL;

	__m256d a, b;
	__m256d scalar_256 = _mm256_set1_pd(scalar);

	__m128d c, d;
	__m128d scalar_128 = _mm_set1_pd(scalar);

	double* other_ptr = other, * out_ptr = out;
	double* out_end = out + size;

	double* first_loop_end = out_end - first_loop_remainder;

	for (; out_ptr != first_loop_end; other_ptr += 4, out_ptr += 4) {
		a = _mm256_loadu_pd(other_ptr);
		a = _mm256_mul_pd(a, scalar_256);
		b = _mm256_loadu_pd(out_ptr);
		b = _mm256_add_pd(a, b);
		_mm256_storeu_pd(out_ptr, b);
	}
	if (size & 2UL) {
		c = _mm_loadu_pd(other_ptr);
		c = _mm_mul_pd(c, scalar_128);
		d = _mm_load_pd(out_ptr);
		d = _mm_add_pd(c, d);
		_mm_storeu_pd(out_ptr, d);
		other_ptr += 2;
		out_ptr += 2;
	}
	if (size & 1UL) { //For some reason intrinsics are quicker than a normal loop here
		c = _mm_load_sd(other_ptr);
		c = _mm_mul_sd(c, scalar_128);
		d = _mm_load_sd(out_ptr);
		d = _mm_add_sd(c, d);
		_mm_store_sd(out_ptr, d);
	}
}

FORCE_INLINE void vec_mult_assign(double* out, double* other, double scalar, uint64_t size) {
	uint64_t first_loop_remainder = size % 4UL;

	__m256d a;
	__m256d scalar_ = _mm256_set1_pd(scalar);

	__m128d c;
	__m128d scalar_128 = _mm_set1_pd(scalar);

	double* other_ptr = other, * out_ptr = out;
	double* out_end = out + size;

	double* first_loop_end = out_end - first_loop_remainder;

	for (; out_ptr != first_loop_end; other_ptr += 4, out_ptr += 4) {
		a = _mm256_loadu_pd(other_ptr);
		a = _mm256_mul_pd(a, scalar_);
		_mm256_storeu_pd(out_ptr, a);
	}
	if(size & 2UL) {
		c = _mm_loadu_pd(other_ptr);
		c = _mm_mul_pd(c, scalar_128);
		_mm_storeu_pd(out_ptr, c);
		other_ptr += 2;
		out_ptr += 2;
	}
	if (size & 1UL) { //For some reason intrinsics are quicker than a normal loop here
		c = _mm_load_sd(other_ptr);
		c = _mm_mul_sd(c, scalar_128);
		_mm_store_sd(out_ptr, c);
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

#endif
