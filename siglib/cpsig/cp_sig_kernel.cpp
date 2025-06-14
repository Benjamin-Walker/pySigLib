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

#include "cppch.h"
#include "cpsig.h"
#include "cp_sig_kernel.h"
#include "macros.h"

void get_sig_kernel_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2
) {
	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;

	// Dyadically refined grid dimensions
	const uint64_t grid_size_1 = 1ULL << dyadic_order_1;
	const uint64_t grid_size_2 = 1ULL << dyadic_order_2;
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	// Allocate(flattened) PDE grid
	double* pde_grid = (double*)malloc(dyadic_length_1 * dyadic_length_2 * sizeof(double));

	// Initialization of K array
	for (uint64_t i = 0; i < dyadic_length_1; ++i) {
		pde_grid[i * dyadic_length_2] = 1.0; // Set K[i, 0] = 1.0
	}

	std::fill(pde_grid, pde_grid + dyadic_length_2, 1.0); // Set K[0, j] = 1.0

	double* deriv_term_1 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 + 0.5 * deriv + deriv2
	if (!deriv_term_1) {
		throw std::bad_alloc();
		return;
	}
	double* deriv_term_2 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 - deriv2
	if (!deriv_term_2) {
		throw std::bad_alloc();
		return;
	}

	double* k11 = pde_grid;
	double* k12 = k11 + 1;
	double* k21 = k11 + dyadic_length_2;
	double* k22 = k21 + 1;

	for (uint64_t ii = 0; ii < length1 - 1; ++ii) {
		for (uint64_t m = 0; m < length2 - 1; ++m) {
			double deriv = gram[ii * (length2 - 1) + m];//dot_product(diff1Ptr, diff2Ptr, dimension);
			deriv *= dyadic_frac;
			double deriv2 = deriv * deriv * twelth;
			deriv_term_1[m] = 1.0 + 0.5 * deriv + deriv2;
			deriv_term_2[m] = 1.0 - deriv2;
		}

		for (uint64_t i = 0; i < grid_size_1; ++i) {
			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				double t1 = deriv_term_1[jj];
				double t2 = deriv_term_2[jj];
				for (uint64_t j = 0; j < grid_size_2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * t1 - *(k11++) * t2;
				}
			}
			++k11;
			++k12;
			++k21;
			++k22;
		}
	}

	*out = pde_grid[dyadic_length_1 * dyadic_length_2 - 1];
	free(deriv_term_1);
	free(deriv_term_2);
	free(pde_grid);
}

void sig_kernel_(
	double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	get_sig_kernel_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2);
}

void batch_sig_kernel_(
	double* gram,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool parallel
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	double* const data_end_1 = gram + gram_length * batch_size;

	std::function<void(double*, double*)> sig_kernel_func;

	sig_kernel_func = [&](double* gram_ptr, double* out_ptr) {
		get_sig_kernel_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2);
		};

	if (parallel) {
		multi_threaded_batch(sig_kernel_func, gram, out, batch_size, gram_length, 1);
	}
	else {
		double* gram_ptr = gram;
		double* out_ptr = out;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, ++out_ptr) {

			sig_kernel_func(gram_ptr, out_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int sig_kernel(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept {
		SAFE_CALL(sig_kernel_(gram, out, dimension, length1, length2, dyadic_order_1, dyadic_order_2));
	}

	CPSIG_API int batch_sig_kernel(double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool parallel) noexcept {
		SAFE_CALL(batch_sig_kernel_(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, parallel));
	}
}
