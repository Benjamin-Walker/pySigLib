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

#include "multithreading.h"

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

void get_sig_kernel_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2,
	bool return_grid
);

template<bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_diag_internal_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2,
	const uint64_t dyadic_length_1,
	const uint64_t dyadic_length_2
) {
	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<double[]>(diag_len * 3);
	double* diagonals = diagonals_uptr.get();

	double* prev_prev_diag = diagonals;
	double* prev_diag = diagonals + diag_len;
	double* next_diag = diagonals + 2 * diag_len;

	// Initialization
	std::fill(diagonals, diagonals + 3 * diag_len, 1.);

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (dyadic_length_2 > p) endj = p;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_1);
				uint64_t jj = ((j - 1) >> dyadic_order_2);

				double deriv = gram[ii * (length2 - 1) + jj];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					1. + 0.5 * deriv + deriv2) - *(prev_prev_diag + j - 1) * (1. - deriv2);

			}
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (dyadic_length_1 > p) endj = p;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_2);
				uint64_t jj = ((j - 1) >> dyadic_order_1);

				double deriv = gram[jj * (length2 - 1) + ii];

				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					1. + 0.5 * deriv + deriv2) - *(prev_prev_diag + j - 1) * (1. - deriv2);

			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		double* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}

	*out = prev_diag[diag_len - 1];
}

void get_sig_kernel_diag_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2
);

void sig_kernel_(
	double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid = false
);

void batch_sig_kernel_(
	double* gram,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs = 1,
	bool return_grid = false
);