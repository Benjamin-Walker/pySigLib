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

FORCE_INLINE void get_a_b(double& a, double& b, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	const double gram_val_2 = gram_val * gram_val * twelth;
	a = 1. + 0.5 * gram_val + gram_val_2;
	b = 1. - gram_val_2;
}

FORCE_INLINE void get_a(double& a, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	double gram_val = gram[idx] * dyadic_frac;
	a = 1. + gram_val * (0.5 + gram_val * twelth);
}

FORCE_INLINE void get_b(double& b, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	b = 1. - gram_val * gram_val * twelth;
}

FORCE_INLINE void get_a_b_deriv(double& a_deriv, double& b_deriv, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	static const double sixth = 1. / 6;
	const double gram_val = gram[idx] * dyadic_frac;
	b_deriv = -gram_val * sixth * dyadic_frac;
	a_deriv = 0.5 * dyadic_frac - b_deriv;
}

void get_sig_kernel_(
	const double* gram,
	uint64_t length1,
	uint64_t length2,
	double* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
);

template<bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_diag_internal_(
	const double* gram,
	uint64_t length2,
	double* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	uint64_t dyadic_length_1,
	uint64_t dyadic_length_2
) {
	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<double[]>(diag_len * 3);
	double* const diagonals = diagonals_uptr.get();

	double* prev_prev_diag = diagonals;
	double* prev_diag = diagonals + diag_len;
	double* next_diag = diagonals + 2 * diag_len;

	// Initialization
	std::fill(diagonals, diagonals + 3 * diag_len, 1.);

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1;
			else startj = p - dyadic_length_1 + 1;
			if (dyadic_length_2 > p) endj = p;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t ii = ((i - 1) >> dyadic_order_1);
				const uint64_t jj = ((j - 1) >> dyadic_order_2);

				const double deriv = gram[ii * (length2 - 1) + jj] * dyadic_frac;
				const double deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					1. + 0.5 * deriv + deriv2) - *(prev_prev_diag + j - 1) * (1. - deriv2);

			}
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1;
			else startj = p - dyadic_length_2 + 1;
			if (dyadic_length_1 > p) endj = p;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t ii = ((i - 1) >> dyadic_order_2);
				const uint64_t jj = ((j - 1) >> dyadic_order_1);

				const double deriv = gram[jj * (length2 - 1) + ii] * dyadic_frac;
				const double deriv2 = deriv * deriv * twelth;

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

template<bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_backprop_diag_internal_(
	const double* gram,
	double* out,
	double deriv,
	const double* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	uint64_t dyadic_length_1,
	uint64_t dyadic_length_2
) {
	// General structure of the grids:
	// 
	// dF / dk = 0 for the first row and column of k_grid, so disregard these.
	// Flip the remaining grid, so that the last element is now in the top left.
	// Now, add a row and column of zeros as initial conditions to the grid, such that it now
	// has the same dimensions as k_grid.
	// The resulting grid is what is traversed by 'diagonals' below.
	// 
	// The grids for A, B, dA and dB are flipped and padded similarly, such that
	// the value at index [1,1] is the value at [-1,-1] in the original grids.
	// We will only need one diagonal for A and one for B, containing the values
	// needed to update the leading diagonal of dF / dk. For dA and dB, we don't
	// need to use diagonals, we can just get the values once when updating dF / dk.
	// Note that for A, these values are lagged, i.e. we need values A(i-1,j) and 
	// A(i,j-1) to update dF / dk(i,j).

	// As with the diagonal method for sig_kernel, it matters which of
	// dyadic_length_1 and dyadic_length_2 is longer.
	const uint64_t ord_dyadic_order_1 = order ? dyadic_order_1 : dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? dyadic_order_2 : dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? dyadic_length_2 : dyadic_length_1;

	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<double[]>(diag_len * 3);
	double* const diagonals = diagonals_uptr.get();

	// Allocate diagonals to store A, B, A_deriv, B_deriv
	auto a_uptr = std::make_unique<double[]>(diag_len);
	double* const a = a_uptr.get();

	auto b_uptr = std::make_unique<double[]>(diag_len);
	double* const b = b_uptr.get();

	// Ptrs for diagonals
	double* prev_prev_diag = diagonals;
	double* prev_diag = prev_prev_diag + diag_len;
	double* next_diag = prev_diag + diag_len;

	// k_grid ptrs
	const double* k11, * k12, * k21;

	// Initialization
	std::fill(out, out + (length1 - 1) * (length2 - 1), 0.);
	std::fill(diagonals, diagonals + 3 * diag_len, 0.);
	std::fill(a, a + diag_len, 0.);
	std::fill(b, b + diag_len, 0.);
	
	*(prev_diag + 1) = deriv;
	double da, db;
	get_a_b_deriv(da, db, gram, gram_length - 1, dyadic_frac);

	//Update dF / dx for first value
	k21 = k_grid + grid_length - 2;
	k12 = k_grid + grid_length - dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
	k11 = k12 - 1;
	out[gram_length - 1] += deriv * ( ((*k21) + (*k12)) * da - *(k11) * db );

	for (uint64_t p = 3; p < num_anti_diag; ++p) { // First three antidiagonals are initialised

		//Update b
		uint64_t startj, endj;
		uint64_t p_ = p - 2;
		startj = ord_dyadic_length_1 > p_ ? 1 : p_ - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p_ ? p_ : ord_dyadic_length_2;

		uint64_t i = p_ - startj; // Calculate corresponding i (since i + j = p)
		uint64_t i_rev = ord_dyadic_length_1 - i - 1;
		uint64_t j_rev = ord_dyadic_length_2 - startj - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_b(b[j], gram, gram_idx, dyadic_frac);
		}

		//Update a
		p_ = p - 1;
		startj = ord_dyadic_length_1 > p_ ? 1 : p_ - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p_ ? p_ : ord_dyadic_length_2;

		i = p_ - startj; // Calculate corresponding i (since i + j = p)
		i_rev = ord_dyadic_length_1 - i - 1;
		j_rev = ord_dyadic_length_2 - startj - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_a(a[j], gram, gram_idx, dyadic_frac);
		}

		//Update diagonals
		startj = ord_dyadic_length_1 > p ? 1 : p - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p ? p : ord_dyadic_length_2;

		i = p - startj; // Calculate corresponding i (since i + j = p)
		i_rev = ord_dyadic_length_1 - i - 1;
		j_rev = ord_dyadic_length_2 - startj - 1;
		uint64_t idx = order ? (i_rev + 1) * dyadic_length_2 + (j_rev + 1) : (j_rev + 1) * dyadic_length_2 + (i_rev + 1); //NOT ord_dyadic_length_2 here, as we are indexing k_grid
		k12 = k_grid + idx - 1;
		k21 = k_grid + idx - dyadic_length_2; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
		k11 = k21 - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);

			//Get da, db
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;
			get_a_b_deriv(da, db, gram, gram_idx, dyadic_frac);

			// Update dF / dk
			*(next_diag + j) = *(prev_diag + j - 1) * a[j-1] + *(prev_diag + j) * a[j] - *(prev_prev_diag + j - 1) * b[j-1];

			// Update dF / dx
			out[gram_idx] += *(next_diag + j) * ( (*(k12) + *(k21)) * da - *(k11) * db );

			if (order) {
				k12 += dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
				k21 += dyadic_length_2 - 1;
				k11 += dyadic_length_2 - 1;
			}
			else {
				k12 -= dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
				k21 -= dyadic_length_2 - 1;
				k11 -= dyadic_length_2 - 1;
			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		double* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}
}

void get_sig_kernel_diag_(
	const double* gram,
	uint64_t length1,
	uint64_t length2,
	double* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);

void sig_kernel_(
	const double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid = false
);

void batch_sig_kernel_(
	const double* gram,
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

void sig_kernel_backprop_(
	const double* gram,
	double* out,
	double deriv,
	const double* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);

void batch_sig_kernel_backprop_(
	const double* gram,
	double* out,
	const double* derivs,
	const double* k_grid,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs
);

void get_sig_kernel_backprop_diag_(
	const double* gram,
	double* out,
	const double deriv,
	const double* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);

void get_sig_kernel_backprop_(
	const double* gram,
	double* out,
	const double deriv,
	const double* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);