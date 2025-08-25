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
#include "cupch.h"

extern __constant__ uint64_t dimension;
extern __constant__ uint64_t length1;
extern __constant__ uint64_t length2;
extern __constant__ uint64_t dyadic_order_1;
extern __constant__ uint64_t dyadic_order_2;

extern __constant__ double twelth;
extern __constant__ double sixth;
extern __constant__ uint64_t dyadic_length_1;
extern __constant__ uint64_t dyadic_length_2;
extern __constant__ uint64_t main_dyadic_length;
extern __constant__ uint64_t num_anti_diag;
extern __constant__ double dyadic_frac;
extern __constant__ uint64_t gram_length;
extern __constant__ uint64_t grid_length;

__device__ double myAtomicAdd(double* address, double val);

inline __device__ void get_a_b(double& a, double& b, const double* const gram, const uint64_t idx, const double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	const double gram_val_2 = gram_val * gram_val * twelth;
	a = 1. + 0.5 * gram_val + gram_val_2;
	b = 1. - gram_val_2;
}

inline __device__ void get_a(double& a, const double* const gram, const uint64_t idx, const double dyadic_frac) {
	static const double twelth = 1. / 12;
	double gram_val = gram[idx] * dyadic_frac;
	a = 1. + gram_val * (0.5 + gram_val * twelth);
}

inline __device__ void get_b(double& b, const double* const gram, const uint64_t idx, const double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	b = 1. - gram_val * gram_val * twelth;
}

inline __device__ void get_a_b_deriv(double& a_deriv, double& b_deriv, const double* const gram, const uint64_t idx, const double dyadic_frac) {
	static const double sixth = 1. / 6;
	const double gram_val = gram[idx] * dyadic_frac;
	b_deriv = -gram_val * sixth * dyadic_frac;
	a_deriv = 0.5 * dyadic_frac - b_deriv;
}

__global__ void goursat_pde(
	double* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	const double* const gram
);

template<bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32(
	double* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* const diagonals,
	const double* const gram,
	const uint64_t iteration,
	const int num_threads
) {
	const int thread_id = threadIdx.x;

	const uint64_t ord_dyadic_order_1 = order ? dyadic_order_1 : dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? dyadic_order_2 : dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? dyadic_length_2 : dyadic_length_1;

	// Initialise to 1
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + thread_id + 1] = 1.;

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prev_prev_diag_idx = 0;
	int prev_diag_idx = 33;
	int next_diag_idx = 66;

	if (thread_id == 0) {
		diagonals[prev_prev_diag_idx] = initial_condition[0];
		diagonals[prev_diag_idx] = initial_condition[1];
	}

	__syncthreads();

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		uint64_t startj, endj;
		if (ord_dyadic_length_1 > p) startj = 1ULL;
		else startj = p - ord_dyadic_length_1 + 1;
		if (num_threads + 1 > p) endj = p;
		else endj = num_threads + 1;

		const uint64_t j = startj + thread_id;

		if (j < endj) {

			// Make sure correct initial condition is filled in for first thread
			if (thread_id == 0 && p < ord_dyadic_length_1) {
				diagonals[next_diag_idx] = initial_condition[p];
			}

			const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			const uint64_t ii = ((i - 1) >> ord_dyadic_order_1);
			const uint64_t jj = ((j + iteration * 32 - 1) >> ord_dyadic_order_2);

			const double deriv = order ? gram[ii * (length2 - 1) + jj] * dyadic_frac : gram[jj * (length2 - 1) + ii] * dyadic_frac;
			const double deriv2 = deriv * deriv * twelth;

			diagonals[next_diag_idx + j] = (diagonals[prev_diag_idx + j] + diagonals[prev_diag_idx + j - 1]) * (
				1. + 0.5 * deriv + deriv2) - diagonals[prev_prev_diag_idx + j - 1] * (1. - deriv2);

		}

		// Wait for all threads to finish
		__syncthreads();

		// Overwrite initial condition with result
		// Safe to do since we won't be using initial_condition[p-num_threads] any more
		if (thread_id == 0 && p >= num_threads && p - num_threads < ord_dyadic_length_1)
			initial_condition[p - num_threads] = diagonals[next_diag_idx + num_threads];

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}
}

template<bool order>
__device__ void goursat_pde_32_full(
	double* const pde_grid, //32 x L2
	const double* const gram,
	const uint64_t iteration,
	const int num_threads
) {
	const int thread_id = threadIdx.x;
	double* const pde_grid_ = order ? pde_grid + iteration * 32 : pde_grid + iteration * 32 * dyadic_length_2;

	const uint64_t ord_dyadic_order_1 = order ? dyadic_order_1 : dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? dyadic_order_2 : dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? dyadic_length_2 : dyadic_length_1;

	__syncthreads();

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		uint64_t startj, endj;
		if (ord_dyadic_length_1 > p) startj = 1ULL;
		else startj = p - ord_dyadic_length_1 + 1;
		if (num_threads + 1 > p) endj = p;
		else endj = num_threads + 1;

		const uint64_t j = startj + thread_id;

		if (j < endj) {

			const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			const uint64_t ii = ((i - 1) >> ord_dyadic_order_1);
			const uint64_t jj = ((j + iteration * 32 - 1) >> ord_dyadic_order_2);

			const double deriv = order ? gram[ii * (length2 - 1) + jj] * dyadic_frac : gram[jj * (length2 - 1) + ii] * dyadic_frac;
			const double deriv2 = deriv * deriv * twelth;

			if (order) {
				pde_grid_[i * dyadic_length_2 + j] = (pde_grid_[(i - 1) * dyadic_length_2 + j] + pde_grid_[i * dyadic_length_2 + (j - 1)]) * (
					1. + 0.5 * deriv + deriv2) - pde_grid_[(i - 1) * dyadic_length_2 + j - 1] * (1. - deriv2);
			}
			else {
				pde_grid_[j * dyadic_length_2 + i] = (pde_grid_[(j - 1) * dyadic_length_2 + i] + pde_grid_[j * dyadic_length_2 + (i - 1)]) * (
					1. + 0.5 * deriv + deriv2) - pde_grid_[(j - 1) * dyadic_length_2 + i - 1] * (1. - deriv2);
			}

		}

		// Wait for all threads to finish
		__syncthreads();
	}
}

template<bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32_deriv(
	const double deriv,
	const double* const k_grid,
	double* const out,
	double* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* const a_initial_condition,
	double* const b_initial_condition,
	double* const diagonals,
	double* const a,
	double* const b,
	const double* const gram,
	const uint64_t iteration,
	const int num_threads
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

	const int thread_id = threadIdx.x;

	// As with the diagonal method for sig_kernel, it matters which of
	// dyadic_length_1 and dyadic_length_2 is longer.
	const uint64_t ord_dyadic_order_1 = order ? dyadic_order_1 : dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? dyadic_order_2 : dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? dyadic_length_2 : dyadic_length_1;

	// Ptrs for diagonals
	double* prev_prev_diag = diagonals;
	double* prev_diag = prev_prev_diag + 33;
	double* next_diag = prev_diag + 33;

	// k_grid ptrs
	const double* k11, * k12, * k21;

	// Initialization
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + thread_id + 1] = 0.;

	a[thread_id + 1] = 1.;
	b[thread_id + 1] = 1.;

	if (thread_id == 0) {
		a[0] = 1.;
		b[0] = 1.;

		*prev_prev_diag = initial_condition[0];
		*prev_diag = initial_condition[1];

		if (iteration == 0) {
			*(prev_diag + 1) = deriv;
			double da, db;
			get_a_b_deriv(da, db, gram, gram_length - 1, dyadic_frac);

			//Update dF / dx for first value
			k21 = k_grid + grid_length - 2;
			k12 = k_grid + grid_length - dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
			k11 = k12 - 1;
			out[gram_length - 1] += deriv * (((*k21) + (*k12)) * da - *(k11)*db);
		}
	}

	__syncthreads();

	// First three antidiagonals are initialised
	// num_anti_diag + 2 so that a and b are updated as initial conds
	for (uint64_t p = (iteration == 0) ? 3 : 2; p < num_anti_diag + 2; ++p) {

		//Update b
		uint64_t startj, endj;
		int64_t p_ = p - 2;
		startj = ord_dyadic_length_1 > p_ ? 1ULL : p_ - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > p_ ? p_ : num_threads + 1;

		uint64_t j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && p_ < ord_dyadic_length_1) {
			b[0] = b_initial_condition[p_];
		}

		if (j < endj) {
			const uint64_t i = p_ - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_b(b[j], gram, gram_idx, dyadic_frac);
		}

		__syncthreads();

		//Overwrite initial conditions
		if (thread_id == 0 && p_ >= num_threads && p_ - num_threads < ord_dyadic_length_1) {
			b_initial_condition[p_ - num_threads] = b[num_threads];
		}

		//Update a
		p_ = p - 1;
		startj = ord_dyadic_length_1 > p_ ? 1ULL : p_ - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > p_ ? p_ : num_threads + 1;

		j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && p_ < ord_dyadic_length_1) {
			a[0] = a_initial_condition[p_];
		}

		if (j < endj) {
			const uint64_t i = p_ - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_a(a[j], gram, gram_idx, dyadic_frac);
		}

		__syncthreads();

		//Overwrite initial conditions
		if (thread_id == 0 && p_ >= num_threads && p_ - num_threads < ord_dyadic_length_1) {
			a_initial_condition[p_ - num_threads] = a[num_threads];
		}

		//Update diagonals
		startj = ord_dyadic_length_1 > p ? 1ULL : p - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > p ? p : num_threads + 1;

		j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && p < ord_dyadic_length_1) {
			*(next_diag) = initial_condition[p];
		}

		if (j < endj) {
			const uint64_t i = p - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t idx = order ? (i_rev + 1) * dyadic_length_2 + (j_rev + 1) : (j_rev + 1) * dyadic_length_2 + (i_rev + 1); //NOT ord_dyadic_length_2 here as we are indexing k_grid
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			//Get da, db
			double da, db;
			get_a_b_deriv(da, db, gram, gram_idx, dyadic_frac);

			// Update dF / dk
			*(next_diag + j) = *(prev_diag + j - 1) * a[j - 1] + *(prev_diag + j) * a[j] - *(prev_prev_diag + j - 1) * b[j - 1];

			// Update dF / dx
			k12 = k_grid + idx - 1;
			k21 = k_grid + idx - dyadic_length_2; //NOT ord_dyadic_length_2 here as we are indexing k_grid
			k11 = k_grid + idx - dyadic_length_2 - 1;
			double result = *(next_diag + j) * ((*(k12)+*(k21)) * da - *(k11)*db);

			// Avoid race conditions for non-zero dyadic orders
			myAtomicAdd(&out[gram_idx], result);
		}

		__syncthreads();

		//Overwrite initial conditions
		if (thread_id == 0 && p >= num_threads && p - num_threads < ord_dyadic_length_1) {
			initial_condition[p - num_threads] = *(next_diag + num_threads);
		}

		// Rotate the diagonals (swap pointers, no data copying)
		double* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;

		__syncthreads();
	}
}

void sig_kernel_cuda_(
	const double* const gram,
	double* const out,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_,
	const bool return_grid
);

__global__ void goursat_pde_deriv(
	double* const initial_condition,
	double* const a_initial_condition,
	double* const b_initial_condition,
	const double* const gram,
	const double* const deriv,
	const double* const k_grid,
	double* const out
);

void sig_kernel_backprop_cuda_(
	const double* const gram,
	double* const out,
	const double* const deriv,
	const double* const k_grid,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_
);