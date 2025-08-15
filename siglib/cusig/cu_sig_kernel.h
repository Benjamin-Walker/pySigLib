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
extern __constant__ uint64_t dyadic_length_1;
extern __constant__ uint64_t dyadic_length_2;
extern __constant__ uint64_t num_anti_diag;
extern __constant__ double dyadic_frac;
extern __constant__ uint64_t gram_length;
extern __constant__ uint64_t grid_length;

__global__ void goursat_pde(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* gram
);

template<bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* diagonals,
	double* gram,
	uint64_t iteration,
	int num_threads
) {
	int thread_id = threadIdx.x;

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

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_1) {
					diagonals[next_diag_idx] = initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_1);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_2);

				double deriv = gram[ii * (length2 - 1) + jj];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				diagonals[next_diag_idx + j] = (diagonals[prev_diag_idx + j] + diagonals[prev_diag_idx + j - 1]) * (
					1. + 0.5 * deriv + deriv2) - diagonals[prev_prev_diag_idx + j - 1] * (1. - deriv2);

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_1)
				initial_condition[p - num_threads] = diagonals[next_diag_idx + num_threads];
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_2) {
					diagonals[next_diag_idx] = initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_2);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_1);

				double deriv = gram[jj * (length2 - 1) + ii];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				diagonals[next_diag_idx + j] = (diagonals[prev_diag_idx + j] + diagonals[prev_diag_idx + j - 1]) * (
					1. + 0.5 * deriv + deriv2) - diagonals[prev_prev_diag_idx + j - 1] * (1. - deriv2);

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_2)
				initial_condition[p - num_threads] = diagonals[next_diag_idx + num_threads];
		}

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}
}

void sig_kernel_cuda_(
	double* gram,
	double* out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_
);