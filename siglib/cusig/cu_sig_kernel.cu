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

#include "cupch.h"
#include "cusig.h"
//#include "cuda_constants.h"
#include "cu_sig_kernel.h"

__constant__ uint64_t dimension;
__constant__ uint64_t length1;
__constant__ uint64_t length2;
__constant__ uint64_t dyadic_order_1;
__constant__ uint64_t dyadic_order_2;

__constant__ double twelth;
__constant__ uint64_t dyadic_length_1;
__constant__ uint64_t dyadic_length_2;
__constant__ uint64_t num_anti_diag;
__constant__ double dyadic_frac;
__constant__ uint64_t gram_length;
__constant__ uint64_t grid_length;


__global__ void goursat_pde(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* gram
) {
	int blockId = blockIdx.x;

	double* initial_condition_ = initial_condition + blockId * dyadic_length_1;
	double* gram_ = gram + blockId * gram_length;

	__shared__ double diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused

	uint64_t num_full_runs = (dyadic_length_2 - 1) / 32;
	uint64_t remainder = (dyadic_length_2 - 1) % 32;

	for (int i = 0; i < num_full_runs; ++i)
		goursat_pde_32(initial_condition_, diagonals, gram_, i, 32);

	if (remainder)
		goursat_pde_32(initial_condition_, diagonals, gram_, num_full_runs, remainder);
}

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

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}
}

__device__ void goursat_pde_32_full(
	double* pde_grid, //32 x L2
	double* gram,
	uint64_t iteration,
	int num_threads
) {
	int thread_id = threadIdx.x;

	double* pde_grid_ = pde_grid + iteration * 32 * dyadic_length_2;

	__syncthreads();

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		uint64_t startj, endj;
		if (dyadic_length_1 > p) startj = 1ULL;
		else startj = p - dyadic_length_1 + 1;
		if (num_threads + 1 > p) endj = p;
		else endj = num_threads + 1;

		uint64_t j = startj + thread_id;

		if (j < endj) {

			uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			uint64_t ii = ((i - 1) >> dyadic_order_1);
			uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_2);

			double deriv = gram[ii * (length2 - 1) + jj];
			deriv *= dyadic_frac;
			double deriv2 = deriv * deriv * twelth;

			pde_grid_[i * dyadic_length_2 + j] = (pde_grid_[(i - 1) * dyadic_length_2 + j] + pde_grid_[i * dyadic_length_2 + j - 1]) * (
				1. + 0.5 * deriv + deriv2) - pde_grid_[(i - 1) * dyadic_length_2 + j - 1] * (1. - deriv2);

		}
		// Wait for all threads to finish
		__syncthreads();
	}
}

__global__ void goursat_pde_full(
	double* pde_grid,
	double* gram
) {
	int blockId = blockIdx.x;

	double* gram_ = gram + blockId * gram_length;
	double* pde_grid_ = pde_grid + blockId * grid_length;

	uint64_t num_full_runs = (dyadic_length_2 - 1) / 32;
	uint64_t remainder = (dyadic_length_2 - 1) % 32;

	for (int i = 0; i < num_full_runs; ++i)
		goursat_pde_32_full(pde_grid_, gram_, i, 32);

	if (remainder)
		goursat_pde_32_full(pde_grid_, gram_, num_full_runs, remainder);
}

void sig_kernel_cuda_(
	double* gram,
	double* out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_,
	bool return_grid
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	static const double twelth_ = 1. / 12;
	const uint64_t dyadic_length_1_ = ((length1_ - 1) << dyadic_order_1_) + 1;
	const uint64_t dyadic_length_2_ = ((length2_ - 1) << dyadic_order_2_) + 1;
	const uint64_t num_anti_diag_ = 33 + dyadic_length_1_ - 1;
	const double dyadic_frac_ = 1. / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const uint64_t gram_length_ = (length1_ - 1) * (length2_ - 1);
	const uint64_t grid_length_ = dyadic_length_1_ * dyadic_length_2_;

	if (dyadic_length_2_ > dyadic_length_1_) { throw std::invalid_argument("The dyadically refined length of path2 must be less than or equal to that of path1. Please swap path1 and path2."); }

	// Allocate constant memory
	cudaMemcpyToSymbol(dimension, &dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length1, &length1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length2, &length2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_order_1, &dyadic_order_1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_order_2, &dyadic_order_2_, sizeof(uint64_t));

	cudaMemcpyToSymbol(twelth, &twelth_, sizeof(double));
	cudaMemcpyToSymbol(dyadic_length_1, &dyadic_length_1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_length_2, &dyadic_length_2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(num_anti_diag, &num_anti_diag_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_frac, &dyadic_frac_, sizeof(double));
	cudaMemcpyToSymbol(gram_length, &gram_length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(grid_length, &grid_length_, sizeof(uint64_t));

	if (!return_grid) {
		// Allocate initial condition
		auto ones_uptr = std::make_unique<double[]>(dyadic_length_1_ * batch_size_);
		double* ones = ones_uptr.get();
		std::fill(ones, ones + dyadic_length_1_ * batch_size_, 1.);

		double* initial_condition;
		cudaMalloc((void**)&initial_condition, dyadic_length_1_ * batch_size_ * sizeof(double));
		cudaMemcpy(initial_condition, ones, dyadic_length_1_ * batch_size_ * sizeof(double), cudaMemcpyHostToDevice);
		ones_uptr.reset();

		goursat_pde << <static_cast<unsigned int>(batch_size_), 32U >> > (initial_condition, gram);

		for (uint64_t i = 0; i < batch_size_; ++i)
			cudaMemcpy(out + i, initial_condition + (i + 1) * dyadic_length_1_ - 1, sizeof(double), cudaMemcpyDeviceToDevice);
		cudaFree(initial_condition);
	}
	else {
		// Allocate pde grid
		auto ones_uptr = std::make_unique<double[]>(grid_length_ * batch_size_);
		double* ones = ones_uptr.get();
		std::fill(ones, ones + batch_size_ * grid_length_, 1.);//TODO: avoid fill with all 1s

		//TODO: avoid cudaMemcpy of entire grid
		double* pde_grid;
		cudaMalloc((void**)&pde_grid, batch_size_ * grid_length_ * sizeof(double));
		cudaMemcpy(pde_grid, ones, batch_size_ * grid_length_ * sizeof(double), cudaMemcpyHostToDevice);
		ones_uptr.reset();

		goursat_pde_full << <static_cast<unsigned int>(batch_size_), 32U >> > (pde_grid, gram);

		cudaMemcpy(out, pde_grid, batch_size_ * grid_length_ * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaFree(pde_grid);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		int error_code = static_cast<int>(err);
        throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
}

#define SAFE_CALL(function_call)                            \
    try {                                                   \
        function_call;                                      \
    }                                                       \
    catch (std::bad_alloc&) {					            \
		std::cerr << "Failed to allocate memory";           \
        return 1;                                           \
    }                                                       \
    catch (std::invalid_argument& e) {                      \
		std::cerr << e.what();					            \
        return 2;                                           \
    }                                                       \
	catch (std::out_of_range& e) {			                \
		std::cerr << e.what();					            \
		return 3;                                           \
	}  											            \
	catch (std::runtime_error& e) {							\
		std::string msg = e.what();							\
		std::regex pattern(R"(CUDA Error \((\d+)\):)");		\
		std::smatch match;									\
		int ret_code = 4;									\
		if (std::regex_search(msg, match, pattern)) {		\
			ret_code = 100000 + std::stoi(match[1]);		\
		}													\
		std::cerr << e.what();								\
		return ret_code;									\
	}														\
    catch (...) {                                           \
		std::cerr << "Unknown exception";		            \
        return 5;                                           \
    }                                                       \
    return 0;


extern "C" {

	CUSIG_API int sig_kernel_cuda(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_cuda_(gram, out, 1ULL, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int batch_sig_kernel_cuda(double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_cuda_(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}
}
