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
__constant__ double sixth;
__constant__ uint64_t dyadic_length_1;
__constant__ uint64_t dyadic_length_2;
__constant__ uint64_t main_dyadic_length;
__constant__ uint64_t num_anti_diag;
__constant__ double dyadic_frac;
__constant__ uint64_t gram_length;
__constant__ uint64_t grid_length;


// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double myAtomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}


__global__ void goursat_pde(
	double* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	const double* const gram
) {
	const int blockId = blockIdx.x;
	const double* const gram_ = gram + blockId * gram_length;

	__shared__ double diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused

	if (dyadic_length_2 <= dyadic_length_1) {
		double* const initial_condition_ = initial_condition + blockId * dyadic_length_1;

		const uint64_t num_full_runs = (dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32<true>(initial_condition_, diagonals, gram_, i, 32);

		if (remainder)
			goursat_pde_32<true>(initial_condition_, diagonals, gram_, num_full_runs, remainder);
	}
	else {
		double* const initial_condition_ = initial_condition + blockId * dyadic_length_2;

		const uint64_t num_full_runs = (dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32<false>(initial_condition_, diagonals, gram_, i, 32);

		if (remainder)
			goursat_pde_32<false>(initial_condition_, diagonals, gram_, num_full_runs, remainder);
	}
}

__global__ void goursat_pde_full(
	double* const pde_grid,
	const double* const gram
) {
	const int blockId = blockIdx.x;

	const double* const gram_ = gram + blockId * gram_length;
	double* const pde_grid_ = pde_grid + blockId * grid_length;

	if (dyadic_length_2 <= dyadic_length_1) {
		const uint64_t num_full_runs = (dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_full<true>(pde_grid_, gram_, i, 32);

		if (remainder)
			goursat_pde_32_full<true>(pde_grid_, gram_, num_full_runs, remainder);
	}
	else {
		const uint64_t num_full_runs = (dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_full<false>(pde_grid_, gram_, i, 32);

		if (remainder)
			goursat_pde_32_full<false>(pde_grid_, gram_, num_full_runs, remainder);
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
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	static const double twelth_ = 1. / 12;
	const uint64_t dyadic_length_1_ = ((length1_ - 1) << dyadic_order_1_) + 1;
	const uint64_t dyadic_length_2_ = ((length2_ - 1) << dyadic_order_2_) + 1;
	const uint64_t main_dyadic_length_ = dyadic_length_2_ <= dyadic_length_1_ ? dyadic_length_1_ : dyadic_length_2_;
	const uint64_t num_anti_diag_ = 33 + main_dyadic_length_ - 1;
	const double dyadic_frac_ = 1. / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const uint64_t gram_length_ = (length1_ - 1) * (length2_ - 1);
	const uint64_t grid_length_ = dyadic_length_1_ * dyadic_length_2_;

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
		auto ones_uptr = std::make_unique<double[]>(main_dyadic_length_ * batch_size_);
		double* const ones = ones_uptr.get();
		std::fill(ones, ones + main_dyadic_length_ * batch_size_, 1.);

		double* initial_condition;
		cudaMalloc((void**)&initial_condition, main_dyadic_length_ * batch_size_ * sizeof(double));
		cudaMemcpy(initial_condition, ones, main_dyadic_length_ * batch_size_ * sizeof(double), cudaMemcpyHostToDevice);
		ones_uptr.reset();

		goursat_pde << <static_cast<unsigned int>(batch_size_), 32U >> > (initial_condition, gram);

		for (uint64_t i = 0; i < batch_size_; ++i)
			cudaMemcpy(out + i, initial_condition + (i + 1) * main_dyadic_length_ - 1, sizeof(double), cudaMemcpyDeviceToDevice);
		cudaFree(initial_condition);
	}
	else {
		// Allocate pde grid
		auto ones_uptr = std::make_unique<double[]>(grid_length_ * batch_size_);
		double* const ones = ones_uptr.get();
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
		const int error_code = static_cast<int>(err);
        throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
}

__global__ void goursat_pde_deriv(
	double* const initial_condition, //This is the top row of the grid, which will be overwritten
	double* const a_initial_condition,
	double* const b_initial_condition,
	const double* const gram,
	const double* const deriv,
	const double* const k_grid,
	double* const out
) {
	const int blockId = blockIdx.x;
	const double* const gram_ = gram + blockId * gram_length;
	const double deriv_ = *(deriv + blockId);
	const double* const k_grid_ = k_grid + blockId * grid_length;
	double* const out_ = out + blockId * gram_length;

	__shared__ double diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused
	__shared__ double a[33];
	__shared__ double b[33];

	if (dyadic_length_2 <= dyadic_length_1) {
		double* const initial_condition_ = initial_condition + blockId * dyadic_length_1;
		double* const a_initial_condition_ = a_initial_condition + blockId * dyadic_length_1;
		double* const b_initial_condition_ = b_initial_condition + blockId * dyadic_length_1;

		const uint64_t num_full_runs = (dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_deriv<true>(deriv_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, i, 32);

		if (remainder)
			goursat_pde_32_deriv<true>(deriv_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, num_full_runs, remainder);
	}
	else {
		double* const initial_condition_ = initial_condition + blockId * dyadic_length_2;
		double* const a_initial_condition_ = a_initial_condition + blockId * dyadic_length_2;
		double* const b_initial_condition_ = b_initial_condition + blockId * dyadic_length_2;

		const uint64_t num_full_runs = (dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i) 
			goursat_pde_32_deriv<false>(deriv_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, i, 32);

		if (remainder)
			goursat_pde_32_deriv<false>(deriv_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, num_full_runs, remainder);
	}
}

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
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	static const double twelth_ = 1. / 12;
	static const double sixth_ = 1. / 6;
	const uint64_t dyadic_length_1_ = ((length1_ - 1) << dyadic_order_1_) + 1;
	const uint64_t dyadic_length_2_ = ((length2_ - 1) << dyadic_order_2_) + 1;
	const uint64_t main_dyadic_length_ = dyadic_length_2_ <= dyadic_length_1_ ? dyadic_length_1_ : dyadic_length_2_;
	const uint64_t num_anti_diag_ = 33 + main_dyadic_length_ - 1;
	const double dyadic_frac_ = 1. / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const uint64_t gram_length_ = (length1_ - 1) * (length2_ - 1);
	const uint64_t grid_length_ = dyadic_length_1_ * dyadic_length_2_;

	// Allocate constant memory
	cudaMemcpyToSymbol(dimension, &dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length1, &length1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length2, &length2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_order_1, &dyadic_order_1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_order_2, &dyadic_order_2_, sizeof(uint64_t));

	cudaMemcpyToSymbol(twelth, &twelth_, sizeof(double));
	cudaMemcpyToSymbol(sixth, &sixth_, sizeof(double));
	cudaMemcpyToSymbol(dyadic_length_1, &dyadic_length_1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_length_2, &dyadic_length_2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(main_dyadic_length, &main_dyadic_length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(num_anti_diag, &num_anti_diag_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadic_frac, &dyadic_frac_, sizeof(double));
	cudaMemcpyToSymbol(gram_length, &gram_length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(grid_length, &grid_length_, sizeof(uint64_t));

	//Initialise out to 0
	cudaMemset(out, 0, batch_size_ * gram_length_ * sizeof(double));

	double* d_initial_condition;
	cudaMalloc((void**)&d_initial_condition, main_dyadic_length_ * batch_size_ * sizeof(double));
	cudaMemset(d_initial_condition, 0, main_dyadic_length_ * batch_size_ * sizeof(double));

	double* d_a_initial_condition;
	cudaMalloc((void**)&d_a_initial_condition, main_dyadic_length_ * batch_size_ * sizeof(double));
	cudaMemset(d_a_initial_condition, 0, main_dyadic_length_ * batch_size_ * sizeof(double));

	double* d_b_initial_condition;
	cudaMalloc((void**)&d_b_initial_condition, main_dyadic_length_ * batch_size_ * sizeof(double));
	cudaMemset(d_b_initial_condition, 0, main_dyadic_length_ * batch_size_ * sizeof(double));

	goursat_pde_deriv << <static_cast<unsigned int>(batch_size_), 32U >> > (d_initial_condition, d_a_initial_condition, d_b_initial_condition, gram, deriv, k_grid, out);

	cudaFree(d_initial_condition);
	cudaFree(d_a_initial_condition);
	cudaFree(d_b_initial_condition);	

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

	CUSIG_API int sig_kernel_cuda(const double* const gram, double* const out, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_cuda_(gram, out, 1ULL, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int batch_sig_kernel_cuda(const double* const gram, double* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_cuda_(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int sig_kernel_backprop_cuda(const double* const gram, double* const out, const double deriv, const double* const k_grid, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2) noexcept {
		SAFE_CALL(sig_kernel_backprop_cuda_(gram, out, &deriv, k_grid, 1ULL, dimension, length1, length2, dyadic_order_1, dyadic_order_2));
	}

	CUSIG_API int batch_sig_kernel_backprop_cuda(const double* const gram, double* const out, const double* const deriv, const double* const k_grid, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2) noexcept {
		SAFE_CALL(sig_kernel_backprop_cuda_(gram, out, deriv, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2));
	}
}
