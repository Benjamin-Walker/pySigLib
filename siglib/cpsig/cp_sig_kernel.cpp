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
	const double* gram,
	uint64_t length1,
	uint64_t length2,
	double* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;

	// Dyadically refined grid dimensions
	const uint64_t grid_size_1 = 1ULL << dyadic_order_1;
	const uint64_t grid_size_2 = 1ULL << dyadic_order_2;
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	// Allocate(flattened) PDE grid
	double* pde_grid;
	if (return_grid)
		pde_grid = out;
	else {
		auto pde_grid_uptr = std::make_unique<double[]>(dyadic_length_1 * dyadic_length_2);
		pde_grid = pde_grid_uptr.get();
	}

	// Initialization of K array
	for (uint64_t i = 0; i < dyadic_length_1; ++i) {
		pde_grid[i * dyadic_length_2] = 1.0; // Set K[i, 0] = 1.0
	}

	std::fill(pde_grid, pde_grid + dyadic_length_2, 1.0); // Set K[0, j] = 1.0

	auto deriv_term_1_uptr = std::make_unique<double[]>(length2 - 1);
	double* const deriv_term_1 = deriv_term_1_uptr.get();

	auto deriv_term_2_uptr = std::make_unique<double[]>(length2 - 1);
	double* const deriv_term_2 = deriv_term_2_uptr.get();

	double* k11 = pde_grid;
	double* k12 = k11 + 1;
	double* k21 = k11 + dyadic_length_2;
	double* k22 = k21 + 1;

	const double* gram_ptr = gram;

	for (uint64_t ii = 0; ii < length1 - 1; ++ii, gram_ptr += length2 - 1) {
		for (uint64_t m = 0; m < length2 - 1; ++m) {
			const double deriv = gram_ptr[m] * dyadic_frac;//dot_product(diff1Ptr, diff2Ptr, dimension);
			const double deriv2 = deriv * deriv * twelth;
			deriv_term_1[m] = 1.0 + 0.5 * deriv + deriv2;
			deriv_term_2[m] = 1.0 - deriv2;
		}

		for (uint64_t i = 0;
			i < grid_size_1;
			++i, ++k11, ++k12, ++k21, ++k22) {

			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				const double t1 = deriv_term_1[jj];
				const double t2 = deriv_term_2[jj];
				for (uint64_t j = 0; j < grid_size_2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * t1 - *(k11++) * t2;
				}
			}
		}
	}

	if (!return_grid)
		*out = pde_grid[dyadic_length_1 * dyadic_length_2 - 1];
}

void get_sig_kernel_diag_(
	const double* gram,
	uint64_t length1,
	uint64_t length2,
	double* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	// Dyadically refined grid dimensions
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
	
	if (dyadic_length_2 <= dyadic_length_1)
		get_sig_kernel_diag_internal_<true>(gram, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
	else
		get_sig_kernel_diag_internal_<false>(gram, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
}

void sig_kernel_(
	const double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	if (return_grid)
		get_sig_kernel_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2, true);
	else
		get_sig_kernel_diag_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2);
}

void batch_sig_kernel_(
	const double* gram,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs,
	bool return_grid
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	if (!gram) {
		std::fill(out, out + batch_size, 1.);
		return;
	}

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const double* const data_end_1 = gram + gram_length * batch_size;
	const uint64_t result_length = return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1;

	std::function<void(const double* const, double* const)> sig_kernel_func;

	if (return_grid) {
		sig_kernel_func = [&](const double* const gram_ptr, double* const out_ptr) {
			get_sig_kernel_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2, true);
			};
	}
	else {
		sig_kernel_func = [&](const double* const gram_ptr, double* const out_ptr) {
			get_sig_kernel_diag_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2);
			};
	}

	if (n_jobs != 1) {
		multi_threaded_batch(sig_kernel_func, gram, out, batch_size, gram_length, result_length, n_jobs);
	}
	else {
		const double* gram_ptr = gram;
		double* out_ptr = out;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += result_length) {

			sig_kernel_func(gram_ptr, out_ptr);
		}
	}
	return;
}

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
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	get_sig_kernel_backprop_(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2);
	//get_sig_kernel_backprop_diag_(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2);
}

void get_sig_kernel_backprop_diag_(
	const double* gram,
	double* out,
	double deriv,
	const double* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	// Dyadically refined grid dimensions
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	if (dyadic_length_2 <= dyadic_length_1)
		get_sig_kernel_backprop_diag_internal_<true>(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
	else
		get_sig_kernel_backprop_diag_internal_<false>(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
}

void get_sig_kernel_backprop_(
	const double* gram,
	double* out,
	double deriv,
	const double* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	static const double sixth = 1. / 6;
	static const double twelth = 1. / 12;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	// Allocate grid for dF / dk
	auto d_grid_uptr = std::make_unique<double[]>(grid_length);
	double* const d_grid = d_grid_uptr.get();

	std::fill(out, out + (length1 - 1) * (length2 - 1), 0.);

	// a, b, da, db
	double a, a_deriv, b_deriv;
	double a10, a01, b11;

	// indices
	uint64_t grid_idx, gram_idx;

	// k_grid and d_grid ptrs
	const double* k11, * k12, * k21;
	const double* d11, * d12, * d21;

	//Start with the last dF / dk, which is known ============================================
	d_grid[grid_length - 1] = deriv;

	//Compute dA(i-1, j-1) and dB(i-1, j-1)
	gram_idx = gram_length - 1;
	get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

	//Update dF / dx
	k21 = k_grid + grid_length - 2;
	k12 = k_grid + grid_length - dyadic_length_2 - 1;
	k11 = k12 - 1;
	out[gram_length - 1] += d_grid[grid_length - 1] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);

	//Loop over last row ============================================
	grid_idx = grid_length - 2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;

	for (int64_t i = dyadic_length_2 - 2;
		i >= 1;
		--i, --grid_idx, --k12, --k21, --k11) {

		const int64_t j = dyadic_length_1 - 1;

		//Precompute indices
		const uint64_t cur_ii = (i >> dyadic_order_2);
		const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
		const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

		//Compute A(i, j-1)
		get_a(a, gram, prev_jj + cur_ii, dyadic_frac);

		//Update dF / dk
		d_grid[grid_idx] = d_grid[grid_idx + 1] * a;

		//Compute dA(i-1, j-1) and dB(i-1, j-1)
		gram_idx = prev_jj + prev_ii;
		get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

		//Update dF / dx
		out[gram_idx] += d_grid[grid_idx] * ( (*k12 + *k21) * a_deriv - *k11 * b_deriv );
	}

	grid_idx = grid_length - 1 - dyadic_length_2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;
	//Loop over last column ============================================
	for (int64_t j = dyadic_length_1 - 2;
		j >= 1;
		--j,
		grid_idx -= dyadic_length_2,
		k21 -= dyadic_length_2,
		k12 -= dyadic_length_2,
		k11 -= dyadic_length_2) {

		const int64_t i = dyadic_length_2 - 1;

		//Precompute indices
		const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
		const uint64_t cur_jj = (j >> dyadic_order_1) * (length2 - 1);
		const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

		//Compute A(i-1, j)
		get_a(a, gram, cur_jj + prev_ii, dyadic_frac);

		//Update dF / dk
		d_grid[grid_idx] = d_grid[grid_idx + dyadic_length_2] * a;

		//Compute dA(i-1, j-1) and dB(i-1, j-1)
		gram_idx = prev_jj + prev_ii;
		get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

		//Update dF / dx
		out[gram_idx] += d_grid[grid_idx] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);
	}

	// Loop over remaining grid ============================================
	grid_idx = grid_length - 2 - dyadic_length_2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;
	d21 = d_grid + grid_idx + 1;
	d12 = d_grid + grid_idx + dyadic_length_2;
	d11 = d12 + 1;
	for (int64_t j = dyadic_length_1 - 2;
		j >= 1;
		--j,
		grid_idx -= 2,
		k12 -= 2,
		k21 -= 2,
		k11 -= 2,
		d12 -= 2,
		d21 -= 2,
		d11 -= 2) {

		for (int64_t i = dyadic_length_2 - 2;
			i >= 1;
			--i,
			--grid_idx,
			--k12,
			--k21,
			--k11,
			--d12,
			--d21,
			--d11) {

			//Precompute indices
			const uint64_t cur_ii = (i >> dyadic_order_2);
			const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
			const uint64_t cur_jj = (j >> dyadic_order_1) * (length2 - 1);
			const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

			// Compute A(i, j-1)
			get_a(a10, gram, prev_jj + cur_ii, dyadic_frac);

			// Compute A(i-1, j)
			get_a(a01, gram, cur_jj + prev_ii, dyadic_frac);

			// Compute B(i, j)
			get_b(b11, gram, cur_jj + cur_ii, dyadic_frac);

			//Update dF / dk
			d_grid[grid_idx] = (*d21) * a10 + (*d12) * a01 - (*d11) * b11;

			//Compute dA(i-1, j-1) and dB(i-1, j-1)
			gram_idx = prev_jj + prev_ii;
			get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

			//Update dF / dx
			out[gram_idx] += d_grid[grid_idx] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);
		}
	}

	return;
}

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
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	
	if (!gram) {
		std::fill(out, out + batch_size * gram_length, 0.);
		return;
	}

	const double* const data_end_1 = gram + gram_length * batch_size;

	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;

	std::function<void(const double* const, const double* const, const double* const, double* const)> sig_kernel_backprop_func;

	sig_kernel_backprop_func = [&](const double* const gram_ptr, const double* const deriv_ptr, const double* const k_grid_ptr, double* const out_ptr) {
		sig_kernel_backprop_(gram_ptr, out_ptr, *deriv_ptr, k_grid_ptr, dimension, length1, length2, dyadic_order_1, dyadic_order_2);
		};

	if (n_jobs != 1) {
		multi_threaded_batch_3(sig_kernel_backprop_func, gram, derivs, k_grid, out, batch_size, gram_length, 1, grid_length, gram_length, n_jobs);
	}
	else {
		const double* gram_ptr = gram;
		double* out_ptr = out;
		const double* deriv_ptr = derivs;
		const double* k_grid_ptr = k_grid;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += gram_length, deriv_ptr += 1, k_grid_ptr += grid_length) {

			sig_kernel_backprop_func(gram_ptr, deriv_ptr, k_grid_ptr, out_ptr);
		}
	}
	return;
}


extern "C" {

	CPSIG_API int sig_kernel(const double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_(gram, out, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CPSIG_API int batch_sig_kernel(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs, bool return_grid) noexcept {
		SAFE_CALL(batch_sig_kernel_(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs, return_grid));
	}

	CPSIG_API int sig_kernel_backprop(const double* gram, double* out, double deriv, const double* k_grid, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept {
		SAFE_CALL(sig_kernel_backprop_(gram, out, deriv, k_grid, dimension, length1, length2, dyadic_order_1, dyadic_order_2));
	}

	CPSIG_API int batch_sig_kernel_backprop(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_kernel_backprop_(gram, out, derivs, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs));
	}
}
