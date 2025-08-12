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
	const uint64_t dyadic_order_2,
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
	double* deriv_term_1 = deriv_term_1_uptr.get();

	auto deriv_term_2_uptr = std::make_unique<double[]>(length2 - 1);
	double* deriv_term_2 = deriv_term_2_uptr.get();

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

	if (!return_grid)
		*out = pde_grid[dyadic_length_1 * dyadic_length_2 - 1];
}

void get_sig_kernel_diag_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2
) {
	// Dyadically refined grid dimensions
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
	
	if (dyadic_length_2 <= dyadic_length_1)
		get_sig_kernel_diag_internal_<true>(gram, length1, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
	else
		get_sig_kernel_diag_internal_<false>(gram, length1, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
}

void sig_kernel_(
	double* gram,
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
	double* gram,
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
	double* const data_end_1 = gram + gram_length * batch_size;
	const uint64_t result_length = return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1;

	std::function<void(double*, double*)> sig_kernel_func;

	if (return_grid) {
		sig_kernel_func = [&](double* gram_ptr, double* out_ptr) {
			get_sig_kernel_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2, true);
			};
	}
	else {
		sig_kernel_func = [&](double* gram_ptr, double* out_ptr) {
			get_sig_kernel_diag_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2);
			};
	}

	if (n_jobs != 1) {
		multi_threaded_batch(sig_kernel_func, gram, out, batch_size, gram_length, result_length, n_jobs);
	}
	else {
		double* gram_ptr = gram;
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
	double* gram,
	double* out,
	double deriv,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
	double* out_ptr = out;

	if (dyadic_length_2 <= dyadic_length_1) {
		for (uint64_t i = 0; i < length1 - 1; ++i) {
			for (uint64_t j = 0; j < length2 - 1; ++j) {
				get_sig_kernel_deriv_<true>(i, j, gram, out_ptr, deriv, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
				++out_ptr;
			}
		}
	}
	else {
		for (uint64_t i = 0; i < length1 - 1; ++i) {
			for (uint64_t j = 0; j < length2 - 1; ++j) {
				get_sig_kernel_deriv_<false>(i, j, gram, out_ptr, deriv, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
				++out_ptr;
			}
		}
	}

	return;
}

void batch_sig_kernel_backprop_(
	double* gram,
	double* out,
	double* derivs,
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

	double* const data_end_1 = gram + gram_length * batch_size;

	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	std::function<void(double*, double*, double*)> sig_kernel_backprop_func;

	if (dyadic_length_2 <= dyadic_length_1) {
		sig_kernel_backprop_func = [&](double* gram_ptr, double* deriv_ptr, double* out_ptr) {
			double* out_ptr_ = out_ptr;
			for (uint64_t i = 0; i < length1 - 1; ++i) {
				for (uint64_t j = 0; j < length2 - 1; ++j) {
					get_sig_kernel_deriv_<true>(i, j, gram_ptr, out_ptr_, *deriv_ptr, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
					++out_ptr_;
				}
			}
			};
	}
	else {
		sig_kernel_backprop_func = [&](double* gram_ptr, double* deriv_ptr, double* out_ptr) {
			double* out_ptr_ = out_ptr;
			for (uint64_t i = 0; i < length1 - 1; ++i) {
				for (uint64_t j = 0; j < length2 - 1; ++j) {
					get_sig_kernel_deriv_<false>(i, j, gram_ptr, out_ptr_, *deriv_ptr, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
					++out_ptr_;
				}
			}
			};
	}

	if (n_jobs != 1) {
		multi_threaded_batch_2(sig_kernel_backprop_func, gram, derivs, out, batch_size, gram_length, 1, gram_length, n_jobs);
	}
	else {
		double* gram_ptr = gram;
		double* out_ptr = out;
		double* deriv_ptr = derivs;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += gram_length, deriv_ptr += 1) {

			sig_kernel_backprop_func(gram_ptr, deriv_ptr, out_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int sig_kernel(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid) noexcept {
		SAFE_CALL(sig_kernel_(gram, out, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CPSIG_API int batch_sig_kernel(double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs, bool return_grid) noexcept {
		SAFE_CALL(batch_sig_kernel_(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs, return_grid));
	}

	CPSIG_API int sig_kernel_backprop(double* gram, double* out, double deriv, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept {
		SAFE_CALL(sig_kernel_backprop_(gram, out, deriv, dimension, length1, length2, dyadic_order_1, dyadic_order_2));
	}

	CPSIG_API int batch_sig_kernel_backprop(double* gram, double* out, double* derivs, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_kernel_backprop_(gram, out, derivs, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs));
	}
}
