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
#include "cpsig.h"
#include "macros.h"

// Calculate power
// Return 0 on error (integer overflow)
uint64_t power(uint64_t base, uint64_t exp) noexcept;

FORCE_INLINE void sig_combine_inplace_(double* sig1, double* sig2, uint64_t degree, uint64_t* level_index) {

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0L; --target_level) {
		for (int64_t left_level = target_level - 1L, right_level = 1L;
			left_level > 0L;
			--left_level, ++right_level) {

			double* result_ptr = sig1 + level_index[target_level];
			const double* left_ptr_upper_bound = sig1 + level_index[left_level + 1L];
			for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				const double* right_ptr_upper_bound = sig2 + level_index[right_level + 1L];
				for (double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*(result_ptr++) += (*left_ptr) * (*right_ptr);
				}
			}
		}

		//left_level = 0
		double* result_ptr = sig1 + level_index[target_level];
		const double* right_ptr_upper_bound = sig2 + level_index[target_level + 1L];
		for (double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
			*(result_ptr++) += *right_ptr;
		}
	}

}

// TODO: change to horner?
FORCE_INLINE void sig_uncombine_linear_inplace_(double* sig1, double* sig2, uint64_t degree, uint64_t* level_index) {
	//SIG2 MUST BE THE SIGNATURE OF A LINEAR SEGMENT

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0L; --target_level) {
		for (int64_t left_level = target_level - 1L, right_level = 1L;
			left_level > 0L;
			--left_level, ++right_level) {

			if (right_level % 2) {

				double* result_ptr = sig1 + level_index[target_level];
				const double* left_ptr_upper_bound = sig1 + level_index[left_level + 1L];
				for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const double* right_ptr_upper_bound = sig2 + level_index[right_level + 1L];
					for (double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) -= (*left_ptr) * (*right_ptr);
					}
				}
			}
			else {

				double* result_ptr = sig1 + level_index[target_level];
				const double* left_ptr_upper_bound = sig1 + level_index[left_level + 1L];
				for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const double* right_ptr_upper_bound = sig2 + level_index[right_level + 1L];
					for (double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) += (*left_ptr) * (*right_ptr);
					}
				}
			}
		}

		//left_level = 0
		if (target_level % 2) {
			double* result_ptr = sig1 + level_index[target_level];
			const double* right_ptr_upper_bound = sig2 + level_index[target_level + 1L];
			for (double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) -= *right_ptr;
			}
		}
		else {
			double* result_ptr = sig1 + level_index[target_level];
			const double* right_ptr_upper_bound = sig2 + level_index[target_level + 1L];
			for (double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) += *right_ptr;
			}
		}
	}

}

FORCE_INLINE void uncombine_sig_deriv(double* sig1, double* sig2, double* sig_concat_deriv, double* sig2_deriv, uint64_t dimension, uint64_t degree, uint64_t* level_index) {
	//sig1, sig2 are two signatures, and sig_concat is
	//the signature of the concatenated paths, sig1 * sig2.
	//sig_concat_deriv is dF/d(sig_concat)
	//This function computes dF/d(sig1) and dF/d(sig2) and writes these
	//into sig_concat_deriv and sig2_deriv respectively

	const uint64_t sig_len_ = sig_length(dimension, degree);
	std::memcpy(sig2_deriv, sig_concat_deriv, sig_len_ * sizeof(double));

	for (uint64_t level = degree; level > 0UL; --level) {
		for (uint64_t left_level = level - 1UL, right_level = 1UL; left_level > 0UL; --left_level, ++right_level) {
			double* result_ptr = sig_concat_deriv + level_index[level];

			for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != sig1 + level_index[left_level + 1UL]; ++left_ptr) {
				const double* right_ptr_upper_bound = sig2_deriv + level_index[right_level + 1UL];
				for (double* right_ptr = sig2_deriv + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*right_ptr += *(result_ptr++) * *left_ptr;
				}
			}
		}
	}


	for (uint64_t left_level = 1UL; left_level < degree; ++left_level) {
		for (uint64_t level = left_level + 1UL, right_level = 1UL; level <= degree; ++level, ++right_level) {
			double* result_ptr = sig_concat_deriv + level_index[level];
			for (double* left_ptr = sig_concat_deriv + level_index[left_level]; left_ptr != sig_concat_deriv + level_index[left_level + 1UL]; ++left_ptr) {
				const double* right_ptr_upper_bound = sig2 + level_index[right_level + 1UL];
				for (double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*left_ptr += *(result_ptr++) * *right_ptr;
				}
			}
		}
	}

}

FORCE_INLINE void linear_sig_deriv_to_increment_deriv(double* sig, double* sig_deriv, uint64_t dimension, uint64_t degree, uint64_t* level_index) {
	//Given sig is the signature of a line segment [a,b] and sig_deriv
	//is the derivative dF/d(sig), then this function computes dF/d(b-a)
	// and writes it into sig_deriv[1:1+dimension].

	for (uint64_t level = degree; level > 1UL; --level) {
		double one_over_level = 1. / level;
		for (uint64_t j = 0UL; j < level_index[level] - level_index[level - 1UL]; ++j) {
			for (uint64_t dd = 0UL; dd < dimension; ++dd) {
				const double ii = sig_deriv[level_index[level] + dd + dimension * j] * one_over_level;
				sig_deriv[level_index[level - 1UL] + j] += sig[1UL + dd] * ii;
				sig_deriv[1UL + dd] += sig[level_index[level - 1UL] + j] * ii;
			}
		}
	}
}