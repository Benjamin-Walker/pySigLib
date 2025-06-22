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
			const double* left_ptr_upper_bound = sig1 + level_index[left_level + 1];
			for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				const double* right_ptr_upper_bound = sig2 + level_index[right_level + 1];
				for (double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*(result_ptr++) += (*left_ptr) * (*right_ptr);
				}
			}
		}

		//left_level = 0
		double* result_ptr = sig1 + level_index[target_level];
		const double* right_ptr_upper_bound = sig2 + level_index[target_level + 1];
		for (double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
			*(result_ptr++) += *right_ptr;
		}
	}

}