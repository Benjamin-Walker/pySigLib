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
#include "cp_tensor_poly.h"
#include "multithreading.h"
#include "macros.h"

uint64_t power(uint64_t base, uint64_t exp) noexcept {
    uint64_t result = 1;
    while (exp > 0UL) {
        if (exp % 2UL == 1UL) {
            const auto _res = result * base;
            if (_res < result)
                return 0UL; // overflow
            result = _res;
        }
        const auto _base = base * base;
        if (_base < base)
            return 0UL; // overflow
        base = _base;
        exp /= 2UL;
    }
    return result;
}

extern "C" CPSIG_API uint64_t poly_length(uint64_t dimension, uint64_t degree) noexcept {
    if (dimension == 0UL) {
        return 1UL;
    }
    else if (dimension == 1UL) {
        return degree + 1UL;
    }
    else {
        const auto pwr = power(dimension, degree + 1UL);
        if (pwr)
            return (pwr - 1UL) / (dimension - 1UL);
        else
            return 0UL; // overflow
    }
}


void poly_mult_(double* poly1, double* poly2, double* out, uint64_t dimension, uint64_t degree)
{
	if (dimension == 0) { throw std::invalid_argument("poly_mult received dimension 0"); }

	std::unique_ptr<uint64_t> level_index_uptr(new uint64_t[degree + 2]);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0UL;
	for (uint64_t i = 1UL; i <= degree + 1UL; i++)
		level_index[i] = level_index[i - 1UL] * dimension + 1;

    std::memcpy(out, poly1, sizeof(double) * level_index[degree + 1]);


	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0L; --target_level) {
		for (int64_t left_level = target_level - 1L, right_level = 1L;
			left_level > 0L;
			--left_level, ++right_level) {

			double* result_ptr = out + level_index[target_level];
			const double* left_ptr_upper_bound = poly1 + level_index[left_level + 1];
			for (double* left_ptr = poly1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				const double* right_ptr_upper_bound = poly2 + level_index[right_level + 1];
				for (double* right_ptr = poly2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*(result_ptr++) += (*left_ptr) * (*right_ptr);
				}
			}

		}

		//left_level = 0
		double* result_ptr = out + level_index[target_level];
		const double* right_ptr_upper_bound = poly2 + level_index[target_level + 1];
		for (double* right_ptr = poly2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
			*(result_ptr++) += *right_ptr;
		}
	}
}

void batch_poly_mult_(double* poly1, double* poly2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool parallel = true)
{
	if (dimension == 0) { throw std::invalid_argument("poly_mult received dimension 0"); }

	const uint64_t polylength = ::poly_length(dimension, degree);
	double* const poly1_end = poly1 + polylength * batch_size;
	double* const poly2_end = poly2 + polylength * batch_size;

	std::function<void(double*, double*, double*)> poly_mult_func;

	poly_mult_func = [&](double* poly1_ptr, double* poly2_ptr, double* out_ptr) {
		poly_mult_(poly1_ptr, poly2_ptr, out_ptr, dimension, degree);
		};

	if (parallel) {
		multi_threaded_batch_2(poly_mult_func, poly1, poly2, out, batch_size, polylength, polylength, polylength);
	}
	else {
		double* poly1_ptr = poly1;
		double* poly2_ptr = poly2;
		double* out_ptr = out;
		for (;
			poly1_ptr < poly1_end;
			poly1_ptr += polylength,
			poly2_ptr += polylength,
			out_ptr += polylength) {

			poly_mult_func(poly1_ptr, poly2_ptr, out_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int poly_mult(double* poly1, double* poly2, double* out, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(poly_mult_(poly1, poly2, out, dimension, degree));
	}

	CPSIG_API int batch_poly_mult(double* poly1, double* poly2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool parallel) noexcept {
		SAFE_CALL(batch_poly_mult_(poly1, poly2, out, batch_size, dimension, degree, parallel));
	}
}
