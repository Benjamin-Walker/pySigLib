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
    while (exp > 0) {
        if (exp % 2 == 1) {
            const auto _res = result * base;
            if (_res < result)
                return 0; // overflow
            result = _res;
        }
        const auto _base = base * base;
        if (_base < base)
            return 0; // overflow
        base = _base;
        exp /= 2;
    }
    return result;
}

extern "C" CPSIG_API uint64_t sig_length(uint64_t dimension, uint64_t degree) noexcept {
    if (dimension == 0) {
        return 1;
    }
    else if (dimension == 1) {
        return degree + 1;
    }
    else {
        const auto pwr = power(dimension, degree + 1);
        if (pwr)
            return (pwr - 1) / (dimension - 1);
        else
            return 0; // overflow
    }
}

extern "C" {

	CPSIG_API int sig_combine_float(const float* sig1, const float* sig2, float* out, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(sig_combine_<float>(sig1, sig2, out, dimension, degree));
	}

    CPSIG_API int sig_combine_double(const double* sig1, const double* sig2, double* out, uint64_t dimension, uint64_t degree) noexcept {
        SAFE_CALL(sig_combine_<double>(sig1, sig2, out, dimension, degree));
    }

    CPSIG_API int batch_sig_combine_float(const float* sig1, const float* sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
        SAFE_CALL(batch_sig_combine_<float>(sig1, sig2, out, batch_size, dimension, degree, n_jobs));
    }

	CPSIG_API int batch_sig_combine_double(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_<double>(sig1, sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int sig_combine_backprop_float(const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(sig_combine_backprop_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, dimension, degree));
	}

    CPSIG_API int sig_combine_backprop_double(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t dimension, uint64_t degree) noexcept {
        SAFE_CALL(sig_combine_backprop_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, dimension, degree));
    }

	CPSIG_API int batch_sig_combine_backprop_float(const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_backprop_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, n_jobs));
	}

    CPSIG_API int batch_sig_combine_backprop_double(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
        SAFE_CALL(batch_sig_combine_backprop_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, n_jobs));
    }
}
