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
#include "cp_log_signature.h"
#include "macros.h"

extern "C" {

	CPSIG_API int log_signature_f(const float* path, float* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, int method) noexcept {
		SAFE_CALL(log_signature_<float>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, method));
	}

	CPSIG_API int log_signature_d(const double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int method) noexcept {
		SAFE_CALL(log_signature_<double>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, method));
	}

	CPSIG_API int batch_log_signature_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, int method, int n_jobs) noexcept {
		SAFE_CALL(batch_log_signature_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, method, n_jobs));
	}

	CPSIG_API int batch_log_signature_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int method, int n_jobs) noexcept {
		SAFE_CALL(batch_log_signature_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, method, n_jobs));
	}

	CPSIG_API int log_signature_backprop_f(const float* path, float* out, float* log_sig_derivs, float* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, int method) noexcept {
		SAFE_CALL(log_sig_backprop_<float>(path, out, log_sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time, method));
	}

	CPSIG_API int log_signature_backprop_d(const double* path, double* out, double* log_sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int method) noexcept {
		SAFE_CALL(log_sig_backprop_<double>(path, out, log_sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time, method));
	}

}
