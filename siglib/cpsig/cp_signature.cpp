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
#include "cp_signature.h"
#include "macros.h"

template class Path<float>;
template class Path<double>;
template class Path<int32_t>;
template class Path<int64_t>;

template<typename T>
PointImpl<T>* Path<T>::point_impl_factory(uint64_t index) const {
	if (!_time_aug && !_lead_lag)
		return new PointImpl(this, index);
	else if (_time_aug && !_lead_lag)
		return new PointImplTimeAug(this, index);
	else if (!_time_aug && _lead_lag)
		return new PointImplLeadLag(this, index);
	else
		return new PointImplTimeAugLeadLag(this, index);
}

extern "C" {

	CPSIG_API int signature_float(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner) noexcept {
		SAFE_CALL(signature_<float>(path, out, dimension, length, degree, time_aug, lead_lag, horner));
	}

	CPSIG_API int signature_double(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner) noexcept {
		SAFE_CALL(signature_<double>(path, out, dimension, length, degree, time_aug, lead_lag, horner));
	}

	CPSIG_API int signature_int32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner) noexcept {
		SAFE_CALL(signature_<int32_t>(path, out, dimension, length, degree, time_aug, lead_lag, horner));
	}

	CPSIG_API int signature_int64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner) noexcept {
		SAFE_CALL(signature_<int64_t>(path, out, dimension, length, degree, time_aug, lead_lag, horner));
	}

	CPSIG_API int batch_signature_float(float* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, horner, n_jobs));
	}

	CPSIG_API int batch_signature_double(double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, horner, n_jobs));
	}

	CPSIG_API int batch_signature_int32(int32_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<int32_t>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, horner, n_jobs));
	}

	CPSIG_API int batch_signature_int64(int64_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<int64_t>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, horner, n_jobs));
	}

	CPSIG_API int sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(sig_backprop_<float>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag));
	}

	CPSIG_API int sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(sig_backprop_<double>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag));
	}

	CPSIG_API int sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(sig_backprop_<int32_t>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag));
	}

	CPSIG_API int sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(sig_backprop_<int64_t>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag));
	}

	CPSIG_API int batch_sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<int32_t>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<int64_t>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, n_jobs));
	}

}
