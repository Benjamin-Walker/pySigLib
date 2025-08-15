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

void signature_horner_step_(double* sig, double* increments, uint64_t dimension, uint64_t degree, uint64_t* level_index, double* horner_step)
{
	//Combines sig with the signature of a linear path given by increments using horner's algorithm

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 1L; --target_level) {

		double one_over_level = 1. / static_cast<double>(target_level);

		//left_level = 0
		//assign z / target_level to horner_step
		for (uint64_t i = 0UL; i < dimension; ++i)
			horner_step[i] = increments[i] * one_over_level;

		for (int64_t left_level = 1L, right_level = target_level - 1L;
			left_level < target_level - 1L;
			++left_level, --right_level) { //for each, add current left_level and times by z / right_level

			const uint64_t left_level_size = level_index[left_level + 1UL] - level_index[left_level];
			one_over_level = 1. / static_cast<double>(right_level);

			//Horner stuff
			//Add
			double* left_ptr_1 = sig + level_index[left_level];
			for (uint64_t i = 0UL; i < left_level_size; ++i) {
				horner_step[i] += *(left_ptr_1++);
			}

			//Multiply
#ifdef VEC
			double left_over_level;
			double* result_ptr = horner_step + level_index[left_level + 2UL] - level_index[left_level + 1UL] - dimension;
			for (double* left_ptr = horner_step + left_level_size - 1UL; left_ptr != horner_step - 1UL; --left_ptr, result_ptr -= dimension) {
				left_over_level = (*left_ptr) * one_over_level;
				vec_mult_assign(result_ptr, increments, left_over_level, dimension);
			}
#else
			double left_over_level;
			double* result_ptr = horner_step + level_index[left_level + 2UL] - level_index[left_level + 1UL];
			for (double* left_ptr = horner_step + left_level_size - 1UL; left_ptr != horner_step - 1UL; --left_ptr) {
				left_over_level = (*left_ptr) * one_over_level;
				for (double* right_ptr = increments + dimension - 1UL; right_ptr != increments - 1UL; --right_ptr) {
					*(--result_ptr) = left_over_level * (*right_ptr);
				}
			}
#endif
		}

		//======================= Do last iteration (left_level = target_level - 1) separately for speed, and add result straight into out

		const uint64_t left_level_size = level_index[target_level] - level_index[target_level - 1UL];

		//Horner stuff
		//Add
		double* left_ptr_1 = sig + level_index[target_level - 1UL];
		for (uint64_t i = 0UL; i < left_level_size; ++i) {
			horner_step[i] += *(left_ptr_1++);
		}

		//Multiply and add, writing straight into out
#ifdef VEC
		double* result_ptr = sig + level_index[target_level + 1] - dimension;
		for (double* left_ptr = horner_step + left_level_size - 1UL; left_ptr != horner_step - 1UL; --left_ptr, result_ptr -= dimension) {
			vec_mult_add(result_ptr, increments, *left_ptr, dimension);
		}
#else
		double* result_ptr = sig + level_index[target_level + 1];
		for (double* left_ptr = horner_step + left_level_size - 1UL; left_ptr != horner_step - 1UL; --left_ptr) {
			for (double* right_ptr = increments + dimension - 1UL; right_ptr != increments - 1UL; --right_ptr) {
				*(--result_ptr) += (*left_ptr) * (*right_ptr); //no one_over_level here, as right_level = 1
			}
		}
#endif
	}
	//Update target_level == 1
	for (uint64_t i = 0; i < dimension; ++i)
		sig[i + 1] += increments[i];
}

extern "C" {

	CPSIG_API int signature_float(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner) noexcept {
		SAFE_CALL(signature_<float>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CPSIG_API int signature_double(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner) noexcept {
		SAFE_CALL(signature_<double>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CPSIG_API int signature_int32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner) noexcept {
		SAFE_CALL(signature_<int32_t>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CPSIG_API int signature_int64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner) noexcept {
		SAFE_CALL(signature_<int64_t>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CPSIG_API int batch_signature_float(float* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, n_jobs));
	}

	CPSIG_API int batch_signature_double(double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, n_jobs));
	}

	CPSIG_API int batch_signature_int32(int32_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<int32_t>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, n_jobs));
	}

	CPSIG_API int batch_signature_int64(int64_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, int n_jobs) noexcept {
		SAFE_CALL(batch_signature_<int64_t>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, n_jobs));
	}

	CPSIG_API int sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(sig_backprop_<float>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CPSIG_API int sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(sig_backprop_<double>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CPSIG_API int sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(sig_backprop_<int32_t>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CPSIG_API int sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(sig_backprop_<int64_t>(path, out, sig_derivs, sig, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CPSIG_API int batch_sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<int32_t>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_backprop_<int64_t>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, n_jobs));
	}

}
