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
#include "macros.h"
#include "cp_path_transforms.h"

void transform_path_backprop_(double* derivs, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) {
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.
	// Also note that if both time_aug and lead_lag are false, we still copy memory to data_out
	
	Path<double> dummy_path(nullptr, dimension, length, time_aug, lead_lag, end_time);
	const uint64_t transformed_length = dummy_path.length();
	const uint64_t transformed_dimension = dummy_path.dimension();
	const uint64_t dim_update = 2 * transformed_dimension - dimension;

	if (!(lead_lag || time_aug)) {
		std::memcpy(data_out, derivs, length * dimension * sizeof(double));
		return;
	}

	if (lead_lag) {
		for (int i = 0; i < length; ++i) {
			std::memcpy(data_out + i * dimension, derivs + 2 * i * transformed_dimension, dimension * sizeof(double));
		}

		double* out_ptr = data_out;
		double* derivs_ptr = derivs + transformed_dimension;
		for (int i = 0; i < length - 1; ++i) {
			for (int j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}

		out_ptr = data_out;
		derivs_ptr = derivs + dimension;
		for (int i = 0; i < length; ++i) {
			for (int j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}

		out_ptr = data_out + dimension;
		derivs_ptr = derivs + transformed_dimension + dimension;
		for (int i = 1; i < length; ++i) {
			for (int j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}
	}
	else {
		for (int i = 0; i < length; ++i) {
			std::memcpy(data_out + i * dimension, derivs + i * transformed_dimension, dimension * sizeof(double));
		}
	}
}

void batch_transform_path_backprop_(double* derivs, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs)
{
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.
	
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("transform_path_backprop received path of dimension 0"); }

	Path<double> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = length * dimension;

	const uint64_t flat_path_length = dummy_path_obj.length() * dummy_path_obj.dimension();
	double* const data_end = derivs + flat_path_length * batch_size;

	std::function<void(double*, double*)> transform_func;

	transform_func = [&](double* derivs_ptr, double* out_ptr) {
		transform_path_backprop_(derivs_ptr, out_ptr, dimension, length, time_aug, lead_lag, end_time);
		};

	double* derivs_ptr;
	double* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(transform_func, derivs, data_out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		for (derivs_ptr = derivs, out_ptr = data_out;
			derivs_ptr < data_end;
			derivs_ptr += flat_path_length, out_ptr += result_length) {

			transform_func(derivs_ptr, out_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int transform_path_float(float* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(transform_path_<float>(data_in, data_out, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int transform_path_double(double* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(transform_path_<double>(data_in, data_out, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int transform_path_int32(int32_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(transform_path_<int32_t>(data_in, data_out, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int transform_path_int64(int64_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(transform_path_<int64_t>(data_in, data_out, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int batch_transform_path_float(float* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<float>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_transform_path_double(double* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<double>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_transform_path_int32(int32_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<int32_t>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_transform_path_int64(int64_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<int64_t>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int transform_path_backprop(double* derivs, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(transform_path_backprop_(derivs, data_out, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int batch_transform_path_backprop(double* derivs, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_backprop_(derivs, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}
}