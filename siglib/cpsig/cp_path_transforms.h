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
#include "cp_path.h"
#include "multithreading.h"

template<typename T>
void transform_path_(T* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time) {
	Path<T> path(data_in, dimension, length, time_aug, lead_lag, end_time);
	double* out_ptr = data_out;
	const uint64_t transformed_dimension = path.dimension();

	Point<T> end_pt = path.end();

	for (Point<T> pt = path.begin(); pt != end_pt; ++pt) {
		for (uint64_t i = 0; i < transformed_dimension; ++i) {
			(*out_ptr) = pt[i];
			++out_ptr;
		}
	}
}

template<typename T>
void batch_transform_path_(T* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("transform_path received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = dummy_path_obj.length() * dummy_path_obj.dimension();

	const uint64_t flat_path_length = dimension * length;
	T* const data_end = data_in + flat_path_length * batch_size;

	std::function<void(T*, double*)> transform_func;

	transform_func = [&](T* path_ptr, double* out_ptr) {
		transform_path_<T>(path_ptr, out_ptr, dimension, length, time_aug, lead_lag, end_time);
		};

	T* path_ptr;
	double* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(transform_func, data_in, data_out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		for (path_ptr = data_in, out_ptr = data_out;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += result_length) {

			transform_func(path_ptr, out_ptr);
		}
	}
	return;
}

void transform_path_backprop_(
	double* data_in,
	double* data_out,
	const uint64_t dimension,
	const uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time
);


void batch_transform_path_backprop_(
	double* data_in,
	double* data_out,
	const uint64_t batch_size,
	const uint64_t dimension,
	const uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time,
	int n_jobs
);
