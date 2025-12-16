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

#include "multithreading.h"
#include "cp_tensor_poly.h"
#include "cp_signature.h"
#include "words.h"

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

template<std::floating_point T>
void log_sig_from_sig_(
	T* sig,
	uint64_t dimension,
	uint64_t degree
) {
	if (degree == 1)
		return;

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; i++)
		level_index[i] = level_index[i - 1] * dimension + 1;

	auto buff1_uptr = std::make_unique<T[]>(::sig_length(dimension, degree - 1));
	T* buff1 = buff1_uptr.get();
	std::fill(buff1, buff1 + ::sig_length(dimension, degree - 1), static_cast<T>(0.));

	auto buff2_uptr = std::make_unique<T[]>(::sig_length(dimension, degree));
	T* buff2 = buff2_uptr.get();
	std::fill(buff2, buff2 + ::sig_length(dimension, degree), static_cast<T>(0.));

	sig[0] = static_cast<T>(0.);

	for (uint64_t k = degree; k > 0; --k) {
		T constant = static_cast<T>(1.) / k;

		for (uint64_t target_level = 2; target_level <= 1 + degree - k; ++target_level) {

			std::fill(buff2 + level_index[target_level], buff2 + level_index[target_level + 1], static_cast<T>(0.));

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				uint64_t right_level = target_level - left_level;

				T* res_ptr = buff2 + level_index[target_level];
				T* const left_ptr_end = sig + level_index[left_level + 1];
				T* const right_ptr_end = buff1 + level_index[right_level + 1];
				for (T* left_ptr = sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr)
					for (T* right_ptr = buff1 + level_index[right_level]; right_ptr < right_ptr_end; ++right_ptr)
						*(res_ptr++) += *left_ptr * *right_ptr;
			}
		}
		if (k > 1) {
			for (uint64_t target_level = 1; target_level <= 1 + degree - k; ++target_level) {

				uint64_t target_level_size = level_index[target_level + 1] - level_index[target_level];
				T* const res_ptr = buff1 + level_index[target_level];
				T* const ptr_1 = sig + level_index[target_level];
				T* const ptr_2 = buff2 + level_index[target_level];

				for (uint64_t i = 0; i < target_level_size; ++i) {
					res_ptr[i] = constant * ptr_1[i] - ptr_2[i];
				}
			}
		}
	}
	for (uint64_t target_level = 2; target_level <= degree; ++target_level) {

		uint64_t target_level_size = level_index[target_level + 1] - level_index[target_level];
		T* const res_ptr = sig + level_index[target_level];
		T* const ptr = buff2 + level_index[target_level];

		for (uint64_t i = 0; i < target_level_size; ++i) {
			res_ptr[i] -= ptr[i];
		}
	}
}

template<typename T>
void log_sig_expanded(
	const T* path,
	T* out,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.
) {
	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);
	call_signature_horner_(path_obj, out, degree);
	log_sig_from_sig_<T>(out, path_obj.dimension(), degree);
}

template<typename T>
void log_sig_lyndon_words(
	const T* path,
	T* out,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.
) {
	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);
	uint64_t aug_dimension = path_obj.dimension();

	auto log_sig_uptr = std::make_unique<T[]>(::sig_length(aug_dimension, degree));
	T* log_sig = log_sig_uptr.get();

	call_signature_horner_(path_obj, log_sig, degree);
	log_sig_from_sig_<T>(log_sig, aug_dimension, degree);

	std::vector<uint64_t> lyndon = all_lyndon_idx(aug_dimension, degree);

	uint64_t m = lyndon.size();
	for (uint64_t i = 0; i < m; ++i) {
		out[i] = log_sig[lyndon[i]];
	}
}

template<std::floating_point T>
void get_log_sig_(
	const T* path,
	T* out,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.,
	int method = 0
)
{
	switch (method) {
	case 0:
		log_sig_expanded<T>(path, out, dimension, length, degree, time_aug, lead_lag, end_time);
		break;
	case 1:
		log_sig_lyndon_words<T>(path, out, dimension, length, degree, time_aug, lead_lag, end_time);
		break;
	}
}

template<std::floating_point T>
void log_signature_(
	const T* path,
	T* out,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.,
	int method = 0
)
{
	if (dimension == 0) { throw std::invalid_argument("log signature received path of dimension 0"); }
	if (degree == 0) { throw std::invalid_argument("log signature received degree 0"); }

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	if (path_obj.length() <= 1) {
		uint64_t result_length = method ? ::log_sig_length(path_obj.dimension(), degree) : ::sig_length(path_obj.dimension(), degree);
		std::fill(out, out + result_length, static_cast<T>(0.));
		return;
	}

	get_log_sig_<T>(path, out, dimension, length, degree, time_aug, lead_lag, end_time, method);
}

template<std::floating_point T>
void batch_log_signature_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.,
	int method = 0,
	int n_jobs = 1
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("signature received path of dimension 0"); }
	if (degree == 0) { throw std::invalid_argument("log signature received degree 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = method ? ::log_sig_length(dummy_path_obj.dimension(), degree) : ::sig_length(dummy_path_obj.dimension(), degree);

	if (dummy_path_obj.length() <= 1) {
		T* const out_end = out + result_length * batch_size;
		std::fill(out, out_end, static_cast<T>(0.));
		return;
	}

	//General case
	const uint64_t flat_path_length = dimension * length;
	const T* const data_end = path + flat_path_length * batch_size;

	std::function<void(const T*, T*)> log_sig_func;

	log_sig_func = [&](const T* path_ptr, T* out_ptr) {
		get_log_sig_<T>(path_ptr, out_ptr, dimension, length, degree, time_aug, lead_lag, end_time, method);
		};

	const T* path_ptr;
	T* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(log_sig_func, path, out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		for (path_ptr = path, out_ptr = out;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += result_length) {

			log_sig_func(path_ptr, out_ptr);
		}
	}
	return;
}
