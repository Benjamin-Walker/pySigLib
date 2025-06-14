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


inline unsigned int get_max_threads() {
	static const unsigned int max_threads = std::thread::hardware_concurrency();
	return max_threads;
}

template<typename T, typename FN>
void multi_threaded_batch(FN& thread_func, T* path, double* out, uint64_t batch_size, uint64_t flat_path_length, uint64_t result_length) {
	const unsigned int max_threads = get_max_threads();
	const uint64_t thread_path_step = flat_path_length * max_threads;
	const uint64_t thread_result_step = result_length * max_threads;
	T* const data_end = path + flat_path_length * batch_size;

	std::vector<std::thread> workers;

	auto batch_thread_func = [&](T* path_ptr, double* out_ptr) {
		double* out_ptr_ = out_ptr;
		for (T* path_ptr_ = path_ptr;
			path_ptr_ < data_end;
			path_ptr_ += thread_path_step, out_ptr_ += thread_result_step) {

			thread_func(path_ptr_, out_ptr_);
		}
		};

	unsigned int num_threads = 0;
	double* out_ptr = out;
	for (T* path_ptr = path;
		(num_threads < max_threads) && (path_ptr < data_end);
		path_ptr += flat_path_length, out_ptr += result_length) {

		workers.emplace_back(batch_thread_func, path_ptr, out_ptr);
		++num_threads;
	}

	for (auto& w : workers)
		w.join();
}


template<typename T, typename FN>
void multi_threaded_batch_2(FN& thread_func, T* path1, T* path2, double* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t result_length) {
	const unsigned int max_threads = get_max_threads();
	const uint64_t thread_path_step_1 = flat_path_length_1 * max_threads;
	const uint64_t thread_path_step_2 = flat_path_length_2 * max_threads;
	const uint64_t thread_result_step = result_length * max_threads;
	T* const data_end_1 = path1 + flat_path_length_1 * batch_size;

	std::vector<std::thread> workers;

	auto batch_thread_func = [&](T* path_ptr_1, T* path_ptr_2, double* out_ptr) {
		double* out_ptr_ = out_ptr;
		for (T* path1_ptr_ = path_ptr_1, *path2_ptr_ = path_ptr_2;
			path1_ptr_ < data_end_1;
			path1_ptr_ += thread_path_step_1, path2_ptr_ += thread_path_step_2, out_ptr_ += thread_result_step) {

			thread_func(path1_ptr_, path2_ptr_, out_ptr_);
		}
		};

	unsigned int num_threads = 0;
	double* out_ptr = out;
	for (T* path1_ptr = path1, *path2_ptr = path2;
		(num_threads < max_threads) && (path1_ptr < data_end_1);
		path1_ptr += flat_path_length_1, path2_ptr += flat_path_length_2, out_ptr += result_length) {

		workers.emplace_back(batch_thread_func, path1_ptr, path2_ptr, out_ptr);
		++num_threads;
	}

	for (auto& w : workers)
		w.join();
}
