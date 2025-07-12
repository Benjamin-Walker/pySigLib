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
void multi_threaded_batch(FN& thread_func, T* path, double* out, uint64_t batch_size, uint64_t flat_path_length, uint64_t result_length, int n_jobs) {
	if (n_jobs == 0)
		throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : get_max_threads() + 1 + n_jobs;
	if (max_threads < 1)
		throw std::invalid_argument("received negative n_jobs which is less than max_threads + 1; n_jobs too low");
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


template<typename S, typename T, typename FN>
void multi_threaded_batch_2(FN& thread_func, S* path1, T* path2, double* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t result_length, int n_jobs) {
	if (n_jobs == 0)
		throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : get_max_threads() + 1 + n_jobs;
	if (max_threads < 1)
		throw std::invalid_argument("received negative n_jobs which is less than max_threads + 1; n_jobs too low");
	const uint64_t thread_path_step_1 = flat_path_length_1 * max_threads;
	const uint64_t thread_path_step_2 = flat_path_length_2 * max_threads;
	const uint64_t thread_result_step = result_length * max_threads;
	S* const data_end_1 = path1 + flat_path_length_1 * batch_size;

	std::vector<std::thread> workers;

	auto batch_thread_func = [&](S* path_ptr_1, T* path_ptr_2, double* out_ptr) {
		double* out_ptr_ = out_ptr;
		S* path1_ptr_ = path_ptr_1;
		T* path2_ptr_ = path_ptr_2;
		for (;
			path1_ptr_ < data_end_1;
			path1_ptr_ += thread_path_step_1, path2_ptr_ += thread_path_step_2, out_ptr_ += thread_result_step) {

			thread_func(path1_ptr_, path2_ptr_, out_ptr_);
		}
		};

	unsigned int num_threads = 0;
	double* out_ptr = out;
	S* path1_ptr = path1;
	T* path2_ptr = path2;
	for (;
		(num_threads < max_threads) && (path1_ptr < data_end_1);
		path1_ptr += flat_path_length_1, path2_ptr += flat_path_length_2, out_ptr += result_length) {

		workers.emplace_back(batch_thread_func, path1_ptr, path2_ptr, out_ptr);
		++num_threads;
	}

	for (auto& w : workers)
		w.join();
}

template<typename R, typename S, typename T, typename FN>
void multi_threaded_batch_3(FN& thread_func, R* path1, S* path2, T* path3, double* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t flat_path_length_3, uint64_t result_length, int n_jobs) {
	if (n_jobs == 0)
		throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : get_max_threads() + 1 + n_jobs;
	if (max_threads < 1)
		throw std::invalid_argument("received negative n_jobs which is less than max_threads + 1; n_jobs too low");
	const uint64_t thread_path_step_1 = flat_path_length_1 * max_threads;
	const uint64_t thread_path_step_2 = flat_path_length_2 * max_threads;
	const uint64_t thread_path_step_3 = flat_path_length_3 * max_threads;
	const uint64_t thread_result_step = result_length * max_threads;
	R* const data_end_1 = path1 + flat_path_length_1 * batch_size;

	std::vector<std::thread> workers;

	auto batch_thread_func = [&](R* path_ptr_1, S* path_ptr_2, T* path_ptr_3, double* out_ptr) {
		double* out_ptr_ = out_ptr;
		R* path1_ptr_ = path_ptr_1;
		S* path2_ptr_ = path_ptr_2;
		T* path3_ptr_ = path_ptr_3;
		for (;
			path1_ptr_ < data_end_1;
			path1_ptr_ += thread_path_step_1, path2_ptr_ += thread_path_step_2, path3_ptr_ += thread_path_step_3, out_ptr_ += thread_result_step) {

			thread_func(path1_ptr_, path2_ptr_, path3_ptr_, out_ptr_);
		}
		};

	unsigned int num_threads = 0;
	double* out_ptr = out;
	R* path1_ptr = path1;
	S* path2_ptr = path2;
	T* path3_ptr = path3;
	for (;
		(num_threads < max_threads) && (path1_ptr < data_end_1);
		path1_ptr += flat_path_length_1, path2_ptr += flat_path_length_2, path3_ptr += flat_path_length_3, out_ptr += result_length) {

		workers.emplace_back(batch_thread_func, path1_ptr, path2_ptr, path3_ptr, out_ptr);
		++num_threads;
	}

	for (auto& w : workers)
		w.join();
}
