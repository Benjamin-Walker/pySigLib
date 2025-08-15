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

#if defined(_WIN32)
#include <Windows.h>
#include <strsafe.h>
#else
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <float.h>
#endif

#include <iostream>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"

template<typename FN, typename T, typename... Args>
void check_result(FN f, std::vector<T>& path, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    T* d_a;
    double* d_out;
    cudaMalloc(&d_a, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);

    f(d_a, d_out, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * true_.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);

    for (uint64_t i = 0; i < true_.size(); ++i)
        std::cout << true_[i] << " " << out[i] << std::endl;

    std::cout << -1. << " " << out[true_.size()] << std::endl;
}

double dot_product_(double* a, double* b, int n);

void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
);

template<typename T>
std::vector<T> test_data(uint64_t sz) {
    std::vector<T> data;
    data.reserve(sz);

    for (int i = 0; i < sz; i++) {
        data.push_back(static_cast<T>(i));
    }
    return data;
}

void print_header(std::string name);

template<typename FN, typename... Args>
void time_function(int num_runs, FN f, Args... args) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double avg_time = 0;
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;

    for (int i = 0; i < num_runs; ++i) {
        auto t1 = high_resolution_clock::now();
        f(args...);
        auto t2 = high_resolution_clock::now();
        std::cout << ".";
        duration<double, std::milli> ms_double = t2 - t1;
        double time = ms_double.count();

        avg_time += time;

        if (time < min_time)
            min_time = time;

        if (time > max_time)
            max_time = time;
    }

    avg_time /= num_runs;

    std::cout << "\n\nAvg run time: " << avg_time << "ms\n";
    std::cout << "Min run time: " << min_time << "ms\n";
    std::cout << "Max run time: " << max_time << "ms\n\n";
}
