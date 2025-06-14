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

#include "CppUnitTest.h"
#include "cusig.h"
#include "cuda_runtime.h"
#include <vector>


#define EPSILON 1e-10

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double dot_product(double* a, double* b, uint64_t N) {
    double out = 0;
    for (int i = 0; i < N; ++i)
        out += a[i] * b[i];
    return out;
}

void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* out_ptr = out;

    uint64_t flat_path1_length = length1 * dimension;
    uint64_t flat_path2_length = length2 * dimension;

    double* path1_start = path1;
    double* path1_end = path1 + flat_path1_length;

    double* path2_start = path2;
    double* path2_end = path2 + flat_path2_length;

    for (uint64_t b = 0; b < batch_size; ++b) {

        for (double* path1_ptr = path1_start; path1_ptr < path1_end - dimension; path1_ptr += dimension) {
            for (double* path2_ptr = path2_start; path2_ptr < path2_end - dimension; path2_ptr += dimension) {
                *(out_ptr++) = dot_product(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}


std::vector<int> int_test_data(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

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
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
void check_result_2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    T* d_a, * d_b;
    double * d_out;
    cudaMalloc(&d_a, sizeof(T) * path1.size());
    cudaMalloc(&d_b, sizeof(T) * path2.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, path1.data(), sizeof(T) * path1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, path2.data(), sizeof(T) * path2.size(), cudaMemcpyHostToDevice);

    f(d_a, d_b, d_out, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * true_.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
}

namespace MyTest
{
    TEST_CLASS(sigKernelTest) {
public:
    TEST_METHOD(LinearPathTest) {
        auto f = sig_kernel_cuda;
        uint64_t dimension = 2, length = 3;
        std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
        std::vector<double> true_sig = { 4.256702149748847 };
        std::vector<double> gram(length * length);
        gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
        check_result(f, gram, true_sig, dimension, length, length, 2, 2);
    }

    TEST_METHOD(ManualTest) {
        auto f = sig_kernel_cuda;
        uint64_t dimension = 3, length = 4;
        std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
        std::vector<double> true_sig = { 2.1529809076880486 };
        std::vector<double> gram(length * length);
        gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
        check_result(f, gram, true_sig, dimension, length, length, 2, 2);
    }
    };
}