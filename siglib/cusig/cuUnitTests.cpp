#include "CppUnitTest.h"
#include "cusig.h"
#include "cuframework.h"
#include "cuda_runtime.h"
#include <vector>


#define EPSILON 1e-10

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


std::vector<int> intTestData(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

//template<typename FN, typename T, typename... Args>
//void checkResult(FN f, std::vector<T>& path, std::vector<double>& true_, Args... args) {
//    std::vector<double> out;
//    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
//    out[true_.size()] = -1.;
//
//    f(path.data(), out.data(), args...);
//
//    for (uint64_t i = 0; i < true_.size(); ++i)
//        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);
//
//    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
//}

template<typename FN, typename T, typename... Args>
void checkResult2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<double>& true_, Args... args) {
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
        auto f = sigKernelDoubleCUDA;
        uint64_t dimension = 2, length = 3;
        std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
        std::vector<double> trueSig = { 4.256702149748847 };
        checkResult2(f, path, path, trueSig, dimension, length, length, 2, 2);
    }

    TEST_METHOD(ManualTest) {
        auto f = sigKernelDoubleCUDA;
        uint64_t dimension = 3, length = 4;
        std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
        std::vector<double> trueSig = { 2.1529809076880486 };
        checkResult2(f, path, path, trueSig, dimension, length, length, 2, 2);
    }
    };
}