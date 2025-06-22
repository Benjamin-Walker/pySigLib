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

// test_app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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
#include <string>
#include <chrono>
#include <limits>

#include "cuda_runtime.h"

const double EPSILON = 1e-5;

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

double dot_product_(double* a, double* b, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += *(a + i) * *(b + i);
    }
    return res;
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
                *(out_ptr++) = dot_product_(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product_(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product_(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product_(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}

std::vector<double> test_data(uint64_t dimension, uint64_t length) {
    std::vector<double> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back((double)i);
    }
    return data;
}

std::vector<int> test_data_int(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

void printExample(std::string name) {
    std::cout << "\n//////////////////////////////////////////////" << std::endl;
    std::cout << "// Running Example " << name << std::endl;
    std::cout << "//////////////////////////////////////////////\n" << std::endl;
}

template<typename FN, typename... Args>
void time_function(int numRuns, FN f, Args... args) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double runningMinTime = DBL_MAX;

    for (int i = 0; i < numRuns; ++i) {
        auto t1 = high_resolution_clock::now();
        f(args...);
        auto t2 = high_resolution_clock::now();
        std::cout << ".";
        duration<double, std::milli> ms_double = t2 - t1;
        double time = ms_double.count();
        if (time < runningMinTime)
            runningMinTime = time;
    }

    std::cout << "\nMin run time: " << runningMinTime << "ms\n";
}


int main(int argc, char* argv[])
{
    std::string dir_path(".");

    if (argc >= 2) {
        dir_path = argv[1];
    }

    //////////////////////////////////////////////
    // Load cpsig
    //////////////////////////////////////////////
    std::string cpsig_path = dir_path + "\\cpsig.dll";

    std::cout << "Loading cpsig from " << cpsig_path << std::endl;


#if defined(_WIN32) && !defined __GNUC__

    HMODULE cpsig = ::LoadLibraryA(cpsig_path.c_str());
    if (cpsig == NULL) {
        // failed to load dll
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = ::GetLastError();

        ::FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        // Display the error message and exit the process

        lpDisplayBuf = (LPVOID)::LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 40) * sizeof(TCHAR));

        ::StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("Failed with error %d: %s"),
            dw, lpMsgBuf);

        //std::cerr << std::string((LPCTSTR)lpDisplayBuf);

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);
        return 1;
    }

#else

    // /home/shmelev/Projects/Daniils/pySigLib/siglib/dist/release
    void* cpsig = dlopen("/home/shmelev/Projects/Daniils/pySigLib/siglib/dist/release/libcpsig.so", RTLD_LAZY | RTLD_DEEPBIND);
    if (!cpsig) {
        fputs(dlerror(), stderr);
        return 1;
    }

#endif


    
    //////////////////////////////////////////////
    // Load cusig
    //////////////////////////////////////////////
    std::string cusig_path = dir_path + "\\cusig.dll";

    std::cout << "Loading cusig from " << cusig_path << std::endl;

    HMODULE cusig = ::LoadLibraryA(cusig_path.c_str());
    if (cpsig == NULL) {
        // failed to load dll
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = GetLastError();

        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        // Display the error message and exit the process

        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 40) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("failed with error %d: %s"),
            dw, lpMsgBuf);
        MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);
        return 1;
    }

    std::cout << "cusig loaded\n" << std::endl;
    

    //////////////////////////////////////////////
    // Getting functions pointers
    //////////////////////////////////////////////

#if defined(_WIN32) && !defined(__GNUC__)
#define CDECL_ __cdecl
#else
#define CDECL_
#endif

    using sig_length_FN              = uint64_t(CDECL_*)(uint64_t, uint64_t);
    using signature_double_FN         = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, bool);
    using signature_int32_FN          = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, bool, bool, bool);
    using batch_signature_double_FN   = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, bool, bool);
    using batch_signature_int32_FN    = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, bool, bool);

    using sig_kernel_FN               = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
    using batch_sig_kernel_FN         = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
    using sig_kernel_cuda_FN          = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    using batch_sig_kernel_cuda_FN    = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    using batch_sig_combine_FN          = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, bool);

#if defined(_WIN32)
#define GET_FN_PTR ::GetProcAddress
#else
#define GET_FN_PTR dlsym
#endif

#define GET_FN(NAME, LIBNAME) NAME ## _FN NAME = reinterpret_cast<NAME ## _FN>(GET_FN_PTR(LIBNAME, #NAME)); \
    if (!NAME) { std::cerr << "Failed to get the address of function " #NAME "\n"; return 2; }

    GET_FN(sig_length, cpsig);
    GET_FN(signature_double, cpsig);
    GET_FN(signature_int32, cpsig);
    GET_FN(batch_signature_double, cpsig);
    GET_FN(batch_signature_int32, cpsig);
    GET_FN(sig_kernel, cpsig);
    GET_FN(batch_sig_kernel, cpsig);
    GET_FN(batch_sig_combine, cpsig);

    GET_FN(sig_kernel_cuda, cusig);
    GET_FN(batch_sig_kernel_cuda, cusig);

    ////////////////////////////////////////////////
    //// Example Signature
    ////////////////////////////////////////////////

    //printExample("Signature");

    //uint64_t dimension1 = 5, length1 = 10000, degree1 = 4;
    //std::vector<double> path1 = test_data(dimension1, length1);

    //std::vector<double> out1;
    //uint64_t data_size1 = sig_length(dimension1, degree1);
    //out1.reserve(data_size1);

    //for (int i = 0; i < data_size1; i++) {
    //    out1.push_back(0.);
    //}

    //time_function(1000, signature_double, path1.data(), out1.data(), dimension1, length1, degree1, false, false, true);

    //std::cout << "done\n";

    ////////////////////////////////////////////////
    //// Example Signature Int
    ////////////////////////////////////////////////

    //printExample("Signature Int");

    //uint64_t dimension2 = dimension1, length2 = length1, degree2 = degree1;
    //std::vector<int> path2 = test_data_int(dimension2, length2);

    //std::vector<double> out2;
    //uint64_t data_size2 = sig_length(dimension2, degree2);
    //out2.reserve(data_size2);

    //for (int i = 0; i < data_size2; i++) {
    //    out2.push_back(0);
    //}

    //time_function(100, signature_int32, path2.data(), out2.data(), dimension2, length2, degree2, false, false, true);

    //std::cout << "done\n";

    //////////////////////////////////////////////
    // Example Batch Signature
    //////////////////////////////////////////////

    /*printExample("Batch Signature");

    uint64_t dimension3 = 10, length3 = 100, degree3 = 5, batch3 = 100;
    std::vector<double> data3;
    uint64_t sz3 = batch3 * dimension3 * length3;
    for (uint64_t i = 0; i < sz3; ++i) data3.push_back((double)i);


    std::vector<double> result3;
    uint64_t data_size3 = sig_length(dimension3, degree3) * batch3;
    result3.reserve(data_size3);

    for (int i = 0; i < data_size3; i++) {
        result3.push_back(0.);
    }

    time_function(10, batch_signature_double, data3.data(), result3.data(), batch3, dimension3, length3, degree3, false, false, true, false);

    std::cout << "done\n";*/

    ////////////////////////////////////////////////
    //// Example Batch Signature Parallel
    ////////////////////////////////////////////////

    /*printExample("Batch Signature Parallel");

    uint64_t dimension5 = 10, length5 = 1000, degree5 = 5, batch5 = 100;
    std::vector<double> data5;
    uint64_t sz5 = batch5 * dimension5 * length5;
    for (uint64_t i = 0; i < sz5; ++i) data5.push_back((double)i);

    std::vector<double> result5;
    uint64_t data_size5 = sig_length(dimension5, degree5) * batch5;

    result5.reserve(data_size5);

    for (int i = 0; i < data_size5; i++) {
        result5.push_back(0.);
    }

    time_function(10, batch_signature_double, data5.data(), result5.data(), batch5, dimension5, length5, degree5, false, false, true, true);

    std::cout << "done\n";*/


    ////////////////////////////////////////////////
    //// Example Batch Signature Int
    ////////////////////////////////////////////////

    //printExample("Batch Signature Int");

    //uint64_t dimension4 = 2, length4 = 4, degree4 = 2;
    //std::vector<int> data4 = { 0, 0, 1, 1, 2, 2, 3, 3,
    //    0, 0, 1, 2, 4, 4, 6, 8 };

    //std::vector<double> result4;
    //uint64_t data_size4 = sig_length(dimension4, degree4) * 2;
    //result4.reserve(data_size4);

    //for (int i = 0; i < data_size4; i++) {
    //    result4.push_back(0);
    //}

    //time_function(1, batch_signature_int32, data4.data(), result4.data(), 2, dimension4, length4, degree4, false, false, true);

    //std::cout << "done\n";

 //   //////////////////////////////////////////////
 //   // Example Signature Kernel
 //   //////////////////////////////////////////////

 //   printExample("Batch Signature Kernel");

 //   uint64_t dimension4 = 1000, length4 = 10000, batch4 = 1;
 //   std::vector<double> data4;
	//data4.resize(dimension4* length4 * batch4);
 //   for (uint64_t i = 0; i < dimension4 * length4 * batch4; ++i) data4[i] = (i % 2 ? 0.1 : 0.5);

 //   double* res = (double*)malloc(batch4 * sizeof(double));
 //   //batchSigKernelDouble(data4.data(), data4.data(), res, batch4, dimension4, length4, length4, 2, 2);

 //   std::vector<double> gram(length4 * length4);

 //   //gram_(data4.data(), data4.data(), gram.data(), batch4, dimension4, length4, length4);

 //   time_function(1, batch_sig_kernel, gram.data(), res, batch4, dimension4, length4, length4, 0,0, true);

 //   /*for (int i = 0; i < batch4; ++i)
 //       std::cout << res[i] << " done\n";*/

 //   free(res);

    ////////////////////////////////////////////////
    //// Example Signature Kernel
    ////////////////////////////////////////////////

    //printExample("Batch Signature Kernel");

    //uint64_t dimension4 = 100, length4 = 1000, batch4 = 10;
    //std::vector<double> data4;
    //data4.resize(dimension4* length4);
    //for (uint64_t i = 0; i < dimension4 * length4; ++i) data4[i] = (i % 2 ? 0.1 : 0.5);

    ////double res;
    //
    //double* d_a;
    //double* d_out;
    //cudaMalloc(&d_a, sizeof(double) * data4.size());
    //cudaMalloc(&d_out, sizeof(double) * batch4);

    //double* res2 = (double*)malloc(batch4 * sizeof(double));

    //// Copy data from the host to the device (CPU -> GPU)
    //cudaMemcpy(d_a, data4.data(), sizeof(double) * data4.size(), cudaMemcpyHostToDevice);

    //time_function(1, batch_sig_kernel_cuda, d_a, d_a, d_out, batch4, dimension4, length4, length4, 0,0);

    //cudaMemcpy(res2, d_out, sizeof(double) * batch4, cudaMemcpyDeviceToHost);

    //cudaFree(d_a);
    //cudaFree(d_out);

    //for (int i = 0; i < batch4; ++i)
    //    std::cout << res2[i] << " done\n";

    //free(res2);

    auto f = sig_kernel_cuda;
    uint64_t dimension = 2, length = 3;
    std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
    std::vector<double> true_sig = { 4.256702149748847 };
    std::vector<double> gram(length* length);
    gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
    check_result(f, gram, true_sig, dimension, length, length, 2, 2);

    return 0;
}