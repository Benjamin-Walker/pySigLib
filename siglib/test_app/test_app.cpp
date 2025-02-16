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

std::vector<double> testData(uint64_t dimension, uint64_t length) {
    std::vector<double> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back((double)i);
    }
    return data;
}

std::vector<int> testDataInt(uint64_t dimension, uint64_t length) {
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
void timeFunction(int numRuns, FN f, Args... args) {
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

    using polyLengthFN              = uint64_t(CDECL_*)(uint64_t, uint64_t);
    using signatureDoubleFN         = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, bool);
    using signatureInt32FN          = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, bool, bool, bool);
    using batchSignatureDoubleFN    = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, bool, bool);
    using batchSignatureInt32FN     = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, bool, bool);

    using sigKernelDoubleFN = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    using batchSigKernelDoubleFN = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    using sigKernelDoubleCUDAFN = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    using batchSigKernelDoubleCUDAFN = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

#if defined(_WIN32)
#define GET_FN_PTR ::GetProcAddress
#else
#define GET_FN_PTR dlsym
#endif

#define GET_FN(NAME, LIBNAME) NAME ## FN NAME = reinterpret_cast<NAME ## FN>(GET_FN_PTR(LIBNAME, #NAME)); \
    if (!NAME) { std::cerr << "Failed to get the address of function " #NAME "\n"; return 2; }

    GET_FN(polyLength, cpsig);
    GET_FN(signatureDouble, cpsig);
    GET_FN(signatureInt32, cpsig);
    GET_FN(batchSignatureDouble, cpsig);
    GET_FN(batchSignatureInt32, cpsig);
    GET_FN(sigKernelDouble, cpsig);
    GET_FN(batchSigKernelDouble, cpsig);

    GET_FN(sigKernelDoubleCUDA, cusig);
    GET_FN(batchSigKernelDoubleCUDA, cusig);

    ////////////////////////////////////////////////
    //// Example Signature
    ////////////////////////////////////////////////

    //printExample("Signature");

    //uint64_t dimension1 = 5, length1 = 10000, degree1 = 4;
    //std::vector<double> path1 = testData(dimension1, length1);

    //std::vector<double> out1;
    //uint64_t data_size1 = polyLength(dimension1, degree1);
    //out1.reserve(data_size1);

    //for (int i = 0; i < data_size1; i++) {
    //    out1.push_back(0.);
    //}

    //timeFunction(1000, signatureDouble, path1.data(), out1.data(), dimension1, length1, degree1, false, false, true);

    //std::cout << "done\n";

    ////////////////////////////////////////////////
    //// Example Signature Int
    ////////////////////////////////////////////////

    //printExample("Signature Int");

    //uint64_t dimension2 = dimension1, length2 = length1, degree2 = degree1;
    //std::vector<int> path2 = testDataInt(dimension2, length2);

    //std::vector<double> out2;
    //uint64_t data_size2 = polyLength(dimension2, degree2);
    //out2.reserve(data_size2);

    //for (int i = 0; i < data_size2; i++) {
    //    out2.push_back(0);
    //}

    //timeFunction(100, signatureInt32, path2.data(), out2.data(), dimension2, length2, degree2, false, false, true);

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
    uint64_t data_size3 = polyLength(dimension3, degree3) * batch3;
    result3.reserve(data_size3);

    for (int i = 0; i < data_size3; i++) {
        result3.push_back(0.);
    }

    timeFunction(10, batchSignatureDouble, data3.data(), result3.data(), batch3, dimension3, length3, degree3, false, false, true, false);

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
    uint64_t data_size5 = polyLength(dimension5, degree5) * batch5;

    result5.reserve(data_size5);

    for (int i = 0; i < data_size5; i++) {
        result5.push_back(0.);
    }

    timeFunction(10, batchSignatureDouble, data5.data(), result5.data(), batch5, dimension5, length5, degree5, false, false, true, true);

    std::cout << "done\n";*/


    ////////////////////////////////////////////////
    //// Example Batch Signature Int
    ////////////////////////////////////////////////

    //printExample("Batch Signature Int");

    //uint64_t dimension4 = 2, length4 = 4, degree4 = 2;
    //std::vector<int> data4 = { 0, 0, 1, 1, 2, 2, 3, 3,
    //    0, 0, 1, 2, 4, 4, 6, 8 };

    //std::vector<double> result4;
    //uint64_t data_size4 = polyLength(dimension4, degree4) * 2;
    //result4.reserve(data_size4);

    //for (int i = 0; i < data_size4; i++) {
    //    result4.push_back(0);
    //}

    //timeFunction(1, batchSignatureInt32, data4.data(), result4.data(), 2, dimension4, length4, degree4, false, false, true);

    //std::cout << "done\n";

    //////////////////////////////////////////////
    // Example Signature Kernel
    //////////////////////////////////////////////

    printExample("Batch Signature Kernel");

    uint64_t dimension4 = 5, length4 = 1000, batch4 = 100;
    std::vector<double> data4;
	data4.resize(dimension4* length4 * batch4);
    for (uint64_t i = 0; i < dimension4 * length4 * batch4; ++i) data4[i] = (i % 2 ? 0.1 : 0.5);

    double* res = (double*)malloc(batch4 * sizeof(double));
    //batchSigKernelDouble(data4.data(), data4.data(), res, batch4, dimension4, length4, length4, 2, 2);

    timeFunction(1, batchSigKernelDouble, data4.data(), data4.data(), res, batch4, dimension4, length4, length4, 0,0);

    /*for (int i = 0; i < batch4; ++i)
        std::cout << res[i] << " done\n";*/

    free(res);

    //////////////////////////////////////////////
    // Example Signature Kernel
    //////////////////////////////////////////////

    printExample("Batch Signature Kernel");

    /*uint64_t dimension4 = 5, length4 = 1000;
    std::vector<double> data4;
    data4.resize(dimension4* length4);
    for (uint64_t i = 0; i < dimension4 * length4; ++i) data4[i] = (i % 2 ? 0.1 : 0.5);*/

    //double res;
    
    double* d_a;
    double* d_out;
    cudaMalloc(&d_a, sizeof(double) * data4.size());
    cudaMalloc(&d_out, sizeof(double) * batch4);

    double* res2 = (double*)malloc(batch4 * sizeof(double));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, data4.data(), sizeof(double) * data4.size(), cudaMemcpyHostToDevice);

    timeFunction(1, batchSigKernelDoubleCUDA, d_a, d_a, d_out, batch4, dimension4, length4, length4, 0,0);

    cudaMemcpy(res2, d_out, sizeof(double) * batch4, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);

    /*for (int i = 0; i < batch4; ++i)
        std::cout << res2[i] << " done\n";*/

    free(res2);

    return 0;
}