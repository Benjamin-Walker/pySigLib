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

#include <iostream>
#include <stdexcept>

#include "dll_funcs.h"

void load_cpsig(const std::string& dir_path) {
    //////////////////////////////////////////////
    // Load cpsig
    //////////////////////////////////////////////
    std::string cpsig_path = dir_path + "\\cpsig.dll";

    std::cout << "Loading cpsig from " << cpsig_path << std::endl;


#if defined(_WIN32) && !defined __GNUC__

    cpsig = ::LoadLibraryA(cpsig_path.c_str());
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
        throw std::runtime_error("Failed to load cpsig");
        //return 1;
    }

#else

    // /home/shmelev/Projects/Daniils/pySigLib/siglib/dist/release
    void* cpsig = dlopen("/home/shmelev/Projects/Daniils/pySigLib/siglib/dist/release/libcpsig.so", RTLD_LAZY | RTLD_DEEPBIND);
    if (!cpsig) {
        fputs(dlerror(), stderr);
        return 1;
    }

#endif

}

void load_cusig(const std::string& dir_path) {
    //////////////////////////////////////////////
    // Load cusig
    //////////////////////////////////////////////
    std::string cusig_path = dir_path + "\\cusig.dll";

    std::cout << "Loading cusig from " << cusig_path << std::endl;

    cusig = ::LoadLibraryA(cusig_path.c_str());
    if (cusig == NULL) {
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
        throw std::runtime_error("Failed to load cusig");
    }

    std::cout << "cusig loaded\n" << std::endl;
}

void unload_cpsig() {
#if defined(_WIN32) && !defined __GNUC__
    if (cpsig)
        ::FreeLibrary(cpsig);
    cpsig = nullptr;
#else
    //TODO
#endif
}

void unload_cusig() {
#if defined(_WIN32) && !defined __GNUC__
    if (cusig)
        ::FreeLibrary(cusig);
    cusig = nullptr;
#else
    //TODO
#endif
}

//////////////////////////////////////////////
// Getting functions pointers
//////////////////////////////////////////////

#define GET_FN(NAME, LIBNAME) NAME = reinterpret_cast<NAME ## _fn>(GET_FN_PTR(LIBNAME, #NAME)); \
    if (!NAME) throw std::runtime_error("Failed to get the address of function " #NAME " from " #LIBNAME " library." );

HMODULE cpsig = nullptr;
HMODULE cusig = nullptr;

sig_length_fn sig_length = nullptr;
signature_double_fn signature_double = nullptr;
signature_int32_fn signature_int32 = nullptr;
batch_signature_double_fn batch_signature_double = nullptr;
batch_signature_int32_fn batch_signature_int32 = nullptr;
sig_kernel_fn sig_kernel = nullptr;
batch_sig_kernel_fn batch_sig_kernel = nullptr;
batch_sig_combine_fn batch_sig_combine = nullptr;
sig_backprop_double_fn sig_backprop_double = nullptr;

sig_kernel_cuda_fn sig_kernel_cuda = nullptr;
batch_sig_kernel_cuda_fn batch_sig_kernel_cuda = nullptr;

sig_kernel_backprop_fn sig_kernel_backprop = nullptr;
batch_sig_kernel_backprop_fn batch_sig_kernel_backprop = nullptr;

sig_kernel_backprop_cuda_fn sig_kernel_backprop_cuda = nullptr;
batch_sig_kernel_backprop_cuda_fn batch_sig_kernel_backprop_cuda = nullptr;


void get_cpsig_fn_ptrs()
{
    GET_FN(sig_length, cpsig);
    GET_FN(signature_double, cpsig);
    GET_FN(signature_int32, cpsig);
    GET_FN(batch_signature_double, cpsig);
    GET_FN(batch_signature_int32, cpsig);
    GET_FN(sig_kernel, cpsig);
    GET_FN(batch_sig_kernel, cpsig);
    GET_FN(batch_sig_combine, cpsig);
    GET_FN(sig_backprop_double, cpsig);
    GET_FN(sig_kernel_backprop, cpsig);
    GET_FN(batch_sig_kernel_backprop, cpsig);
}

void get_cusig_fn_ptrs()
{
    GET_FN(sig_kernel_cuda, cusig);
    GET_FN(batch_sig_kernel_cuda, cusig);
    GET_FN(batch_sig_kernel_backprop_cuda, cusig);
}