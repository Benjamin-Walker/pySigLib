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

#include <string>
#include <limits>

#if defined(_WIN32) && !defined(__GNUC__)
#define CDECL_ __cdecl
#else
#define CDECL_
#endif

void load_cpsig(const std::string&);
void load_cusig(const std::string&);

void unload_cpsig();
void unload_cusig();

void get_cpsig_fn_ptrs();
void get_cusig_fn_ptrs();

using sig_length_fn = uint64_t(CDECL_*)(uint64_t, uint64_t);
using signature_double_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double, bool);
using signature_int32_fn = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double, bool);
using batch_signature_double_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, int);
using batch_signature_int32_fn = void(CDECL_*)(int*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, int);

using sig_kernel_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_kernel_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int, bool);
using sig_kernel_cuda_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_kernel_cuda_fn = void(CDECL_*)(double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_combine_fn = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, int);
using sig_backprop_double_fn = void(CDECL_*)(double*, double*, double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double);

using sig_kernel_backprop_fn = void(CDECL_*)(double*, double*, double, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using batch_sig_kernel_backprop_fn = void(CDECL_*)(double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int);

extern HMODULE cpsig;
extern HMODULE cusig;

extern sig_length_fn sig_length;
extern signature_double_fn signature_double;
extern signature_int32_fn signature_int32;
extern batch_signature_double_fn batch_signature_double;
extern batch_signature_int32_fn batch_signature_int32;
extern sig_kernel_fn sig_kernel;
extern batch_sig_kernel_fn batch_sig_kernel;
extern batch_sig_combine_fn batch_sig_combine;
extern sig_backprop_double_fn sig_backprop_double;

extern sig_kernel_cuda_fn sig_kernel_cuda;
extern batch_sig_kernel_cuda_fn batch_sig_kernel_cuda;

extern sig_kernel_backprop_fn sig_kernel_backprop;
extern batch_sig_kernel_backprop_fn batch_sig_kernel_backprop;

#if defined(_WIN32)
#define GET_FN_PTR ::GetProcAddress
#else
#define GET_FN_PTR dlsym
#endif
