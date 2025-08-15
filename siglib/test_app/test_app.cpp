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

#include "dll_funcs.h"
#include "tests.h"


int main(int argc, char* argv[])
{
    std::string dir_path(".");

    if (argc >= 2) {
        dir_path = argv[1];
    }

    load_cpsig(dir_path);
    get_cpsig_fn_ptrs();

    load_cusig(dir_path);
    get_cusig_fn_ptrs();

    /*example_signature_double();
    example_signature_int32();
    example_batch_signature_double();
    example_batch_signature_int32();
    example_batch_signature_kernel();
    example_batch_signature_kernel_cuda();*/
    example_sig_backprop_double();

    unload_cpsig();
    unload_cusig();
}