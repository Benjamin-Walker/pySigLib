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
#include <iostream>

#ifndef __APPLE__
	//#define AVX
	#define ALIGNMENT 64

	#if ALIGNMENT>0
		#ifdef _MSC_VER
			#define ALIGNED_MALLOC(SZ) std::assume_aligned<ALIGNMENT>(_aligned_malloc(SZ, ALIGNMENT))
			#define ALIGNED_FREE(P) _aligned_free(P)
		#else
			#define ALIGNED_MALLOC(SZ) std::assume_aligned<ALIGNMENT>(std::aligned_alloc(ALIGNMENT, SZ))
			#define ALIGNED_FREE(P) std::free(P)
		#endif
	#else
		#define ALIGNED_MALLOC(SZ) malloc(SZ)
		#define ALIGNED_FREE(P) free(P)	
	#endif
#else
	#define ALIGNED_MALLOC(SZ) malloc(SZ)
	#define ALIGNED_FREE(P) free(P)	
#endif


#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

#define SAFE_CALL(function_call)                 \
    try {                                        \
        function_call;                           \
    }                                            \
    catch (std::bad_alloc&) {					 \
		std::cerr << "Failed to allocate memory";\
        return 1;                                \
    }                                            \
    catch (std::invalid_argument& e) {           \
		std::cerr << e.what();					 \
        return 2;                                \
    }                                            \
	catch (std::out_of_range& e) {			     \
		std::cerr << e.what();					 \
		return 3;                                \
	}  											 \
    catch (...) {                                \
		std::cerr << "Unknown exception";		 \
        return 4;                                \
    }                                            \
    return 0;
