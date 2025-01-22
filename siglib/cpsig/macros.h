#pragma once
#include <iostream>

#ifndef __APPLE__
	#define AVX
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
