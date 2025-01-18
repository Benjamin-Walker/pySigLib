#pragma once

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
