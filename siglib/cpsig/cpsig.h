#pragma once
#include "cppch.h"

#if defined(CPSIG_EXPORTS)
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllexport)
	#elif defined (__GNUC__)
		#define CPSIG_API __attribute__((visibility("default")))
	#else
		#define CPSIG_API
	#endif
#else
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllimport)
	#elif defined (__GNUC__)
		#define CPSIG_API 
	#else
		#define CPSIG_API 
	#endif
#endif


extern "C" {

	CPSIG_API uint64_t polyLength(uint64_t dimension, uint64_t degree) noexcept;
	CPSIG_API double getPathElement(double* dataPtr, int dataLength, int dataDimension, int lengthIndex, int dimIndex);

	CPSIG_API int signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) noexcept; //bool timeAug = false, bool leadLag = false, bool horner = true);
	CPSIG_API int signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true) noexcept;
	CPSIG_API int signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true) noexcept;
	CPSIG_API int signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true) noexcept;

	CPSIG_API int batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true) noexcept;
	CPSIG_API int batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true) noexcept;
	CPSIG_API int batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true) noexcept;
	CPSIG_API int batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true) noexcept;

}


