#pragma once
#include <cstdint>

#ifdef CPSIG_EXPORTS
#define CPSIG_API __declspec(dllexport)
#else
#define CPSIG_API __declspec(dllimport)
#endif

extern "C" {

	CPSIG_API uint64_t polyLength(uint64_t dimension, uint64_t degree);
	CPSIG_API double getPathElement(double* dataPtr, int dataLength, int dataDimension, int lengthIndex, int dimIndex);

	CPSIG_API void signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner); //bool timeAug = false, bool leadLag = false, bool horner = true);
	CPSIG_API void signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);
	CPSIG_API void signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);
	CPSIG_API void signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);

	CPSIG_API void batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true);
	CPSIG_API void batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true);
	CPSIG_API void batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true);
	CPSIG_API void batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true); 

}


