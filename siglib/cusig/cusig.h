#pragma once

#ifdef CUSIG_EXPORTS
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" CUSIG_API void cusig_hello_world(const long x);

extern "C" {
	CUSIG_API int sigKernelFloatCUDA(float* path1, float* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int sigKernelDoubleCUDA(double* path1, double* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int sigKernelInt32CUDA(int32_t* path1, int32_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int sigKernelInt64CUDA(int64_t* path1, int64_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;

	CUSIG_API int batchSigKernelFloatCUDA(float* path1, float* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int batchSigKernelDoubleCUDA(double* path1, double* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int batchSigKernelInt32CUDA(int32_t* path1, int32_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int batchSigKernelInt64CUDA(int64_t* path1, int64_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
}
