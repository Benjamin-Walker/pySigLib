#include "cppch.h"
#include "cpsig.h"
#include "cpSigKernel.h"
#include "macros.h"

extern "C" {

	CPSIG_API int sigKernelFloat(float* path1, float* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(sigKernel_<float>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int sigKernelDouble(double* path1, double* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(sigKernel_<double>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int sigKernelInt32(int32_t* path1, int32_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(sigKernel_<int32_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int sigKernelInt64(int64_t* path1, int64_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(sigKernel_<int64_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int batchSigKernelFloat(float* path1, float* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(batchSigKernel_<float>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int batchSigKernelDouble(double* path1, double* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(batchSigKernel_<double>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int batchSigKernelInt32(int32_t* path1, int32_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(batchSigKernel_<int32_t>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int batchSigKernelInt64(int64_t* path1, int64_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(batchSigKernel_<int64_t>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

}
