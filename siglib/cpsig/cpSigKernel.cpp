#include "cppch.h"
#include "cpsig.h"
#include "cpSigKernel.h"
#include "macros.h"

void getSigKernelPreferSpeed_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	const uint64_t dimension,
	double* out,
	const uint64_t dyadicOrder1,
	const uint64_t dyadicOrder2
) {
	const double dyadicFrac = 1. / (1ULL << (dyadicOrder1 + dyadicOrder2));
	const double twelth = 1. / 12;

	// Dyadically refined grid dimensions
	const uint64_t gridSize1 = 1ULL << dyadicOrder1;
	const uint64_t gridSize2 = 1ULL << dyadicOrder2;
	const uint64_t dyadicLength1 = ((length1 - 1) << dyadicOrder1) + 1;
	const uint64_t dyadicLength2 = ((length2 - 1) << dyadicOrder2) + 1;

	// Allocate(flattened) PDE grid
	double* pdeGrid = (double*)malloc(dyadicLength1 * dyadicLength2 * sizeof(double));

	// Initialization of K array
	for (uint64_t i = 0; i < dyadicLength1; ++i) {
		pdeGrid[i * dyadicLength2] = 1.0; // Set K[i, 0] = 1.0
	}

	std::fill(pdeGrid, pdeGrid + dyadicLength2, 1.0); // Set K[0, j] = 1.0

	double* derivTerm1 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 + 0.5 * deriv + deriv2
	if (!derivTerm1) {
		throw std::bad_alloc();
		return;
	}
	double* derivTerm2 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 - deriv2
	if (!derivTerm2) {
		throw std::bad_alloc();
		return;
	}

	double* k11 = pdeGrid;
	double* k12 = k11 + 1;
	double* k21 = k11 + dyadicLength2;
	double* k22 = k21 + 1;

	for (uint64_t ii = 0; ii < length1 - 1; ++ii) {
		for (uint64_t m = 0; m < length2 - 1; ++m) {
			double deriv = gram[ii * (length2 - 1) + m];//dot_product(diff1Ptr, diff2Ptr, dimension);
			deriv *= dyadicFrac;
			double deriv2 = deriv * deriv * twelth;
			derivTerm1[m] = 1.0 + 0.5 * deriv + deriv2;
			derivTerm2[m] = 1.0 - deriv2;
		}

		for (uint64_t i = 0; i < gridSize1; ++i) {
			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				double t1 = derivTerm1[jj];
				double t2 = derivTerm2[jj];
				for (uint64_t j = 0; j < gridSize2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * t1 - *(k11++) * t2;
				}
			}
			++k11;
			++k12;
			++k21;
			++k22;
		}
	}

	*out = pdeGrid[dyadicLength1 * dyadicLength2 - 1];
	free(derivTerm1);
	free(derivTerm2);
	free(pdeGrid);
}

void sigKernel_(
	double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	getSigKernelPreferSpeed_(gram, length1, length2, dimension, out, dyadicOrder1, dyadicOrder2);
	/*if (preferSpeedOverMemory) {
		getSigKernelPreferSpeed_(path1, path2, gram, length1, length2, dimension, out, dyadicOrder1, dyadicOrder2);
	}
	else {
		getSigKernelPreferMemory_(path1, path2, gram, length1, length2, dimension, out, dyadicOrder1, dyadicOrder2);
	}*/
}

void batchSigKernel_(
	double* gram,
	double* out,
	uint64_t batchSize,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	bool parallel
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t gramLength = (length1 - 1) * (length2 - 1);
	double* const dataEnd1 = gram + gramLength * batchSize;

	std::function<void(double*, double*)> sigKernelFunc;

	sigKernelFunc = [&](double* gramPtr, double* outPtr) {
		getSigKernelPreferSpeed_(gramPtr, length1, length2, dimension, outPtr, dyadicOrder1, dyadicOrder2);
		};

	if (parallel) {
		multiThreadedBatch(sigKernelFunc, gram, out, batchSize, gramLength, 1);
	}
	else {
		double* gramPtr = gram;
		double* outPtr = out;
		for (;
			gramPtr < dataEnd1;
			gramPtr += gramLength, ++outPtr) {

			sigKernelFunc(gramPtr, outPtr);
		}
	}
	return;
}

//For convenience only
//extern "C" CPSIG_API void gram_(
//	double* path1,
//	double* path2,
//	double* out,
//	uint64_t batchSize,
//	uint64_t dimension,
//	uint64_t length1,
//	uint64_t length2
//) {
//	double* outPtr = out;
//
//	uint64_t flatPath1Length = length1 * dimension;
//	uint64_t flatPath2Length = length2 * dimension;
//
//	double* path1Start = path1;
//	double* path1End = path1 + flatPath1Length;
//
//	double* path2Start = path2;
//	double* path2End = path2 + flatPath2Length;
//
//	for (uint64_t b = 0; b < batchSize; ++b) {
//
//		for (double* path1Ptr = path1Start; path1Ptr < path1End - dimension; path1Ptr += dimension) {
//			for (double* path2Ptr = path2Start; path2Ptr < path2End - dimension; path2Ptr += dimension) {
//				*(outPtr++) = dot_product(path1Ptr + dimension, path2Ptr + dimension, dimension)
//					- dot_product(path1Ptr + dimension, path2Ptr, dimension)
//					- dot_product(path1Ptr, path2Ptr + dimension, dimension)
//					+ dot_product(path1Ptr, path2Ptr, dimension);
//			}
//		}
//
//		path1Start += flatPath1Length;
//		path1End += flatPath1Length;
//		path2Start += flatPath2Length;
//		path2End += flatPath2Length;
//	}
//}

extern "C" {

	/*CPSIG_API int sigKernelFloat(float* path1, float* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(sigKernel_<float>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, preferSpeedOverMemory));
	}

	CPSIG_API int sigKernelDouble(double* path1, double* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(sigKernel_<double>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, preferSpeedOverMemory));
	}

	CPSIG_API int sigKernelInt32(int32_t* path1, int32_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(sigKernel_<int32_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, preferSpeedOverMemory));
	}

	CPSIG_API int sigKernelInt64(int64_t* path1, int64_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(sigKernel_<int64_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, preferSpeedOverMemory));
	}

	CPSIG_API int batchSigKernelFloat(float* path1, float* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool parallel, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(batchSigKernel_<float>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2, parallel, preferSpeedOverMemory));
	}

	CPSIG_API int batchSigKernelDouble(double* path1, double* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool parallel, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(batchSigKernel_<double>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2, parallel, preferSpeedOverMemory));
	}

	CPSIG_API int batchSigKernelInt32(int32_t* path1, int32_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool parallel, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(batchSigKernel_<int32_t>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2, parallel, preferSpeedOverMemory));
	}

	CPSIG_API int batchSigKernelInt64(int64_t* path1, int64_t* path2, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool parallel, bool preferSpeedOverMemory) noexcept {
		SAFE_CALL(batchSigKernel_<int64_t>(path1, path2, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2, parallel, preferSpeedOverMemory));
	}*/

	CPSIG_API int sigKernel(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept {
		SAFE_CALL(sigKernel_(gram, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2));
	}

	CPSIG_API int batchSigKernel(double* gram, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool parallel) noexcept {
		SAFE_CALL(batchSigKernel_(gram, out, batchSize, dimension, length1, length2, dyadicOrder1, dyadicOrder2, parallel));
	}
}
