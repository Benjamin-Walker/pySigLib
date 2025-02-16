#pragma once
#include "cupch.h"
#include "cudaConstants.h"

//template<typename T>
//__global__ void getSigKernel_basic(
//	double* pdeGrid,
//	T* path1,
//	T* path2,
//	double* out,
//	uint64_t dimension,
//	uint64_t length1,
//	uint64_t length2,
//	uint64_t dyadicOrder1,
//	uint64_t dyadicOrder2
//) {
//	uint64_t blockId = blockIdx.x;
//	uint64_t threadId = threadIdx.x;
//
//	// Initialization of K array
//	for (uint64_t i = 0; i < dyadicLength1; ++i) {
//		pdeGrid[i * dyadicLength2] = 1.0; // Set K[i, 0] = 1.0
//	}
//
//	for (uint64_t i = 0; i < dyadicLength2; ++i) {
//		pdeGrid[i] = 1.0; // Set K[i, 0] = 1.0
//	}
//
//	for (uint64_t p = 2; p < numAntiDiag; ++p) { // First two antidiagonals are initialised to 1
//		uint64_t startj, endj;
//		if (dyadicLength1 > p) startj = 1ULL;
//		else startj = p - dyadicLength1 + 1;
//		if (dyadicLength2 > p) endj = p;
//		else endj = dyadicLength2;
//
//		uint64_t j = startj + threadId;
//
//		if (j < endj) {
//
//			uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
//			uint64_t ii = ((i - 1) >> dyadicOrder1) + 1;
//			uint64_t jj = ((j - 1) >> dyadicOrder2) + 1;
//
//			double deriv = 0;
//			for (uint64_t k = 0; k < dimension; ++k) {
//				deriv += (path1[ii * dimension + k] - path1[(ii - 1) * dimension + k]) * (path2[jj * dimension + k] - path2[(jj - 1) * dimension + k]);
//			}
//			deriv *= dyadicFrac;
//			double deriv2 = deriv * deriv * twelth;
//
//			pdeGrid[i * dyadicLength2 + j] = (pdeGrid[i * dyadicLength2 + j - 1] + pdeGrid[(i - 1) * dyadicLength2 + j]) * (
//				1. + 0.5 * deriv + deriv2) - pdeGrid[(i - 1) * dyadicLength2 + j - 1] * (1. - deriv2);
//
//		}
//		__syncthreads();
//	}
//
//	if (threadId == 0)
//		*out = pdeGrid[dyadicLength1 * dyadicLength2 - 1];
//}

template<typename T>
__host__ void getSigKernel(
	T* path1,
	T* path2,
	double* out,
	uint64_t batchSize,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	uint64_t dyadicLength1_,
	uint64_t dyadicLength2_
) {
	uint64_t numFull32Runs = (dyadicLength2_ - 1) / 32;
	uint64_t remainder = (dyadicLength2_ - 1) % 32;

	// Allocate initial condition
	double* ones = (double*)malloc(dyadicLength1_ * batchSize * sizeof(double));
	std::fill(ones, ones + dyadicLength1_ * batchSize, 1.);

	double* initialCondition;
	cudaMalloc((void**)&initialCondition, dyadicLength1_ * batchSize * sizeof(double));
	cudaMemcpy(initialCondition, ones, dyadicLength1_ * batchSize * sizeof(double), cudaMemcpyHostToDevice);
	free(ones);

	for (uint64_t i = 0; i < numFull32Runs; ++i) {
		goursatPde32 << <batchSize, 32 >> > (initialCondition, path1, path2, dimension, length1, length2, dyadicOrder1, dyadicOrder2, i);
		cudaDeviceSynchronize();
	}

	if (remainder)
		goursatPde32 << <batchSize, remainder >> > (initialCondition, path1, path2, dimension, length1, length2, dyadicOrder1, dyadicOrder2, numFull32Runs);

	for (uint64_t i = 0; i < batchSize; ++i)
		cudaMemcpy(out + i, initialCondition + (i+1) * dyadicLength1_ - 1, sizeof(double), cudaMemcpyDeviceToDevice);
	cudaFree(initialCondition);
}

template<typename T>
__global__ void goursatPde32(
	double* initialCondition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	T* path1,
	T* path2,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	uint64_t iteration
) {
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int numThreads = blockDim.x;

	double* initialCondition_ = initialCondition + blockId * dyadicLength1;
	T* path1_ = path1 + blockId * length1 * dimension;
	T* path2_ = path2 + blockId * length2 * dimension;

	__shared__ double diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused

	// Initialise to 1
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + threadId + 1] = 1.;

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prevPrevDiagIdx = 0;
	int prevDiagIdx = 33;
	int nextDiagIdx = 66;

	if (threadId == 0) {
		diagonals[prevPrevDiagIdx] = initialCondition_[0];
		diagonals[prevDiagIdx] = initialCondition_[1];
	}

	__syncthreads();

	for (uint64_t p = 2; p < numAntiDiag; ++p) { // First two antidiagonals are initialised to 1
		
		uint64_t startj, endj;
		if (dyadicLength1 > p) startj = 1ULL;
		else startj = p - dyadicLength1 + 1;
		if (numThreads + 1 > p) endj = p;
		else endj = numThreads + 1;

		uint64_t j = startj + threadId;

		if (j < endj) {

			// Make sure correct initial condition is filled in for first thread
			if (threadId == 0 && p < dyadicLength1) {
				diagonals[nextDiagIdx] = initialCondition_[p];
			}

			uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			uint64_t ii = ((i - 1) >> dyadicOrder1) + 1;
			uint64_t jj = ((j + iteration * 32 - 1) >> dyadicOrder2) + 1;

			double deriv = 0;
			for (uint64_t k = 0; k < dimension; ++k) {
				deriv += (path1_[ii * dimension + k] - path1_[(ii - 1) * dimension + k]) * (path2_[jj * dimension + k] - path2_[(jj - 1) * dimension + k]);
			}
			deriv *= dyadicFrac;
			double deriv2 = deriv * deriv * twelth;
			
			diagonals[nextDiagIdx + j] = (diagonals[prevDiagIdx + j] + diagonals[prevDiagIdx + j - 1]) * (
				1. + 0.5 * deriv + deriv2) - diagonals[prevPrevDiagIdx + j - 1] * (1. - deriv2);

		}
		// Wait for all threads to finish
		__syncthreads();

		// Overwrite initial condition with result
		// Safe to do since we won't be using initialCondition[p-numThreads] any more
		if (threadId == 0 && p >= numThreads && p - numThreads < dyadicLength1)
			initialCondition_[p - numThreads] = diagonals[nextDiagIdx + numThreads];

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prevPrevDiagIdx;
		prevPrevDiagIdx = prevDiagIdx;
		prevDiagIdx = nextDiagIdx;
		nextDiagIdx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}
}

template<typename T>
void sigKernelCUDA_(
	T* path1,
	T* path2,
	double* out,
	uint64_t batchSize,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	static const double twelth_ = 1. / 12;
	const uint64_t dyadicLength1_ = ((length1 - 1) << dyadicOrder1) + 1;
	const uint64_t dyadicLength2_ = ((length2 - 1) << dyadicOrder2) + 1;
	const uint64_t numAntiDiag_ = dyadicLength1_ + dyadicLength2_ - 1;
	const double dyadicFrac_ = 1. / (1ULL << (dyadicOrder1 + dyadicOrder2));

	// Allocate constant memory
	cudaMemcpyToSymbol(twelth, &twelth_, sizeof(double));
	cudaMemcpyToSymbol(dyadicLength1, &dyadicLength1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicLength2, &dyadicLength2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(numAntiDiag, &numAntiDiag_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicFrac, &dyadicFrac_, sizeof(double));
	
	uint64_t numFull32Runs = (dyadicLength2_ - 1) / 32;
	uint64_t remainder = (dyadicLength2_ - 1) % 32;
	
	getSigKernel(
		path1,
		path2,
		out,
		batchSize,
		dimension,
		length1,
		length2,
		dyadicOrder1,
		dyadicOrder2,
		dyadicLength1_,
		dyadicLength2_);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}
}


//template<typename T>
//void batchSigKernelCUDA_(
//	T* path1,
//	T* path2,
//	double* out,
//	uint64_t batchSize,
//	uint64_t dimension,
//	uint64_t length1,
//	uint64_t length2,
//	uint64_t dyadicOrder1,
//	uint64_t dyadicOrder2,
//	bool timeAug = false,
//	bool leadLag = false,
//	bool parallel = true
//) {
//	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
//
//	const uint64_t flatPathLength1 = dimension * length1;
//	const uint64_t flatPathLength2 = dimension * length2;
//	T* const dataEnd1 = path1 + flatPathLength1 * batchSize;
//
//	auto sigKernelFunc = [&](T* pathPtr, double* outPtr) {
//		Path<T> pathObj1(path1, dimension, length1, timeAug, leadLag);
//		Path<T> pathObj2(path2, dimension, length2, timeAug, leadLag);
//		getSigKernel_(pathObj1, pathObj2, outPtr, dyadicOrder1, dyadicOrder2);
//		};
//
//	if (parallel) {
//		multiThreadedBatch2(sigKernelFunc, path1, path2, out, batchSize, flatPathLength1, flatPathLength2, 1);
//	}
//	else {
//		for (T* path1Ptr = path1, path2Ptr = path2, outPtr = out;
//			path1Ptr < dataEnd1;
//			path1Ptr += flatPathLength1, path2Ptr += flatPathLength2, ++outPtr) {
//
//			sigKernelFunc(path1Ptr, path2Ptr, outPtr);
//		}
//	}
//	return;
//}