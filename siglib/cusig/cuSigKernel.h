#pragma once
#include "cupch.h"
#include "cudaConstants.h"

template<typename T>
__global__ void goursatPde(
	double* initialCondition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	T* path1,
	T* path2
) {
	int blockId = blockIdx.x;

	double* initialCondition_ = initialCondition + blockId * dyadicLength1;
	T* path1_ = path1 + blockId * length1 * dimension;
	T* path2_ = path2 + blockId * length2 * dimension;

	__shared__ double diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused

	uint64_t numFullRuns = (dyadicLength2 - 1) / 32;
	uint64_t remainder = (dyadicLength2 - 1) % 32;

	for (int i = 0; i < numFullRuns; ++i)
		goursatPde32(initialCondition_, diagonals, path1_, path2_, i, 32);

	if (remainder)
		goursatPde32(initialCondition_, diagonals, path1_, path2_, numFullRuns, remainder);
}

template<typename T>
__device__ void goursatPde32(
	double* initialCondition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* diagonals,
	T* path1,
	T* path2,
	uint64_t iteration,
	int numThreads
) {
	int threadId = threadIdx.x;

	// Initialise to 1
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + threadId + 1] = 1.;

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prevPrevDiagIdx = 0;
	int prevDiagIdx = 33;
	int nextDiagIdx = 66;

	if (threadId == 0) {
		diagonals[prevPrevDiagIdx] = initialCondition[0];
		diagonals[prevDiagIdx] = initialCondition[1];
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
				diagonals[nextDiagIdx] = initialCondition[p];
			}

			uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			uint64_t ii = ((i - 1) >> dyadicOrder1) + 1;
			uint64_t jj = ((j + iteration * 32 - 1) >> dyadicOrder2) + 1;

			double deriv = 0;
			for (uint64_t k = 0; k < dimension; ++k) {
				deriv += (path1[ii * dimension + k] - path1[(ii - 1) * dimension + k]) * (path2[jj * dimension + k] - path2[(jj - 1) * dimension + k]);
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
			initialCondition[p - numThreads] = diagonals[nextDiagIdx + numThreads];

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
	uint64_t batchSize_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadicOrder1_,
	uint64_t dyadicOrder2_
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	static const double twelth_ = 1. / 12;
	const uint64_t dyadicLength1_ = ((length1_ - 1) << dyadicOrder1_) + 1;
	const uint64_t dyadicLength2_ = ((length2_ - 1) << dyadicOrder2_) + 1;
	const uint64_t numAntiDiag_ = dyadicLength1_ + dyadicLength2_ - 1;
	const double dyadicFrac_ = 1. / (1ULL << (dyadicOrder1_ + dyadicOrder2_));

	// Allocate constant memory
	cudaMemcpyToSymbol(dimension, &dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length1, &length1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length2, &length2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicOrder1, &dyadicOrder1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicOrder2, &dyadicOrder2_, sizeof(uint64_t));

	cudaMemcpyToSymbol(twelth, &twelth_, sizeof(double));
	cudaMemcpyToSymbol(dyadicLength1, &dyadicLength1_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicLength2, &dyadicLength2_, sizeof(uint64_t));
	cudaMemcpyToSymbol(numAntiDiag, &numAntiDiag_, sizeof(uint64_t));
	cudaMemcpyToSymbol(dyadicFrac, &dyadicFrac_, sizeof(double));

	// Allocate initial condition
	double* ones = (double*)malloc(dyadicLength1_ * batchSize_ * sizeof(double));
	std::fill(ones, ones + dyadicLength1_ * batchSize_, 1.);

	double* initialCondition;
	cudaMalloc((void**)&initialCondition, dyadicLength1_ * batchSize_ * sizeof(double));
	cudaMemcpy(initialCondition, ones, dyadicLength1_ * batchSize_ * sizeof(double), cudaMemcpyHostToDevice);
	free(ones);

	goursatPde << <batchSize_, 32 >> > (initialCondition, path1, path2);

	for (uint64_t i = 0; i < batchSize_; ++i)
		cudaMemcpy(out + i, initialCondition + (i + 1) * dyadicLength1_ - 1, sizeof(double), cudaMemcpyDeviceToDevice);
	cudaFree(initialCondition);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}
}