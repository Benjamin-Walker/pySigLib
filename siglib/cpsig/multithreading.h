#pragma once
#include "cppch.h"


inline unsigned int getMaxThreads() {
	static const unsigned int maxThreads = std::thread::hardware_concurrency();
	return maxThreads;
}

template<typename T, typename FN>
void multiThreadedBatch(FN& threadFunc, T* path, double* out, uint64_t batchSize, uint64_t flatPathLength, uint64_t resultLength) {
	const unsigned int maxThreads = getMaxThreads();
	const uint64_t threadPathStep = flatPathLength * maxThreads;
	const uint64_t threadResultStep = resultLength * maxThreads;
	T* const dataEnd = path + flatPathLength * batchSize;

	std::vector<std::thread> workers;

	auto batchThreadFunc = [&](T* pathPtr, double* outPtr) {
		double* outPtr_ = outPtr;
		for (T* pathPtr_ = pathPtr;
			pathPtr_ < dataEnd;
			pathPtr_ += threadPathStep, outPtr_ += threadResultStep) {

			threadFunc(pathPtr_, outPtr_);
		}
		};

	unsigned int numThreads = 0;
	double* outPtr = out;
	for (T* pathPtr = path;
		(numThreads < maxThreads) && (pathPtr < dataEnd);
		pathPtr += flatPathLength, outPtr += resultLength) {

		workers.emplace_back(batchThreadFunc, pathPtr, outPtr);
		++numThreads;
	}

	for (auto& w : workers)
		w.join();
}


template<typename T, typename FN>
void multiThreadedBatch2(FN& threadFunc, T* path1, T* path2, double* out, uint64_t batchSize, uint64_t flatPathLength1, uint64_t flatPathLength2, uint64_t resultLength) {
	const unsigned int maxThreads = getMaxThreads();
	const uint64_t threadPathStep1 = flatPathLength1 * maxThreads;
	const uint64_t threadPathStep2 = flatPathLength2 * maxThreads;
	const uint64_t threadResultStep = resultLength * maxThreads;
	T* const dataEnd1 = path1 + flatPathLength1 * batchSize;

	std::vector<std::thread> workers;

	auto batchThreadFunc = [&](T* pathPtr1, T* pathPtr2, double* outPtr) {
		double* outPtr_ = outPtr;
		for (T* path1Ptr_ = pathPtr1, *path2Ptr_ = pathPtr2;
			path1Ptr_ < dataEnd1;
			path1Ptr_ += threadPathStep1, path2Ptr_ += threadPathStep2, outPtr_ += threadResultStep) {

			threadFunc(path1Ptr_, path2Ptr_, outPtr_);
		}
		};

	unsigned int numThreads = 0;
	double* outPtr = out;
	for (T* path1Ptr = path1, *path2Ptr = path2;
		(numThreads < maxThreads) && (path1Ptr < dataEnd1);
		path1Ptr += flatPathLength1, path2Ptr += flatPathLength2, outPtr += resultLength) {

		workers.emplace_back(batchThreadFunc, path1Ptr, path2Ptr, outPtr);
		++numThreads;
	}

	for (auto& w : workers)
		w.join();
}
