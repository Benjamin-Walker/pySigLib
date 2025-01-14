#pragma once
#include "cppch.h"
#include "cpsig.h"

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