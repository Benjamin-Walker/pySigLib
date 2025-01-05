#pragma once
#include "cppch.h"
#include "cpsig.h"

inline int getMaxThreads() {
	static const int maxThreads = std::thread::hardware_concurrency();
	return maxThreads;
}

template<typename T, typename FN>
void multiThreadedBatch(FN& threadFunc, T* path, double* out, uint64_t batchSize, uint64_t flatPathLength, uint64_t resultLength) {
	const int maxThreads = getMaxThreads();
	const uint64_t threadPathStep = flatPathLength * maxThreads;
	const uint64_t threadResultStep = resultLength * maxThreads;
	T* const dataEnd = path + flatPathLength * batchSize;

	std::vector<std::thread> workers;

	auto batchThreadFunc = [&](T* pathPtr, double* outPtr) {
		T* pathPtr_;
		double* outPtr_;
		for (pathPtr_ = pathPtr, outPtr_ = outPtr;
			pathPtr_ < dataEnd;
			pathPtr_ += threadPathStep, outPtr_ += threadResultStep) {

			threadFunc(pathPtr_, outPtr_);
		}
		};

	int numThreads = 0;
	T* pathPtr;
	double* outPtr;
	for (pathPtr = path, outPtr = out;
		(numThreads < maxThreads) && (pathPtr < dataEnd);
		pathPtr += flatPathLength, outPtr += resultLength) {

		workers.emplace_back(batchThreadFunc, pathPtr, outPtr);
		++numThreads;
	}
	for (auto& w : workers) w.join();
}