#pragma once
#include "cppch.h"
#include "cpsig.h"
#include "cpPath.h"
#include "cpVectorFuncs.h"

#define AVX
#define ALIGNMENT 64

#if ALIGNMENT>0
	#define ALIGNED_MALLOC(SZ) std::assume_aligned<ALIGNMENT>(_aligned_malloc(SZ, ALIGNMENT))
	#define ALIGNED_FREE(P) _aligned_free(P)
#else
	#define ALIGNED_MALLOC(SZ) malloc(SZ)
	#define ALIGNED_FREE(P) free(P)
#endif


template<typename T>
__forceinline void linearSignature_(Point<T>& prevPt, Point<T>& nextPt, double* out, uint64_t dimension, uint64_t degree, uint64_t* levelIndex)
{
	out[0] = 1.;

	for (uint64_t i = 0UL; i < dimension; ++i)
		out[i + 1] = static_cast<double>(nextPt[i] - prevPt[i]);
	
	double oneOverLevel;
	double leftOverLevel;

	for (uint64_t level = 2UL; level <= degree; ++level) {
		oneOverLevel = 1. / level;
		double* resultPtr = out + levelIndex[level];

		for (double* leftPtr = out + levelIndex[level - 1]; leftPtr != out + levelIndex[level]; ++leftPtr) {
			leftOverLevel = (*leftPtr) * oneOverLevel;
			for (double* rightPtr = out + levelIndex[1]; rightPtr != out + levelIndex[2]; ++rightPtr) {
				*(resultPtr++) = leftOverLevel * (*rightPtr);
			}
		}
	}
}

template<typename T>
void signatureNaive_(Path<T>& path, double* out, uint64_t degree)
{
	const uint64_t dimension = path.dimension();

	Point<T> prevPt(path.begin());
	Point<T> nextPt(path.begin());
	++nextPt;

	uint64_t* levelIndex = (uint64_t*)malloc(sizeof(uint64_t) * (degree + 2));
	levelIndex[0] = 0UL;
	for (uint64_t i = 1UL; i <= degree + 1UL; i++)
		levelIndex[i] = levelIndex[i - 1UL] * dimension + 1;

	linearSignature_(prevPt, nextPt, out, dimension, degree, levelIndex); //Zeroth step

	if (path.length() == 2UL) { free(levelIndex); return; }

	++prevPt;
	++nextPt;

	double* linearSignature = (double*)malloc(sizeof(double) * polyLength(dimension, degree));

	double* resultPtr;

	Point<T> lastPt(path.end());

	for (; nextPt != lastPt; ++prevPt, ++nextPt) { //Could start this loop from the second level onwards, first level is just increments

		linearSignature_(prevPt, nextPt, linearSignature, dimension, degree, levelIndex);

		for (int64_t targetLevel = degree; targetLevel > 0UL; --targetLevel) {
			for (int64_t leftLevel = targetLevel - 1UL, rightLevel = 1UL;
				leftLevel > 0UL;
				--leftLevel, ++rightLevel) {

				resultPtr = out + levelIndex[targetLevel];

				for (double* leftPtr = out + levelIndex[leftLevel]; leftPtr != out + levelIndex[leftLevel + 1]; ++leftPtr) {
					for (double* rightPtr = linearSignature + levelIndex[rightLevel]; rightPtr != linearSignature + levelIndex[rightLevel + 1]; ++rightPtr) {
						*(resultPtr++) += (*leftPtr) * (*rightPtr);
					}
				}
				
			}

			//leftLevel = 0
			resultPtr = out + levelIndex[targetLevel];

			for (double* rightPtr = linearSignature + levelIndex[targetLevel]; rightPtr != linearSignature + levelIndex[targetLevel + 1]; ++rightPtr) {
				*(resultPtr++) += *rightPtr;
			}
		}
	}
	free(linearSignature);
	free(levelIndex);
}

template<typename T>
void signatureHorner_(Path<T>& path, double* out, uint64_t degree)
{
	const uint64_t dimension = path.dimension();

	Point<T> prevPt(path.begin());
	Point<T> nextPt(path.begin());
	++nextPt;

	uint64_t* levelIndex = (uint64_t*)ALIGNED_MALLOC(sizeof(uint64_t) * (degree + 2));
	levelIndex[0] = 0UL;
	for (uint64_t i = 1UL; i <= degree + 1UL; i++)
		levelIndex[i] = levelIndex[i - 1UL] * dimension + 1UL;

	linearSignature_(prevPt, nextPt, out, dimension, degree, levelIndex); //Zeroth step

	if (path.length() == 2UL) { ALIGNED_FREE(levelIndex); return; }

	++prevPt;
	++nextPt;

	double* hornerStep = (double*)ALIGNED_MALLOC(sizeof(double) * (levelIndex[degree + 1UL] - levelIndex[degree])); //This will hold intermediary computations
	double* increments = (double*)ALIGNED_MALLOC(sizeof(double) * dimension);

	double* resultPtr;

	Point<T> lastPt(path.end());

	for (; nextPt != lastPt; ++prevPt, ++nextPt) { //Could start this loop from the second level onwards, first level is just increments
		for (uint64_t i = 0UL; i < dimension; ++i)
			increments[i] = nextPt[i] - prevPt[i];

		for (int64_t targetLevel = degree; targetLevel > 1UL; --targetLevel) {

			double oneOverLevel = 1. / targetLevel;
			const uint64_t targetLevelSize = levelIndex[targetLevel + 1UL] - levelIndex[targetLevel];

			//leftLevel = 0
			//assign z / targetLevel to hornerStep
			for (uint64_t i = 0UL; i < dimension; ++i)
				hornerStep[i] = increments[i] * oneOverLevel;

			for (int64_t leftLevel = 1UL, rightLevel = targetLevel - 1UL;
				leftLevel < targetLevel - 1UL; 
				++leftLevel, --rightLevel) { //for each, add current leftLevel and times by z / rightLevel

				const uint64_t leftLevelSize = levelIndex[leftLevel + 1UL] - levelIndex[leftLevel];
				oneOverLevel = 1. / rightLevel;

				//Horner stuff
				//Add
				double* leftPtr = out + levelIndex[leftLevel];
				for (uint64_t i = 0UL; i < leftLevelSize; ++i) {
					hornerStep[i] += *(leftPtr++);
				}

				//Multiply
#ifdef AVX
				double leftOverLevel;
				resultPtr = hornerStep + levelIndex[leftLevel + 2UL] - levelIndex[leftLevel + 1UL] - dimension;
				for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr, resultPtr -= dimension) {
					leftOverLevel = (*leftPtr) * oneOverLevel;
					vecMultAssign(resultPtr, increments, leftOverLevel, dimension);
				}
#else
				double leftOverLevel;
				resultPtr = hornerStep + levelIndex[leftLevel + 2UL] - levelIndex[leftLevel + 1UL];
				for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr) {
					leftOverLevel = (*leftPtr) * oneOverLevel;
					for (double* rightPtr = increments + dimension - 1UL; rightPtr != increments - 1UL; --rightPtr) {
						*(--resultPtr) = leftOverLevel * (*rightPtr);
					}
				}
#endif
			}

			//======================= Do last iteration (leftLevel = targetLevel - 1) separately for speed, and add result straight into out

			const uint64_t leftLevelSize = levelIndex[targetLevel] - levelIndex[targetLevel - 1UL];

			//Horner stuff
			//Add
			double* leftPtr = out + levelIndex[targetLevel - 1UL];
			for (uint64_t i = 0UL; i < leftLevelSize; ++i) {
				hornerStep[i] += *(leftPtr++);
			}

			//Multiply and add, writing straight into out
#ifdef AVX
			double leftOverLevel;
			resultPtr = out + levelIndex[targetLevel + 1] - dimension;
			for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr, resultPtr -= dimension) {
				leftOverLevel = (*leftPtr);
				//for (double* rightPtr = increments + dimension - 1UL; rightPtr != increments - 1UL; --rightPtr) {
				//	*(--resultPtr) += (*leftPtr) * (*rightPtr); //no oneOverLevel here, as rightLevel = 1
				//}
				vecMultAdd(resultPtr, increments, *leftPtr, dimension);
			}
#else
			double leftOverLevel;
			resultPtr = out + levelIndex[targetLevel + 1];
			for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr) {
				leftOverLevel = (*leftPtr);
				for (double* rightPtr = increments + dimension - 1UL; rightPtr != increments - 1UL; --rightPtr) {
					*(--resultPtr) += (*leftPtr) * (*rightPtr); //no oneOverLevel here, as rightLevel = 1
				}
			}
#endif
		}
		//Update targetLevel == 1
		for (uint64_t i = 0; i < dimension; ++i)
			out[i + 1] += increments[i];
	}
	ALIGNED_FREE(increments);
	ALIGNED_FREE(hornerStep);
	ALIGNED_FREE(levelIndex);
}

template<typename T>
void signature_(T* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true)
{
	if (dimension == 0) { throw std::invalid_argument("signature received path of dimension 0"); }

	Path<T> pathObj(path, dimension, length, timeAug, leadLag); //Work with pathObj to capture timeAug, leadLag transformations

	if (pathObj.length() <= 1) {
		out[0] = 1.;
		uint64_t resultLength = polyLength(pathObj.dimension(), degree);
		std::fill(out + 1, out + resultLength, 0.);
		return;
	}
	if (degree == 0) { out[0] = 1.; return; }
	if (degree == 1) {
		Point<T> firstPt = pathObj.begin();
		Point<T> lastPt = --pathObj.end();
		out[0] = 1.;
		uint64_t dimension_ = pathObj.dimension();
		for (uint64_t i = 0; i < dimension_; ++i)
			out[i + 1] = lastPt[i] - firstPt[i];
		return; 
	}

	if (horner)
		signatureHorner_(pathObj, out, degree);
	else
		signatureNaive_(pathObj, out, degree);
}

template<typename T>
void batchSignature_(T* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true, bool parallel = true)
{
	if (dimension == 0) { throw std::invalid_argument("signature received path of dimension 0"); }

	Path<T> dummyPathObj(nullptr, dimension, length, timeAug, leadLag); //Work with pathObj to capture timeAug, leadLag transformations

	const uint64_t resultLength = polyLength(dummyPathObj.dimension(), degree);

	if (dummyPathObj.length() <= 1) {
		double* const outEnd = out + resultLength * batchSize;
		std::fill(out, outEnd, 0.);
		for (double* outPtr = out;
			outPtr < outEnd;
			outPtr += resultLength) {
			outPtr[0] = 1.;
		}
		return;
	}
	if (degree == 0) { 
		std::fill(out, out + batchSize, 1.);
		return; }

	const uint64_t flatPathLength = dummyPathObj.dimension() * dummyPathObj.length();
	T* const dataEnd = path + flatPathLength * batchSize;

	if (degree == 1) {

		T* pathPtr;
		double* outPtr;

		if (parallel) {

			std::vector<std::thread> workers;

			auto threadFunc = [dimension, length, degree, timeAug, leadLag](T* pathPtr, double* outPtr) {
				Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
				Point<T> firstPt = pathObj.begin();
				Point<T> lastPt = --pathObj.end();
				outPtr[0] = 1.;
				for (uint64_t i = 0; i < pathObj.dimension(); ++i)
					outPtr[i + 1] = static_cast<double>(lastPt[i] - firstPt[i]);
				};

			for (pathPtr = path, outPtr = out;
				pathPtr < dataEnd;
				pathPtr += flatPathLength, outPtr += resultLength) {
				workers.emplace_back(threadFunc, pathPtr, outPtr);
			}
			for (auto& w : workers) w.join();
		}
		else {
			for (pathPtr = path, outPtr = out;
				pathPtr < dataEnd;
				pathPtr += flatPathLength, outPtr += resultLength) {

				Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
				Point<T> firstPt = pathObj.begin();
				Point<T> lastPt = --pathObj.end();
				outPtr[0] = 1.;
				for (uint64_t i = 0; i < pathObj.dimension(); ++i)
					outPtr[i + 1] = static_cast<double>(lastPt[i] - firstPt[i]);
			}
		}
		return;
	}

	auto f = horner ? &signatureHorner_<T> : &signatureNaive_<T>;

	T* pathPtr;
	double* outPtr;

	if (parallel) {
		std::vector<std::thread> workers;

		auto threadFunc = [f, dimension, length, degree, timeAug, leadLag](T* pathPtr, double* outPtr) {
			Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
			(*f)(pathObj, outPtr, degree);
			};

		for (pathPtr = path, outPtr = out;
			pathPtr < dataEnd;
			pathPtr += flatPathLength, outPtr += resultLength) {

			workers.emplace_back(threadFunc, pathPtr, outPtr);
		}
		for (auto& w : workers) w.join();
	}
	else {
		for (pathPtr = path, outPtr = out;
			pathPtr < dataEnd;
			pathPtr += flatPathLength, outPtr += resultLength) {

			Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
			(*f)(pathObj, outPtr, degree);
		}
	}
}