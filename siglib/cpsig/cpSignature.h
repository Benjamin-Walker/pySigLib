#pragma once
#include "cppch.h"

#include "multithreading.h"

#include "cpPath.h"
#include "macros.h"
#ifdef AVX
#include "cpVectorFuncs.h"
#endif


template<typename T>
FORCE_INLINE void linearSignature_(Point<T>& startPt, Point<T>& endPt, double* out, uint64_t dimension, uint64_t degree, uint64_t* levelIndex)
{
	//Computes the signature of a linear segment joining startPt and endPt
	out[0] = 1.;

	for (uint64_t i = 0UL; i < dimension; ++i)
		out[i + 1] = static_cast<double>(endPt[i] - startPt[i]);
	
	double oneOverLevel;
	double leftOverLevel;

	for (uint64_t level = 2UL; level <= degree; ++level) {
		oneOverLevel = 1. / static_cast<double>(level);
		double* resultPtr = out + levelIndex[level];

		for (double* leftPtr = out + levelIndex[level - 1]; leftPtr != out + levelIndex[level]; ++leftPtr) {
			leftOverLevel = (*leftPtr) * oneOverLevel;
			for (double* rightPtr = out + 1; rightPtr != out + dimension + 1; ++rightPtr) {
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

	double* linearSignature = (double*)malloc(sizeof(double) * ::polyLength(dimension, degree));

	Point<T> lastPt(path.end());

	for (; nextPt != lastPt; ++prevPt, ++nextPt) {

		linearSignature_(prevPt, nextPt, linearSignature, dimension, degree, levelIndex);

		for (int64_t targetLevel = static_cast<int64_t>(degree); targetLevel > 0L; --targetLevel) {
			for (int64_t leftLevel = targetLevel - 1L, rightLevel = 1L;
				leftLevel > 0L;
				--leftLevel, ++rightLevel) {

				double* resultPtr = out + levelIndex[targetLevel];

				for (double* leftPtr = out + levelIndex[leftLevel]; leftPtr != out + levelIndex[leftLevel + 1]; ++leftPtr) {
					for (double* rightPtr = linearSignature + levelIndex[rightLevel]; rightPtr != linearSignature + levelIndex[rightLevel + 1]; ++rightPtr) {
						*(resultPtr++) += (*leftPtr) * (*rightPtr);
					}
				}
				
			}

			//leftLevel = 0
			double* resultPtr = out + levelIndex[targetLevel];

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

	Point<T> prevPt = path.begin();
	Point<T> nextPt = path.begin();
	++nextPt;

	uint64_t* levelIndex = (uint64_t*)ALIGNED_MALLOC(sizeof(uint64_t) * (degree + 2));
	/*if (!levelIndex) {
		raise std::cerr << "Failed to allocate memory for levelIndex." << std::endl;
		return;
	}*/
	levelIndex[0] = 0UL;
	for (uint64_t i = 1UL; i <= degree + 1UL; i++)
		levelIndex[i] = levelIndex[i - 1UL] * dimension + 1UL;

	linearSignature_(prevPt, nextPt, out, dimension, degree, levelIndex); //Zeroth step

	if (path.length() == 2UL) { ALIGNED_FREE(levelIndex); return; }

	++prevPt;
	++nextPt;

	double* hornerStep = (double*)ALIGNED_MALLOC(sizeof(double) * (levelIndex[degree + 1UL] - levelIndex[degree])); //This will hold intermediary computations
	double* increments = (double*)ALIGNED_MALLOC(sizeof(double) * dimension);

	Point<T> lastPt(path.end());

	for (; nextPt != lastPt; ++prevPt, ++nextPt) {
		for (uint64_t i = 0UL; i < dimension; ++i)
			increments[i] = static_cast<double>(nextPt[i] - prevPt[i]);

		for (int64_t targetLevel = static_cast<int64_t>(degree); targetLevel > 1L; --targetLevel) {

			double oneOverLevel = 1. / static_cast<double>(targetLevel);

			//leftLevel = 0
			//assign z / targetLevel to hornerStep
			for (uint64_t i = 0UL; i < dimension; ++i)
				hornerStep[i] = increments[i] * oneOverLevel;

			for (int64_t leftLevel = 1L, rightLevel = targetLevel - 1L;
				leftLevel < targetLevel - 1L; 
				++leftLevel, --rightLevel) { //for each, add current leftLevel and times by z / rightLevel

				const uint64_t leftLevelSize = levelIndex[leftLevel + 1UL] - levelIndex[leftLevel];
				oneOverLevel = 1. / static_cast<double>(rightLevel);

				//Horner stuff
				//Add
				double* leftPtr1 = out + levelIndex[leftLevel];
				for (uint64_t i = 0UL; i < leftLevelSize; ++i) {
					hornerStep[i] += *(leftPtr1++);
				}

				//Multiply
#ifdef AVX
				double leftOverLevel;
				double* resultPtr = hornerStep + levelIndex[leftLevel + 2UL] - levelIndex[leftLevel + 1UL] - dimension;
				for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr, resultPtr -= dimension) {
					leftOverLevel = (*leftPtr) * oneOverLevel;
					vecMultAssign(resultPtr, increments, leftOverLevel, dimension);
				}
#else
				double leftOverLevel;
				double* resultPtr = hornerStep + levelIndex[leftLevel + 2UL] - levelIndex[leftLevel + 1UL];
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
			double* leftPtr1 = out + levelIndex[targetLevel - 1UL];
			for (uint64_t i = 0UL; i < leftLevelSize; ++i) {
				hornerStep[i] += *(leftPtr1++);
			}

			//Multiply and add, writing straight into out
#ifdef AVX
			double* resultPtr = out + levelIndex[targetLevel + 1] - dimension;
			for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr, resultPtr -= dimension) {
				vecMultAdd(resultPtr, increments, *leftPtr, dimension);
			}
#else
			double* resultPtr = out + levelIndex[targetLevel + 1];
			for (double* leftPtr = hornerStep + leftLevelSize - 1UL; leftPtr != hornerStep - 1UL; --leftPtr) {
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
		uint64_t resultLength = ::polyLength(pathObj.dimension(), degree);
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
			out[i + 1] = static_cast<double>(lastPt[i] - firstPt[i]);
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
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("signature received path of dimension 0"); }

	Path<T> dummyPathObj(nullptr, dimension, length, timeAug, leadLag); //Work with pathObj to capture timeAug, leadLag transformations

	const uint64_t resultLength = ::polyLength(dummyPathObj.dimension(), degree);

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

	//General case and degree = 1 case
	const uint64_t flatPathLength = dummyPathObj.dimension() * dummyPathObj.length();
	T* const dataEnd = path + flatPathLength * batchSize;

	std::function<void(T*, double*)> sigFunc;

	if (degree == 1) {
		sigFunc = [&](T* pathPtr, double* outPtr) {
			Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
			Point<T> firstPt = pathObj.begin();
			Point<T> lastPt = --pathObj.end();
			outPtr[0] = 1.;
			for (uint64_t i = 0; i < pathObj.dimension(); ++i)
				outPtr[i + 1] = static_cast<double>(lastPt[i] - firstPt[i]);
			};
	}
	else {
		if (horner) {
			sigFunc = [&](T* pathPtr, double* outPtr) {
				Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
				signatureHorner_<T>(pathObj, outPtr, degree);
				};
		}
		else {
			sigFunc = [&](T* pathPtr, double* outPtr) {
				Path<T> pathObj(pathPtr, dimension, length, timeAug, leadLag);
				signatureNaive_<T>(pathObj, outPtr, degree);
				};
		}
	}

	T* pathPtr;
	double* outPtr;

	if (parallel) {
		multiThreadedBatch(sigFunc, path, out, batchSize, flatPathLength, resultLength);
	}
	else {
		for (pathPtr = path, outPtr = out;
			pathPtr < dataEnd;
			pathPtr += flatPathLength, outPtr += resultLength) {

			sigFunc(pathPtr, outPtr);
		}
	}
	return;
}