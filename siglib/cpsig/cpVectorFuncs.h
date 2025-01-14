#pragma once
#include "cppch.h"

__forceinline void vecMultAdd(double* out, double* other, double scalar, uint64_t size)
{
	uint64_t firstLoopRemainder = size % 4UL;

	__m256d a, b;
	__m256d scalar_256 = _mm256_set1_pd(scalar);

	__m128d c, d;
	__m128d scalar_128 = _mm_set1_pd(scalar);

	double* otherPtr = other, * outPtr = out;
	double* outEnd = out + size;

	double* firstLoopEnd = outEnd - firstLoopRemainder;

	for (; outPtr != firstLoopEnd; otherPtr += 4, outPtr += 4) {
		a = _mm256_loadu_pd(otherPtr);
		a = _mm256_mul_pd(a, scalar_256);
		b = _mm256_loadu_pd(outPtr);
		b = _mm256_add_pd(a, b);
		_mm256_storeu_pd(outPtr, b);
	}
	if (size & 2UL) {
		c = _mm_loadu_pd(otherPtr);
		c = _mm_mul_pd(c, scalar_128);
		d = _mm_load_pd(outPtr);
		d = _mm_add_pd(c, d);
		_mm_storeu_pd(outPtr, d);
		otherPtr += 2;
		outPtr += 2;
	}
	if (size & 1UL) { //For some reason intrinsics are quicker than a normal loop here
		c = _mm_load_sd(otherPtr);
		c = _mm_mul_sd(c, scalar_128);
		d = _mm_load_sd(outPtr);
		d = _mm_add_sd(c, d);
		_mm_store_sd(outPtr, d);
	}
}

__forceinline void vecMultAssign(double* out, double* other, double scalar, uint64_t size) {
	uint64_t firstLoopRemainder = size % 4UL;

	__m256d a;
	__m256d scalar_ = _mm256_set1_pd(scalar);

	__m128d c;
	__m128d scalar_128 = _mm_set1_pd(scalar);

	double* otherPtr = other, * outPtr = out;
	double* outEnd = out + size;

	double* firstLoopEnd = outEnd - firstLoopRemainder;

	for (; outPtr != firstLoopEnd; otherPtr += 4, outPtr += 4) {
		a = _mm256_loadu_pd(otherPtr);
		a = _mm256_mul_pd(a, scalar_);
		_mm256_storeu_pd(outPtr, a);
	}
	if(size & 2UL) {
		c = _mm_loadu_pd(otherPtr);
		c = _mm_mul_pd(c, scalar_128);
		_mm_storeu_pd(outPtr, c);
		otherPtr += 2;
		outPtr += 2;
	}
	if (size & 1UL) { //For some reason intrinsics are quicker than a normal loop here
		c = _mm_load_sd(otherPtr);
		c = _mm_mul_sd(c, scalar_128);
		_mm_store_sd(outPtr, c);
	}
}
