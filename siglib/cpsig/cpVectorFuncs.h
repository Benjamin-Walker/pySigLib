#pragma once
#include "cppch.h"

__forceinline void vecMultAdd(double* out, double* other, double scalar, uint64_t size)
{
	uint64_t firstLoopItrs = size / 4;
	uint64_t secondLoopItrs = size % 4;

	__m256d a, b;
	__m256d scalar_ = _mm256_set1_pd(scalar);
	double* otherPtr = other, * outPtr = out;
	double* outEnd = out + size;

	double* firstLoopEnd = outEnd - secondLoopItrs;

	for (; outPtr != firstLoopEnd; otherPtr += 4, outPtr += 4) {
		a = _mm256_loadu_pd(otherPtr);
		a = _mm256_mul_pd(a, scalar_);
		b = _mm256_loadu_pd(outPtr);
		b = _mm256_add_pd(a, b);
		_mm256_storeu_pd(outPtr, b);
	}
	__m128d scalar_128 = _mm_set1_pd(scalar);
	for (; outPtr != outEnd; ++otherPtr, ++outPtr) {
		__m128d other_val = _mm_load_sd(otherPtr); // Load one double from otherPtr
		other_val = _mm_mul_sd(other_val, scalar_128); // Multiply
		__m128d out_val = _mm_load_sd(outPtr); // Load one double from outPtr
		__m128d result = _mm_add_sd(other_val, out_val); // Add
		_mm_store_sd(outPtr, result); // Store back the result
	}
}

__forceinline void vecMultAssign(double* out, double* other, double scalar, uint64_t size) {
	uint64_t firstLoopItrs = size / 4;
	uint64_t secondLoopItrs = size % 4;

	__m256d a;
	__m256d scalar_ = _mm256_set1_pd(scalar);
	double* otherPtr = other, * outPtr = out;
	double* outEnd = out + size;

	double* firstLoopEnd = outEnd - secondLoopItrs;

	for (; outPtr != firstLoopEnd; otherPtr += 4, outPtr += 4) {
		a = _mm256_loadu_pd(otherPtr);
		a = _mm256_mul_pd(a, scalar_);
		_mm256_storeu_pd(outPtr, a);
	}
	__m128d scalar_128 = _mm_set1_pd(scalar);
	for (; outPtr != outEnd; ++otherPtr, ++outPtr) {
		__m128d other_val = _mm_load_sd(otherPtr); // Load one double from otherPtr
		other_val = _mm_mul_sd(other_val, scalar_128); // Multiply
		_mm_store_sd(outPtr, other_val); // Store back the result
	}
}
