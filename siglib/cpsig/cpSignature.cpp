#include "cppch.h"
#include "cpTensorPoly.h"
#include "cpSignature.h"

void signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<float>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<double>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<int32_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<int64_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
	batchSignature_<float>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
}

void batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
	batchSignature_<double>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
}

void batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
	batchSignature_<int32_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
}

void batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
	batchSignature_<int64_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
}