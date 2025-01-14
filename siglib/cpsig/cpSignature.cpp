#include "cppch.h"
#include "cpsig.h"
#include "cpPath.h"
#include "cpVectorFuncs.h"
#include "multithreading.h"
#include "cpTensorPoly.h"
#include "cpSignature.h"


template<typename T>
PointImpl<T>* Path<T>::pointImplFactory(uint64_t index) const {
	if (!_timeAug && !_leadLag)
		return new PointImpl(this, index);
	else if (_timeAug && !_leadLag)
		return new PointImplTimeAug(this, index);
	else if (!_timeAug && _leadLag)
		return new PointImplLeadLag(this, index);
	else
		return new PointImplTimeAugLeadLag(this, index);
}

extern "C" {

	__declspec(dllexport) void signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<float>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	__declspec(dllexport) void signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<double>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	__declspec(dllexport) void signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<int32_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	__declspec(dllexport) void signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<int64_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	__declspec(dllexport) void batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<float>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	__declspec(dllexport) void batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<double>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	__declspec(dllexport) void batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<int32_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	__declspec(dllexport) void batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<int64_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

}
