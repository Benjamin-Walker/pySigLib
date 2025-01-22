#include "cppch.h"
#include "cpsig.h"
#include "cpSignature.h"
#include "macros.h"


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

	CPSIG_API int signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) noexcept {
		SAFE_CALL(signature_<float>(path, out, dimension, length, degree, timeAug, leadLag, horner));
	}

	CPSIG_API int signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) noexcept {
		SAFE_CALL(signature_<double>(path, out, dimension, length, degree, timeAug, leadLag, horner));
	}

	CPSIG_API int signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) noexcept {
		SAFE_CALL(signature_<int32_t>(path, out, dimension, length, degree, timeAug, leadLag, horner));
	}

	CPSIG_API int signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) noexcept {
		SAFE_CALL(signature_<int64_t>(path, out, dimension, length, degree, timeAug, leadLag, horner));
	}

	CPSIG_API int batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) noexcept {
		SAFE_CALL(batchSignature_<float>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel));
	}

	CPSIG_API int batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) noexcept {
		SAFE_CALL(batchSignature_<double>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel));
	}

	CPSIG_API int batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) noexcept {
		SAFE_CALL(batchSignature_<int32_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel));
	}

	CPSIG_API int batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) noexcept {
		SAFE_CALL(batchSignature_<int64_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel));
	}

}
