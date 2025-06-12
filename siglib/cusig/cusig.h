#pragma once

#ifdef CUSIG_EXPORTS
#ifdef _WIN32
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API
#endif
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" CUSIG_API void cusig_hello_world(const long x);

extern "C" {
	CUSIG_API int sig_kernel_cuda(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept;
	CUSIG_API int batch_sig_kernel_cuda(double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept;
}
