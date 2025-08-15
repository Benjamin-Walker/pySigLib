/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include "cppch.h"

#if defined(CPSIG_EXPORTS)
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllexport)
	#elif defined (__GNUC__)
		#define CPSIG_API __attribute__((visibility("default")))
	#else
		#define CPSIG_API
	#endif
#else
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllimport)
	#elif defined (__GNUC__)
		#define CPSIG_API 
	#else
		#define CPSIG_API 
	#endif
#endif


extern "C" {

	CPSIG_API int transform_path_float(float* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int transform_path_double(double* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int transform_path_int32(int32_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int transform_path_int64(int64_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;

	CPSIG_API int batch_transform_path_float(float* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_transform_path_double(double* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_transform_path_int32(int32_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_transform_path_int64(int64_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;

	CPSIG_API int transform_path_backprop(double* derivs, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int batch_transform_path_backprop(double* derivs, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;

	CPSIG_API uint64_t sig_length(uint64_t dimension, uint64_t degree) noexcept;
	CPSIG_API int sig_combine(double* sig1, double* sig2, double* out, uint64_t dimension, uint64_t degree) noexcept;
	CPSIG_API int batch_sig_combine(double* sig1, double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;

	CPSIG_API int sig_combine_backprop(double* sig_combined_derivs, double* sig1_deriv, double* sig2_deriv, double* sig1, double* sig2, uint64_t dimension, uint64_t degree) noexcept;
	CPSIG_API int batch_sig_combine_backprop(double* sig_combined_derivs, double* sig1_deriv, double* sig2_deriv, double* sig1, double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;

	CPSIG_API double get_path_element(double* data_ptr, int data_length, int data_dimension, int length_index, int dim_index);

	CPSIG_API int signature_float(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true) noexcept; //bool time_aug = false, bool lead_lag = false, bool horner = true);
	CPSIG_API int signature_double(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true) noexcept;
	CPSIG_API int signature_int32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true) noexcept;
	CPSIG_API int signature_int64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true) noexcept;

	CPSIG_API int batch_signature_float(float* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, int n_jobs = 1) noexcept;
	CPSIG_API int batch_signature_double(double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, int n_jobs = 1) noexcept;
	CPSIG_API int batch_signature_int32(int32_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, int n_jobs = 1) noexcept;
	CPSIG_API int batch_signature_int64(int64_t* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, int n_jobs = 1) noexcept;

	CPSIG_API int sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	CPSIG_API int sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;

	CPSIG_API int batch_sig_backprop_float(float* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_sig_backprop_double(double* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_sig_backprop_int32(int32_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	CPSIG_API int batch_sig_backprop_int64(int64_t* path, double* out, double* sig_derivs, double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;

	CPSIG_API int sig_kernel(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	CPSIG_API int batch_sig_kernel(double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs = 1, bool return_grid = false) noexcept;

	CPSIG_API int sig_kernel_backprop(double* gram, double* out, double deriv, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2) noexcept;
	CPSIG_API int batch_sig_kernel_backprop(double* gram, double* out, double* derivs, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, int n_jobs = 1) noexcept;

}


