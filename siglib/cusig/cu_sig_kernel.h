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
#include "cupch.h"

__global__ void goursat_pde(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* gram
);

__device__ void goursat_pde_32(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* diagonals,
	double* gram,
	uint64_t iteration,
	int num_threads
);

void sig_kernel_cuda_(
	double* gram,
	double* out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_
);