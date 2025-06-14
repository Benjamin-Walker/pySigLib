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

#include "cppch.h"
#include "cpsig.h"
#include <iostream>

#include "cp_path.h"
#include "cp_tensor_poly.h"


double get_path_element(double* data_ptr, int data_length, int data_dimension, int length_index, int dim_index) {
	Path<double> path(data_ptr, static_cast<uint64_t>(data_dimension), static_cast<uint64_t>(data_length));
	return path[static_cast<uint64_t>(length_index)][static_cast<uint64_t>(dim_index)];
}
