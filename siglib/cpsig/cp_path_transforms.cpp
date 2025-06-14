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

void add_time(double* data_in, double* data_out, const uint64_t dimension, const uint64_t length) {
	uint64_t dataInSize = dimension * length;

	double* in_ptr = data_in;
	double* out_ptr = data_out;
	double* in_end = data_in + dataInSize;
	auto pointSize = sizeof(double) * dimension;
	double time = 0.;
	double step = 1. / static_cast<double>(length);

	while (in_ptr < in_end) {
		memcpy(out_ptr, in_ptr, pointSize);
		in_ptr += dimension;
		out_ptr += dimension;
		(*out_ptr) = time;
		++out_ptr;
		time += step;
	}
}