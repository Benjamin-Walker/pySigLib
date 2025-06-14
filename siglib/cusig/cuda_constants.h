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

#ifndef CUDACONSTANTS_H
#define CUDACONSTANTS_H

extern __constant__ uint64_t dimension;
extern __constant__ uint64_t length1;
extern __constant__ uint64_t length2;
extern __constant__ uint64_t dyadic_order_1;
extern __constant__ uint64_t dyadic_order_2;

extern __constant__ double twelth;
extern __constant__ uint64_t dyadic_length_1;
extern __constant__ uint64_t dyadic_length_2;
extern __constant__ uint64_t num_anti_diag;
extern __constant__ double dyadic_frac;
extern __constant__ uint64_t gram_length;

#endif