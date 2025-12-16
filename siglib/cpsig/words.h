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
#include "cp_tensor_poly.h"

typedef std::vector<uint64_t> word;

bool is_lyndon(word w);
std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree);
std::vector<uint64_t> all_lyndon_idx(uint64_t dimension, uint64_t degree);
