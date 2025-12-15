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
#include "words.h"


bool is_lyndon(word w) {
	const uint64_t n = w.size();
	if (n <= 1)
		return true;
	for (uint64_t i = 1; i < n; ++i) {
		if (!std::lexicographical_compare(
			w.begin(), w.end(),
			w.begin() + i, w.end()
		))
			return false;
	}
	return true;
}

void build_words_(uint64_t dimension, uint64_t degree, word& w, std::vector<word>& res) {
	if (w.size() > degree) {
		return;
	}
	if (is_lyndon(w))
		res.push_back(w);
	for (uint64_t j = 0; j < dimension; ++j) {
		w.push_back(j);
		build_words_(dimension, degree, w, res);
		w.pop_back();
	}
}

std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree) {
	std::vector<word> res;
	word w;
	build_words_(dimension, degree, w, res);
	std::sort(res.begin(), res.end(),
		[](const word& a, const word& b) {
			if (a.size() != b.size())
				return a.size() < b.size(); // length first
			return a < b;                   // lexicographic
		}
	);
	return res;
}
