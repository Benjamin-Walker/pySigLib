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
	if (n == 0)
		return false;
	if (n == 1)
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

std::vector<word> all_lyndon_words_of_length_n(std::vector<word>& res, uint64_t n, uint64_t dimension) {
	word w;
	w.push_back(0);
  
	while (!w.empty())
	{
		uint64_t m = w.size();
		if (m == n)
			res.push_back(w);

		while (w.size() < n)
			w.push_back(w[w.size() - m]);

		while (!w.empty() && w.back() == dimension - 1)
			w.pop_back();

		if (!w.empty())
			++w.back();
	}
	return res;
}

std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree) {
	std::vector<word> res;
	for (uint64_t n = 1; n <= degree; ++n)
		all_lyndon_words_of_length_n(res, n, dimension);
	return res;
}

uint64_t word_to_idx(word w, uint64_t dimension) {
	if (!w.size())
		return 0;

	uint64_t idx = 0;
	for (uint64_t i : w) {
		idx = idx * dimension + (i + 1);
	}
	return idx;
}

std::vector<uint64_t> all_lyndon_idx(uint64_t dimension, uint64_t degree) {
	std::vector<word> words = all_lyndon_words(dimension, degree);
	std::vector<uint64_t> res;
	for (word w : words) {
		res.push_back(word_to_idx(w, dimension));
	}
	return res;
}

