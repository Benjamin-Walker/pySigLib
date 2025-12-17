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
#include "sparse.h"

typedef std::vector<uint64_t> word;

bool is_lyndon(word w);
std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree);
std::vector<uint64_t> all_lyndon_idx(uint64_t dimension, uint64_t degree);
uint64_t word_to_idx(word w, uint64_t dimension);
word longest_lyndon_suffix_(word w, std::vector<word>& lyndon_words);
word concatenate_words(word& a, word& b);
uint64_t concatenate_idx(uint64_t i, uint64_t j, uint64_t len_j, uint64_t dimension);

struct WordHash {
	std::size_t operator()(const word& w) const noexcept {
		std::size_t h = 0;
		for (uint64_t x : w) {
			h ^= std::hash<uint64_t>{}(x)
				+0x9e3779b97f4a7c15ULL
				+ (h << 6)
				+ (h >> 2);
		}
		return h;
	}
};

template<std::floating_point T>
SparseMatrix<T> lyndon_proj_matrix(
	uint64_t dimension,
	uint64_t degree
) {

	std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
	std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);

	uint64_t n = sig_length(dimension, degree);
	uint64_t m = lyndon_words.size();

	SparseMatrix<T> out(n, m);

	std::unordered_map<word, uint64_t, WordHash> col_idx;

	for (uint64_t i = 0; i < m; ++i) {
		col_idx[lyndon_words[i]] = i;
	}
	
	for (uint64_t i = 0; i < m; ++i) {
		word w = lyndon_words[i];

		if (w.size() == 1) {
			uint64_t iw = word_to_idx(w, dimension);
			out.insert_entry(iw, i, 1.);
		}
		else {
			word v = longest_lyndon_suffix_(w, lyndon_words);
			word u(w.begin(), w.end() - v.size());

			uint64_t jw = col_idx[w];
			uint64_t jv = col_idx[v];
			uint64_t ju = col_idx[u];

			// First term in Lie bracket
			for (uint64_t j = 0; j < n; ++j) {
				if (out.is_nonzero(j, ju)) {
					for (uint64_t k = 0; k < n; ++k) {
						if (out.is_nonzero(k, jv)) {
							uint64_t ic = concatenate_idx(j, k, v.size(), dimension);
							out.add_to_entry(ic, jw, out.get(j, ju) * out.get(k, jv));
						}
					}
				}
			}

			// Second term in Lie bracket
			for (uint64_t j = 0; j < n; ++j) {
				if (out.is_nonzero(j, jv)) {
					for (uint64_t k = 0; k < n; ++k) {
						if (out.is_nonzero(k, ju)) {
							uint64_t ic = concatenate_idx(j, k, u.size(), dimension);
							out.add_to_entry(ic, jw, -out.get(j, jv) * out.get(k, ju));
						}
					}
				}
			}
		}
	}

	for (uint64_t i = 0; i < n; ++i) {
		uint64_t i_ = n - i - 1;
		if (!count(lyndon_idx.begin(), lyndon_idx.end(), i_))
			out.drop_row(i_);
	}
	return out;
}
