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

std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BasisCache>, PairHash> basis_cache;

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

void all_lyndon_words_of_length_n(std::vector<word>& res, uint64_t n, uint64_t dimension) {
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

word longest_lyndon_suffix_(word w, const std::set<word>& lyndon_set) {
	uint64_t n = w.size();
	for (uint64_t i = 1; i < n; ++i) {
		word suffix(w.begin() + i, w.end());
		if (lyndon_set.find(suffix) != lyndon_set.end()) {
			return suffix;
		}
	}
	throw std::runtime_error("Error looking for lyndon suffix");
}

word concatenate_words(word& a, word& b) {
	word c(a);
	c.insert(c.end(), b.begin(), b.end());
	return c;
}

uint64_t concatenate_idx(uint64_t i, uint64_t j, uint64_t len_j, uint64_t dimension) {
	// If i and j correspond to word_to_idx(a) and word_to_idx(b),
	// then this function outputs word_to_idx(c) where c is the
	// concatenation of a and b.
	uint64_t idx = i;
	idx *= ::power(dimension, len_j);
	idx += j;
	return idx;

}

SparseIntMatrix lyndon_proj_matrix(
	const std::vector<word>& lyndon_words,
	std::vector<uint64_t> lyndon_idx, // copy here is intentional
	uint64_t dimension,
	uint64_t degree
) {
	std::set<word> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	uint64_t n = sig_length(dimension, degree);
	uint64_t m = lyndon_words.size();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; i++)
		level_index[i] = level_index[i - 1] * dimension + 1;

	SparseIntMatrix out(n, m);

	std::unordered_map<word, uint64_t, WordHash> col_idx;

	for (uint64_t i = 0; i < m; ++i) {
		col_idx[lyndon_words[i]] = i;
	}

	for (uint64_t i = 0; i < m; ++i) {
		word w = lyndon_words[i];

		if (w.size() == 1) {
			out.insert_entry(w[0] + 1, i, 1);
		}
		else {
			word v = longest_lyndon_suffix_(w, lyndon_set);
			word u(w.begin(), w.end() - v.size());

			uint64_t jw = col_idx[w];
			uint64_t jv = col_idx[v];
			uint64_t ju = col_idx[u];

			uint64_t v_start = level_index[v.size()];
			uint64_t v_end = level_index[v.size() + 1];

			uint64_t u_start = level_index[u.size()];
			uint64_t u_end = level_index[u.size() + 1];

			// First term in Lie bracket
			for (uint64_t j = u_start; j < u_end; ++j) {
				int val1 = out.get(j, ju);
				if (val1) {
					for (uint64_t k = v_start; k < v_end; ++k) {
						int val2 = out.get(k, jv);
						if (val2) {
							uint64_t ic = concatenate_idx(j, k, v.size(), dimension);
							out.add_to_entry(ic, jw, val1 * val2);
						}
					}
				}
			}

			// Second term in Lie bracket
			for (uint64_t j = v_start; j < v_end; ++j) {
				int val1 = out.get(j, jv);
				if (val1) {
					for (uint64_t k = u_start; k < u_end; ++k) {
						int val2 = out.get(k, ju);
						if (val2) {
							uint64_t ic = concatenate_idx(j, k, u.size(), dimension);
							out.add_to_entry(ic, jw, -val1 * val2);
						}
					}
				}
			}
		}
	}

	for (uint64_t i = n - 1; i > 0; --i) {
		if (lyndon_idx.back() != i) {
			out.drop_row(i);
		}
		else {
			lyndon_idx.pop_back();
		}
	}
    out.drop_row(0);
	return out;
}

void set_basis_cache(uint64_t dimension, uint64_t degree) {
	std::pair<uint64_t, uint64_t> key(dimension, degree);

	auto it = basis_cache.find(key);
	if (it == basis_cache.end()) {

		std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
		std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
		SparseIntMatrix p = lyndon_proj_matrix(lyndon_words, lyndon_idx, dimension, degree);

		auto basis_obj = std::make_unique<BasisCache>(
			std::move(lyndon_words),
			std::move(lyndon_idx),
			std::move(p),
			std::move(p.inverse())
		);
		basis_cache.emplace(key, std::move(basis_obj));
	}
}

const BasisCache& get_basis_cache(uint64_t dimension, uint64_t degree) {
	std::pair<uint64_t, uint64_t> key(dimension, degree);

	auto it = basis_cache.find(key);
	if (it == basis_cache.end()) {
		throw std::runtime_error("Could not find basis cache");
	}
	return *(it->second);
}

extern "C" {

	CPSIG_API int prepare_log_sig(uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(set_basis_cache(dimension, degree));
	}

	CPSIG_API void reset_log_sig() noexcept {
		basis_cache.clear();
	}

}
