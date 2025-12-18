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

struct Entry {
    uint64_t col;
    int val;
};

class SparseIntMatrix {
public:
    uint64_t n;
    uint64_t m;
    std::vector<std::vector<Entry>> rows;

    SparseIntMatrix()
        : n(0), m(0), rows(0) {
    }

    SparseIntMatrix(uint64_t n_)
        : n(n_), m(n_), rows(n_) {
    }

    SparseIntMatrix(uint64_t n_, uint64_t m_)
        : n(n_), m(m_), rows(n_) {
    }

    SparseIntMatrix(SparseIntMatrix&& other) noexcept {
        n = other.n;
        m = other.m;
        rows.swap(other.rows);
    }

    void resize(uint64_t n_, uint64_t m_) {
        n = n_;
        m = m_;
        rows.resize(n);
    }

    void populate_diagonal() {
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseIntMatrix.populate_diagonal");
        }
#endif

        for (uint64_t i = 0; i < n; ++i) {
            this->insert_entry(i, i, 1);
        }
    }

    bool is_nonzero(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.is_nonzero");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val != 0;
        }
        return false;
    }

    int get(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.is_nonzero");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val;
        }
        return 0;
    }


    void insert_entry(uint64_t i, uint64_t j, int v) {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.insert_entry");
        }
#endif
        rows[i].push_back({ j, v });
    }

    void drop_row(uint64_t i) {
#ifdef _DEBUG
        if (i > n) {
            throw std::out_of_range("i out of range in SparseIntMatrix.drop_row");
        }
#endif
        rows.erase(rows.begin() + i);
        --n;
    }

    SparseIntMatrix inverse() const {
        // This assumes matrix is lower triangular with ones on the diagonal
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseIntMatrix.inverse");
        }
#endif

        SparseIntMatrix inv(n);

        for (uint64_t i = 0; i < n; ++i) {
            inv.insert_entry(i, i, 1);
        }

        for (uint64_t i = 0; i < n; ++i) {
            std::unordered_map<uint64_t, int> row_i;

            row_i[i] = 1;

            for (const auto& e : rows[i]) {
                uint64_t k = e.col;
                int Lik = e.val;
                if (k >= i) continue;

                for (const auto& ek : inv.rows[k]) {
                    uint64_t j = ek.col;
                    if (j >= k) continue;
                    row_i[j] -= Lik * ek.val;
                }

                row_i[k] -= Lik;
            }

            inv.rows[i].clear();
            for (const auto& [j, v] : row_i) {
                if (v != 0) {
                    inv.rows[i].push_back({ j, v });
                }
            }
        }

        return inv;
    }

    SparseIntMatrix transpose() const {
        SparseIntMatrix tr(m, n);

        for (uint64_t i = 0; i < n; ++i) {
            for (const auto& e : rows[i]) {
                tr.rows[e.col].push_back({ i, e.val });
            }
        }

        return tr;
    }

    template<std::floating_point T>
    void mul_vec_inplace(T* arr) const {
        // This assumes matrix is lower triangular with ones on the diagonal
        for (uint64_t i_ = 0; i_ < n; ++i_) {
            uint64_t i = n - i_ - 1;
            for (const auto& e : rows[i]) {
                uint64_t j = e.col;
                if (j < i) {
                    arr[i] += e.val * arr[j];
                }
            }
        }
    }

    bool operator==(const SparseIntMatrix& other) const {
        if (n != other.n) return false;

        for (uint64_t i = 0; i < n; ++i) {
            std::unordered_map<uint64_t, int> row;

            for (const auto& e : rows[i])
                row[e.col] += e.val;

            for (const auto& e : other.rows[i])
                row[e.col] -= e.val;

            for (const auto& [j, v] : row) {
                if (v)
                    return false;
            }
        }

        return true;
    }

    void add_to_entry(uint64_t i, uint64_t j, int v) {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.insert_entry");
        }
#endif
        for (auto& e : rows[i]) {
            if (e.col == j) {
                e.val += v;
                return;
            }
        }
        rows[i].push_back({ j, v });
    }
};
