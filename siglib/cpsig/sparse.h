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

template<std::floating_point T>
struct Entry {
    uint64_t col;
    T val;
};

template<std::floating_point T>
class SparseMatrix {
public:
    uint64_t n;
    uint64_t m;
    std::vector<std::vector<Entry<T>>> rows;

    SparseMatrix(uint64_t n_)
        : n(n_), m(n_), rows(n_) {
    }

    SparseMatrix(uint64_t n_, uint64_t m_)
        : n(n_), m(m_), rows(n_) {
    }

    void populate_diagonal() {
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseMatrix.populate_diagonal");
        }
#endif

        for (uint64_t i = 0; i < n; ++i) {
            this->insert_entry(i, i, 1.);
        }
    }

    bool is_nonzero(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseMatrix.is_nonzero");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val != static_cast<T>(0.);
        }
        return false;
    }

    T get(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseMatrix.is_nonzero");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val;
        }
        return static_cast<T>(0.);
    }


    void insert_entry(uint64_t i, uint64_t j, T v) {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseMatrix.insert_entry");
        }
#endif
        rows[i].push_back({ j, v });
    }

    void drop_row(uint64_t i) {
#ifdef _DEBUG
        if (i > n) {
            throw std::out_of_range("i out of range in SparseMatrix.drop_row");
        }
#endif
        rows.erase(rows.begin() + i);
        --n;
    }

    SparseMatrix<T> inverse() const {
        // This assumes matrix is lower triangular with ones on the diagonal
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseMatrix.inverse");
        }
#endif

        SparseMatrix<T> inv(n);

        for (uint64_t i = 0; i < n; ++i) {
            inv.insert_entry(i, i, static_cast<T>(1));
        }

        for (uint64_t i = 0; i < n; ++i) {
            for (const auto& e : rows[i]) {
                uint64_t k = e.col;
                T Lik = e.val;

                if (k >= i) continue;

                inv.add_to_entry(i, k, -Lik);

                for (const auto& ek : inv.rows[k]) {
                    uint64_t j = ek.col;
                    if (j >= k) continue;
                    inv.add_to_entry(i, j, -Lik * ek.val);
                }
            }
        }

        return inv;
    }

    void mul_vec_inplace(T* arr) {
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

    bool operator==(const SparseMatrix<T>& other) const {
        if (n != other.n) return false;

        constexpr T tol = 1e-10;

        for (uint64_t i = 0; i < n; ++i) {
            std::unordered_map<uint64_t, T> row;

            for (const auto& e : rows[i])
                row[e.col] += e.val;

            for (const auto& e : other.rows[i])
                row[e.col] -= e.val;

            for (const auto& [j, v] : row) {
                if (v > tol || v < -tol)
                    return false;
            }
        }

        return true;
    }

    void add_to_entry(uint64_t i, uint64_t j, T v) {
#ifdef _DEBUG
        if (i > n || j > m) {
            throw std::out_of_range("i,j out of range in SparseMatrix.insert_entry");
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
