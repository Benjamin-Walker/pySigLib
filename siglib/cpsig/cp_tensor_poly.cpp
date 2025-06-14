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
#include "cp_tensor_poly.h"

uint64_t power(uint64_t base, uint64_t exp) noexcept {
    uint64_t result = 1;
    while (exp > 0UL) {
        if (exp % 2UL == 1UL) {
            const auto _res = result * base;
            if (_res < result)
                return 0UL; // overflow
            result = _res;
        }
        const auto _base = base * base;
        if (_base < base)
            return 0UL; // overflow
        base = _base;
        exp /= 2UL;
    }
    return result;
}

extern "C" CPSIG_API uint64_t poly_length(uint64_t dimension, uint64_t degree) noexcept {
    if (dimension == 0UL) {
        return 1UL;
    }
    else if (dimension == 1UL) {
        return degree + 1UL;
    }
    else {
        const auto pwr = power(dimension, degree + 1UL);
        if (pwr)
            return (pwr - 1UL) / (dimension - 1UL);
        else
            return 0UL; // overflow
    }
}
