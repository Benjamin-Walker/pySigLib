#include "cppch.h"
#include "cpTensorPoly.h"

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

extern "C" CPSIG_API uint64_t polyLength(uint64_t dimension, uint64_t degree) noexcept {
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
