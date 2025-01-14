#include "cppch.h"
#include "cpTensorPoly.h"

uint64_t power(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0UL) {
        if (exp % 2UL == 1UL) {
            result *= base;
        }
        base *= base;
        exp /= 2UL;
    }
    return result;
}

extern "C" CPSIG_API uint64_t polyLength(uint64_t dimension, uint64_t degree) {
    if (dimension == 0UL) {
        return 1UL;
    }
    else if (dimension == 1UL) {
        return degree + 1UL;
    }
    else {
        return (power(dimension, degree + 1UL) - 1UL) / (dimension - 1UL);
    }
}