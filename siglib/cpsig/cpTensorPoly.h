#pragma once
#include "cppch.h"
#include "cpsig.h"

// Calculate power
// Return 0 on error (integer overflow)
uint64_t power(uint64_t base, uint64_t exp) noexcept;
