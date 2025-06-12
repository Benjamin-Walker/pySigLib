#pragma once
#include "cupch.h"

#ifndef CUDACONSTANTS_H
#define CUDACONSTANTS_H

extern __constant__ uint64_t dimension;
extern __constant__ uint64_t length1;
extern __constant__ uint64_t length2;
extern __constant__ uint64_t dyadic_order_1;
extern __constant__ uint64_t dyadic_order_2;

extern __constant__ double twelth;
extern __constant__ uint64_t dyadic_length_1;
extern __constant__ uint64_t dyadic_length_2;
extern __constant__ uint64_t num_anti_diag;
extern __constant__ double dyadic_frac;
extern __constant__ uint64_t gram_length;

#endif