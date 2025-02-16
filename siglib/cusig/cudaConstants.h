#pragma once
#include "cupch.h"

#ifndef CUDACONSTANTS_H
#define CUDACONSTANTS_H

extern __constant__ uint64_t dimension;
extern __constant__ uint64_t length1;
extern __constant__ uint64_t length2;
extern __constant__ uint64_t dyadicOrder1;
extern __constant__ uint64_t dyadicOrder2;

extern __constant__ double twelth;
extern __constant__ uint64_t dyadicLength1;
extern __constant__ uint64_t dyadicLength2;
extern __constant__ uint64_t numAntiDiag;
extern __constant__ double dyadicFrac;

#endif CUDACONSTANTS